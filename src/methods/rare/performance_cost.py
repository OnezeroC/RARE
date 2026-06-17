from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.data.adaptors.response_records import build_response_feature_bundle
from src.models.global_local_router import (
    encode_queries_gpu,
    infer_factor_features,
    infer_logits,
    local_residual_rectify,
    train_global_backbone,
)
from src.models.expert_fusion_router import gpu_weighted_knn_scores
from src.methods.rare.performance import (
    _build_override_features,
    _candidate_list_hit_rate,
    _candidate_prob_matrix,
    _response_candidate_features,
    augment_with_retrieval_features,
    knn_scores_leave_self,
    local_delta_features,
    local_delta_labels,
    retrieval_knn_chunk_size,
)
from src.methods.rare.response_rerank import union_shortlists
from src.config.paths import artifacts_results_root
from src.evaluation.performance_shared import model_usage_stats
from src.shared import (
    EMBEDDING_MODEL,
    LLMROUTERBENCH_ROOT,
    ROOT,
    infer_risk,
    set_seed,
    split_indices,
    stage1_gate_features,
    standardize_fit,
    train_risk_gate,
)


LLMRB_ROOT = LLMROUTERBENCH_ROOT
CONFIG_PATH = LLMRB_ROOT / "config" / "baseline_config_performance_cost.yaml"
BASE_COST_RESULT_JSON = artifacts_results_root() / "result_cost_suite_llmrouterbench.json"
RESULT_JSON = artifacts_results_root() / "result_rare_performance_cost.json"
ALPHAS = [0.0, 0.25, 0.39, 0.53, 0.8, 1.0]
TRAIN_RATIO = 0.7


def resolve_loader():
    sys.path.insert(0, str(LLMRB_ROOT))
    from baselines.data_loader import BaselineDataLoader

    cfg = yaml.safe_load(CONFIG_PATH.read_text())["baseline"]
    resolved = dict(cfg)
    resolved["results_dir"] = str(LLMRB_ROOT / cfg["results_dir"])
    loader = BaselineDataLoader(config=resolved)
    return loader, resolved


def split_records_by_dataset_then_prompt(
    records: list[Any],
    *,
    train_ratio: float,
    random_seed: int,
) -> tuple[list[Any], list[Any]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    rng = random.Random(random_seed)
    dataset_groups: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        dataset_groups[str(record.dataset_id)].append(record)

    train_records: list[Any] = []
    test_records: list[Any] = []
    for dataset_id in sorted(dataset_groups):
        dataset_records = dataset_groups[dataset_id]
        prompt_to_records: dict[str, list[Any]] = defaultdict(list)
        for record in dataset_records:
            prompt_to_records[str(record.prompt)].append(record)

        unique_prompts = sorted(
            prompt_to_records,
            key=lambda prompt: min(int(r.record_index) for r in prompt_to_records[prompt]),
        )
        n_train = int(len(unique_prompts) * train_ratio)
        prompt_indices = list(range(len(unique_prompts)))
        rng.shuffle(prompt_indices)
        train_idx = set(prompt_indices[:n_train])

        for idx, prompt in enumerate(unique_prompts):
            if idx in train_idx:
                train_records.extend(prompt_to_records[prompt])
            else:
                test_records.extend(prompt_to_records[prompt])

    return train_records, test_records


def build_payload(
    train_records: list[Any],
    test_records: list[Any],
    models: list[str],
) -> dict[str, Any]:
    model_to_idx = {name: idx for idx, name in enumerate(models)}

    def records_to_payload(records: list[Any]) -> tuple[np.ndarray, np.ndarray, list[str], dict[int, dict[str, Any]]]:
        grouped: dict[tuple[str, str, int], dict[str, Any]] = {}
        for record in records:
            ds = str(record.dataset_id).lower()
            split = str(record.split)
            record_index = int(record.record_index)
            key = (ds, split, record_index)
            if key not in grouped:
                grouped[key] = {
                    "dataset": ds,
                    "split": split,
                    "index": record_index,
                    "query": str(record.prompt or record.origin_query or ""),
                    "scores": {},
                    "costs": {},
                }
            grouped[key]["scores"][str(record.model_name)] = float(record.score)
            grouped[key]["costs"][str(record.model_name)] = float(record.cost or 0.0)

        ordered_keys = sorted(grouped.keys(), key=lambda x: (x[0], x[2], x[1]))
        perf = np.zeros((len(ordered_keys), len(models)), dtype=np.float32)
        cost = np.zeros((len(ordered_keys), len(models)), dtype=np.float32)
        queries: list[str] = []
        meta: dict[int, dict[str, Any]] = {}
        for row_idx, key in enumerate(ordered_keys):
            row = grouped[key]
            queries.append(row["query"])
            meta[row_idx] = {
                "dataset": row["dataset"],
                "index": row["index"],
                "split": row["split"],
            }
            for model_name, score in row["scores"].items():
                col = model_to_idx[model_name]
                perf[row_idx, col] = score
                cost[row_idx, col] = row["costs"].get(model_name, 0.0)
        return perf, cost, queries, meta

    train_perf, train_cost, train_queries, train_meta = records_to_payload(train_records)
    test_perf, test_cost, test_queries, test_meta = records_to_payload(test_records)
    return {
        "models": models,
        "train_perf": train_perf,
        "train_cost": train_cost,
        "train_queries": train_queries,
        "train_meta": train_meta,
        "test_perf": test_perf,
        "test_cost": test_cost,
        "test_queries": test_queries,
        "test_meta": test_meta,
    }


def model_reference_table_from_records(test_records: list[Any], models: list[str]) -> dict[str, Any]:
    by_model_scores: dict[str, list[float]] = defaultdict(list)
    by_model_costs: dict[str, list[float]] = defaultdict(list)
    by_question: dict[tuple[str, str, int], dict[str, float]] = defaultdict(dict)
    by_model_dataset_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for record in test_records:
        ds = str(record.dataset_id).lower()
        split = str(record.split)
        record_index = int(record.record_index)
        model_name = str(record.model_name)
        score = float(record.score)
        cost = float(record.cost or 0.0)
        by_model_scores[model_name].append(score)
        by_model_costs[model_name].append(cost)
        by_model_dataset_scores[model_name][ds].append(score)
        by_question[(ds, split, record_index)][model_name] = score

    per_model = {}
    for model_name in models:
        scores = by_model_scores.get(model_name, [])
        costs = by_model_costs.get(model_name, [])
        dataset_map = by_model_dataset_scores.get(model_name, {})
        dataset_avg = float(np.mean([
            sum(ds_scores) / len(ds_scores)
            for ds_scores in dataset_map.values()
            if ds_scores
        ])) if dataset_map else 0.0
        per_model[model_name] = {
            "dataset_avg": dataset_avg,
            "sample_avg": float(sum(scores) / len(scores)) if scores else 0.0,
            "avg_cost": float(sum(costs) / len(costs)) if costs else 0.0,
            "n_samples": len(scores),
        }

    best_single_dataset = max(models, key=lambda model: per_model[model]["dataset_avg"])
    best_single_sample = max(models, key=lambda model: per_model[model]["sample_avg"])
    gpt5_key = "gpt-5" if "gpt-5" in per_model else best_single_dataset
    oracle_sample_avg = float(np.mean([max(scores.values()) for scores in by_question.values()]))
    random_router_sample_avg = float(np.mean([sum(scores.values()) / len(scores) for scores in by_question.values()]))
    return {
        "per_model": per_model,
        "best_single_dataset_model": best_single_dataset,
        "best_single_dataset_avg": per_model[best_single_dataset]["dataset_avg"],
        "best_single_dataset_cost": per_model[best_single_dataset]["avg_cost"],
        "best_single_sample_model": best_single_sample,
        "best_single_sample_avg": per_model[best_single_sample]["sample_avg"],
        "best_single_sample_cost": per_model[best_single_sample]["avg_cost"],
        "gpt5_reference_model": gpt5_key,
        "gpt5_reference_dataset_avg": per_model[gpt5_key]["dataset_avg"],
        "gpt5_reference_sample_avg": per_model[gpt5_key]["sample_avg"],
        "gpt5_reference_cost": per_model[gpt5_key]["avg_cost"],
        "oracle_sample_avg": oracle_sample_avg,
        "random_router_sample_avg": random_router_sample_avg,
    }


def load_official_split(split_seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    loader, resolved_config = resolve_loader()
    all_records = loader.load_all_records()
    train_records, test_records = split_records_by_dataset_then_prompt(
        records=all_records,
        train_ratio=TRAIN_RATIO,
        random_seed=split_seed,
    )

    configured_models = resolved_config.get("filters", {}).get("models") or []
    observed_models = {str(record.model_name) for record in all_records}
    models = [model_name for model_name in configured_models if model_name in observed_models]
    if not models:
        models = sorted(observed_models)

    payload = build_payload(train_records, test_records, models)
    refs = model_reference_table_from_records(test_records, models)
    return payload, refs


def normalize_perf_scores(perf_scores: np.ndarray) -> np.ndarray:
    row_min = perf_scores.min(axis=1, keepdims=True)
    row_max = perf_scores.max(axis=1, keepdims=True)
    return (perf_scores - row_min) / np.clip(row_max - row_min, 1e-8, None)


def cost_score_matrix(cost_matrix: np.ndarray) -> np.ndarray:
    row_max = np.clip(cost_matrix.max(axis=1, keepdims=True), 1e-8, None)
    return 1.0 - (cost_matrix / row_max)


def evaluate_selected(
    selected_idx: np.ndarray,
    test_perf: np.ndarray,
    test_cost: np.ndarray,
    test_meta: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    correct = {}
    total = {}
    for row_idx, model_idx in enumerate(selected_idx):
        ds = test_meta[row_idx]["dataset"]
        total[ds] = total.get(ds, 0) + 1
        if test_perf[row_idx, int(model_idx)] > 0:
            correct[ds] = correct.get(ds, 0) + 1
    per_dataset = {ds: correct.get(ds, 0) / total[ds] for ds in sorted(total)}
    return {
        "sample_avg": float(test_perf[np.arange(len(test_perf)), selected_idx].mean()),
        "dataset_avg": float(np.mean(list(per_dataset.values()))),
        "avg_cost": float(test_cost[np.arange(len(test_cost)), selected_idx].mean()),
        "total_cost": float(test_cost[np.arange(len(test_cost)), selected_idx].sum()),
        "per_dataset": per_dataset,
    }


def alpha_sweep(
    perf_scores: np.ndarray,
    test_perf: np.ndarray,
    test_cost: np.ndarray,
    test_meta: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    perf_norm = normalize_perf_scores(perf_scores.astype(np.float32))
    cost_norm = cost_score_matrix(test_cost)
    curve = {}
    for alpha in ALPHAS:
        utility = alpha * perf_norm + (1.0 - alpha) * cost_norm
        selected_idx = np.argmax(utility, axis=1)
        curve[str(alpha)] = evaluate_selected(selected_idx, test_perf, test_cost, test_meta) | {
            "selected_indices": selected_idx.tolist(),
        }
    return curve


def summarize_curve(curve: dict[str, Any], refs: dict[str, Any]) -> dict[str, Any]:
    points = [curve[str(alpha)] | {"alpha": alpha} for alpha in ALPHAS]
    best_acc_point = max(points, key=lambda row: row["dataset_avg"])
    feasible = [
        row for row in points
        if row["dataset_avg"] >= refs["gpt5_reference_dataset_avg"]
    ]
    best_costsave_point = min(feasible, key=lambda row: row["avg_cost"]) if feasible else None
    cost_save_gpt5 = 0.0
    if best_costsave_point is not None and refs["gpt5_reference_cost"] > 0:
        cost_save_gpt5 = (
            refs["gpt5_reference_cost"] - best_costsave_point["avg_cost"]
        ) / refs["gpt5_reference_cost"]
    return {
        "best_accuracy_alpha": best_acc_point["alpha"],
        "best_accuracy_dataset_avg": best_acc_point["dataset_avg"],
        "best_accuracy_sample_avg": best_acc_point["sample_avg"],
        "best_accuracy_avg_cost": best_acc_point["avg_cost"],
        "perf_gain_vs_gpt5_dataset": best_acc_point["dataset_avg"] - refs["gpt5_reference_dataset_avg"],
        "perf_gain_vs_best_single_dataset": best_acc_point["dataset_avg"] - refs["best_single_dataset_avg"],
        "perf_gain_vs_gpt5_sample": best_acc_point["sample_avg"] - refs["gpt5_reference_sample_avg"],
        "perf_gain_vs_best_single_sample": best_acc_point["sample_avg"] - refs["best_single_sample_avg"],
        "cost_save_vs_gpt5": cost_save_gpt5,
    }


def default_result_path(split_seed: int) -> Path:
    if split_seed == 42:
        return RESULT_JSON
    return artifacts_results_root() / f"result_rare_performance_cost_seed{split_seed}.json"


def response_override_result_path(split_seed: int) -> Path:
    return artifacts_results_root() / f"result_rare_performance_cost_seed{split_seed}_response_override.json"


def backbone_loss_weights_from_args(args: argparse.Namespace) -> dict[str, float]:
    presets: dict[str, dict[str, float]] = {
        "baseline": {
            "bce": 0.5,
            "listwise": 1.0,
            "base_listwise": 0.25,
            "margin": 0.35,
            "entropy": 0.04,
            "balance": 0.05,
            "ortho": 0.03,
        },
        "margin_heavy": {
            "bce": 0.35,
            "listwise": 1.0,
            "base_listwise": 0.2,
            "margin": 0.6,
            "entropy": 0.04,
            "balance": 0.05,
            "ortho": 0.03,
        },
        "listwise_onlyish": {
            "bce": 0.0,
            "listwise": 1.6,
            "base_listwise": 0.5,
            "margin": 0.1,
            "entropy": 0.04,
            "balance": 0.05,
            "ortho": 0.03,
        },
    }
    return dict(presets[args.backbone_loss_preset])


def utility_profile_matrix(
    perf_matrix: np.ndarray,
    cost_matrix: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    perf_norm = normalize_perf_scores(perf_matrix.astype(np.float32))
    cost_norm = cost_score_matrix(cost_matrix.astype(np.float32))
    return (alpha * perf_norm + (1.0 - alpha) * cost_norm).astype(np.float32)


def augment_with_cost_aware_retrieval_features(
    *,
    args: argparse.Namespace,
    x_train: np.ndarray,
    train_perf: np.ndarray,
    train_cost: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if args.retrieval_feature_preset == "perf_only":
        xtr_ret, xval_ret, xtest_ret, ret_cfg = augment_with_retrieval_features(
            x_train,
            train_perf,
            x_val,
            x_test,
        )
        return xtr_ret, xval_ret, xtest_ret, ret_cfg | {"feature_preset": args.retrieval_feature_preset}

    if args.retrieval_feature_preset == "perf_cost_concat":
        profile_matrix = np.concatenate(
            [train_perf.astype(np.float32), cost_score_matrix(train_cost.astype(np.float32))],
            axis=1,
        ).astype(np.float32)
        profile_name = "perf_cost_concat"
    elif args.retrieval_feature_preset == "utility_0.8":
        profile_matrix = utility_profile_matrix(train_perf, train_cost, alpha=0.8)
        profile_name = "utility_0.8"
    else:
        raise ValueError(f"Unsupported retrieval feature preset: {args.retrieval_feature_preset}")

    best = None
    best_payload = None
    chunk_size = retrieval_knn_chunk_size()
    for k in [16, 24, 32]:
        for tau in [0.03, 0.05]:
            tr_scores = knn_scores_leave_self(
                x_train,
                profile_matrix,
                k=k,
                tau=tau,
                chunk_size=chunk_size,
            )
            val_scores = gpu_weighted_knn_scores(
                x_train=x_train,
                train_matrix=profile_matrix,
                x_query=x_val,
                k=k,
                tau=tau,
                chunk_size=chunk_size,
            )
            score = float(val_scores.max(axis=1).mean())
            if best is None or score > best:
                best = score
                test_scores = gpu_weighted_knn_scores(
                    x_train=x_train,
                    train_matrix=profile_matrix,
                    x_query=x_test,
                    k=k,
                    tau=tau,
                    chunk_size=chunk_size,
                )
                best_payload = (
                    np.concatenate([x_train, tr_scores], axis=1).astype(np.float32),
                    np.concatenate([x_val, val_scores], axis=1).astype(np.float32),
                    np.concatenate([x_test, test_scores], axis=1).astype(np.float32),
                    {
                        "k": k,
                        "tau": tau,
                        "neighbor_conf_mean": score,
                        "feature_preset": profile_name,
                        "profile_dim": int(profile_matrix.shape[1]),
                    },
                )
    assert best_payload is not None
    return best_payload


def gate_score_stats(scores: np.ndarray) -> dict[str, Any]:
    if scores.size == 0:
        return {"count": 0}
    quantile_points = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    quantiles = {
        str(q): float(v)
        for q, v in zip(quantile_points, np.quantile(scores, quantile_points), strict=True)
    }
    rounded = np.round(scores, decimals=6)
    return {
        "count": int(scores.size),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "zero_rate": float(np.mean(np.isclose(scores, 0.0))),
        "distinct_rounded": int(np.unique(rounded).size),
        "quantiles": quantiles,
    }


def gate_threshold_quantile_from_args(args: argparse.Namespace) -> float:
    if args.gate_target_rate is not None:
        return float(np.clip(1.0 - float(args.gate_target_rate), 0.0, 1.0))
    return float(np.clip(args.gate_threshold_quantile, 0.0, 1.0))


def quantile_trigger(
    cal_scores: np.ndarray,
    test_scores: np.ndarray,
    *,
    threshold_quantile: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    threshold = float(np.quantile(cal_scores, threshold_quantile))
    return cal_scores >= threshold, test_scores >= threshold, threshold


def topk_trigger(
    scores: np.ndarray,
    eligible_mask: np.ndarray,
    *,
    target_rate: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    selected = np.zeros(len(scores), dtype=bool)
    eligible_idx = np.flatnonzero(eligible_mask)
    requested_k = int(round(float(target_rate) * len(scores)))
    if requested_k <= 0 or eligible_idx.size == 0:
        return selected, {
            "requested_k": requested_k,
            "applied_k": 0,
            "eligible_count": int(eligible_idx.size),
            "selected_threshold": None,
        }

    applied_k = min(requested_k, int(eligible_idx.size))
    if applied_k == int(eligible_idx.size):
        chosen = eligible_idx
    else:
        chosen_local = np.argpartition(scores[eligible_idx], -applied_k)[-applied_k:]
        chosen = eligible_idx[chosen_local]
    selected[chosen] = True
    selected_threshold = float(scores[chosen].min()) if chosen.size > 0 else None
    return selected, {
        "requested_k": requested_k,
        "applied_k": applied_k,
        "eligible_count": int(eligible_idx.size),
        "selected_threshold": selected_threshold,
    }


def resolve_gate_trigger(
    *,
    args: argparse.Namespace,
    raw_helpful_cal: np.ndarray,
    raw_helpful_test: np.ndarray,
    meta_disagree: np.ndarray,
    test_disagree: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    masked_helpful_cal = raw_helpful_cal * meta_disagree.astype(np.float32)
    masked_helpful_test = raw_helpful_test * test_disagree.astype(np.float32)
    threshold_quantile = gate_threshold_quantile_from_args(args)
    trigger_meta: dict[str, Any] = {
        "policy": args.gate_policy,
        "threshold_quantile": threshold_quantile,
        "target_rate": None if args.gate_target_rate is None else float(args.gate_target_rate),
    }

    if args.gate_policy == "quantile_masked":
        use_local_cal, use_local, threshold = quantile_trigger(
            masked_helpful_cal,
            masked_helpful_test,
            threshold_quantile=threshold_quantile,
        )
        trigger_meta["threshold"] = threshold
    elif args.gate_policy == "quantile_nonzero_masked":
        nonzero_cal = masked_helpful_cal[masked_helpful_cal > 0]
        if nonzero_cal.size == 0:
            use_local_cal = np.zeros(len(masked_helpful_cal), dtype=bool)
            use_local = np.zeros(len(masked_helpful_test), dtype=bool)
            trigger_meta["threshold"] = None
        else:
            use_local_cal, use_local, threshold = quantile_trigger(
                nonzero_cal,
                masked_helpful_test,
                threshold_quantile=threshold_quantile,
            )
            use_local_cal = masked_helpful_cal >= threshold
            trigger_meta["threshold"] = threshold
    elif args.gate_policy == "quantile_raw_disagree":
        disagree_cal_scores = raw_helpful_cal[meta_disagree]
        if disagree_cal_scores.size == 0:
            use_local_cal = np.zeros(len(raw_helpful_cal), dtype=bool)
            use_local = np.zeros(len(raw_helpful_test), dtype=bool)
            trigger_meta["threshold"] = None
        else:
            _, disagree_test_selected, threshold = quantile_trigger(
                disagree_cal_scores,
                raw_helpful_test,
                threshold_quantile=threshold_quantile,
            )
            use_local_cal = meta_disagree & (raw_helpful_cal >= threshold)
            use_local = test_disagree & disagree_test_selected
            trigger_meta["threshold"] = threshold
    elif args.gate_policy == "topk_raw_disagree":
        if args.gate_target_rate is None:
            raise ValueError("--gate-target-rate is required when --gate-policy=topk_raw_disagree")
        use_local_cal, cal_topk_meta = topk_trigger(
            raw_helpful_cal,
            meta_disagree,
            target_rate=float(args.gate_target_rate),
        )
        use_local, test_topk_meta = topk_trigger(
            raw_helpful_test,
            test_disagree,
            target_rate=float(args.gate_target_rate),
        )
        trigger_meta["calibration_topk"] = cal_topk_meta
        trigger_meta["test_topk"] = test_topk_meta
        trigger_meta["threshold"] = test_topk_meta["selected_threshold"]
    else:
        raise ValueError(f"Unsupported gate policy: {args.gate_policy}")

    trigger_meta["diagnostics"] = {
        "raw_helpful_cal": gate_score_stats(raw_helpful_cal),
        "raw_helpful_test": gate_score_stats(raw_helpful_test),
        "masked_helpful_cal": gate_score_stats(masked_helpful_cal),
        "masked_helpful_test": gate_score_stats(masked_helpful_test),
        "meta_disagree_rate": float(meta_disagree.mean()),
        "test_disagree_rate": float(test_disagree.mean()),
    }
    return use_local_cal, use_local, trigger_meta


def apply_local_inference_mode(
    *,
    args: argparse.Namespace,
    test_logits: np.ndarray,
    test_local_logits: np.ndarray,
    raw_helpful_test: np.ndarray,
    use_local: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    final_logits = test_logits.copy()
    blend_beta = float(args.blend_beta)
    blend_gamma = float(args.blend_gamma)

    if args.local_inference_mode == "hard_switch":
        final_logits[use_local] = test_local_logits[use_local]
        return final_logits, {
            "mode": args.local_inference_mode,
            "blend_beta": 1.0,
            "blend_gamma": None,
            "mean_weight": float(use_local.mean()),
            "max_weight": 1.0 if bool(use_local.any()) else 0.0,
        }

    if args.local_inference_mode == "masked_blend":
        weights = blend_beta * use_local.astype(np.float32)
    elif args.local_inference_mode == "soft_blend":
        helpful = np.clip(raw_helpful_test.astype(np.float32), 0.0, 1.0)
        weights = blend_beta * np.power(helpful, blend_gamma)
    else:
        raise ValueError(f"Unsupported local inference mode: {args.local_inference_mode}")

    final_logits = test_logits + weights[:, None] * (test_local_logits - test_logits)
    return final_logits, {
        "mode": args.local_inference_mode,
        "blend_beta": blend_beta,
        "blend_gamma": blend_gamma if args.local_inference_mode == "soft_blend" else None,
        "mean_weight": float(weights.mean()),
        "max_weight": float(weights.max()) if weights.size > 0 else 0.0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    split_seed = int(args.split_seed)
    set_seed(split_seed)
    payload, refs = load_official_split(split_seed)
    x_train, x_test = encode_queries_gpu(
        payload["train_queries"],
        payload["test_queries"],
        batch_size=int(os.getenv("RARE_EMBED_BATCH_SIZE", "64")),
        embedding_model_name=EMBEDDING_MODEL,
        cache_prefix=f"llmrouterbench_performance_cost_prompt_seed{split_seed}",
    )

    base_idx, meta_idx = split_indices(len(x_train), split_seed)
    x_base = x_train[base_idx]
    y_base = payload["train_perf"][base_idx]
    x_meta_all = x_train[meta_idx]
    y_meta_all = payload["train_perf"][meta_idx]

    model_train_idx, model_val_idx = split_indices(len(x_base), split_seed + 1)
    x_model_train = x_base[model_train_idx]
    y_model_train = y_base[model_train_idx]
    x_model_val = x_base[model_val_idx]
    y_model_val = y_base[model_val_idx]

    train_cost_base = payload["train_cost"][base_idx]
    train_cost_model = train_cost_base[model_train_idx]
    xtr_ret, xmeta_ret, xtest_ret, ret_cfg = augment_with_cost_aware_retrieval_features(
        args=args,
        x_train=x_model_train,
        train_perf=y_model_train,
        train_cost=train_cost_model,
        x_val=x_meta_all,
        x_test=x_test,
    )
    knn_chunk_size = retrieval_knn_chunk_size()
    if args.retrieval_feature_preset == "perf_only":
        xval_profile = gpu_weighted_knn_scores(
            x_train=x_model_train,
            train_matrix=y_model_train,
            x_query=x_model_val,
            k=int(ret_cfg["k"]),
            tau=float(ret_cfg["tau"]),
            chunk_size=knn_chunk_size,
        )
    elif args.retrieval_feature_preset == "perf_cost_concat":
        train_retrieval_profile = np.concatenate(
            [y_model_train.astype(np.float32), cost_score_matrix(train_cost_model.astype(np.float32))],
            axis=1,
        ).astype(np.float32)
        xval_profile = gpu_weighted_knn_scores(
            x_train=x_model_train,
            train_matrix=train_retrieval_profile,
            x_query=x_model_val,
            k=int(ret_cfg["k"]),
            tau=float(ret_cfg["tau"]),
            chunk_size=knn_chunk_size,
        )
    elif args.retrieval_feature_preset == "utility_0.8":
        train_retrieval_profile = utility_profile_matrix(y_model_train, train_cost_model, alpha=0.8)
        xval_profile = gpu_weighted_knn_scores(
            x_train=x_model_train,
            train_matrix=train_retrieval_profile,
            x_query=x_model_val,
            k=int(ret_cfg["k"]),
            tau=float(ret_cfg["tau"]),
            chunk_size=knn_chunk_size,
        )
    else:
        raise ValueError(f"Unsupported retrieval feature preset: {args.retrieval_feature_preset}")
    xval_ret = np.concatenate([x_model_val, xval_profile], axis=1).astype(np.float32)
    backbone_loss_weights = backbone_loss_weights_from_args(args)
    model, stage1_meta = train_global_backbone(
        x_train=xtr_ret,
        y_train=y_model_train,
        x_val=xval_ret,
        y_val=y_model_val,
        batch_size=int(os.getenv("RARE_TRAIN_BATCH_SIZE", "2048")),
        loss_weights=backbone_loss_weights,
    )

    train_logits = infer_logits(model, xtr_ret)
    meta_logits = infer_logits(model, xmeta_ret)
    test_logits = infer_logits(model, xtest_ret)
    meta_mix = infer_factor_features(model, xmeta_ret, feature_kind="mixture")
    test_mix = infer_factor_features(model, xtest_ret, feature_kind="mixture")

    local_cfg = {
        "k": int(args.local_k),
        "alpha": float(args.local_alpha),
        "tau": float(args.local_tau),
        "uncertainty_threshold": float(args.local_uncertainty_threshold),
    }
    meta_local_logits = local_residual_rectify(
        train_embeddings=x_model_train,
        train_targets=y_model_train,
        train_logits=train_logits,
        query_embeddings=x_meta_all,
        query_logits=meta_logits,
        **local_cfg,
    )
    test_local_logits = local_residual_rectify(
        train_embeddings=x_model_train,
        train_targets=y_model_train,
        train_logits=train_logits,
        query_embeddings=x_test,
        query_logits=test_logits,
        **local_cfg,
    )

    meta_gate_features = np.concatenate(
        [stage1_gate_features(meta_logits, meta_mix), local_delta_features(meta_logits, meta_local_logits)],
        axis=1,
    ).astype(np.float32)
    test_gate_features = np.concatenate(
        [stage1_gate_features(test_logits, test_mix), local_delta_features(test_logits, test_local_logits)],
        axis=1,
    ).astype(np.float32)
    meta_helpful, _, meta_disagree = local_delta_labels(meta_logits, meta_local_logits, y_meta_all)
    _, _, test_disagree = local_delta_labels(test_logits, test_local_logits, payload["test_perf"])

    rng = np.random.default_rng(split_seed)
    idx = np.arange(len(meta_gate_features))
    rng.shuffle(idx)
    n_cal = max(128, int(len(idx) * 0.25))
    cal_idx = idx[:n_cal]
    gate_idx = idx[n_cal:]

    x_gate_train, x_gate_cal = standardize_fit(meta_gate_features[gate_idx], meta_gate_features[cal_idx])
    _, x_gate_test = standardize_fit(meta_gate_features[gate_idx], test_gate_features)
    gate_seed = split_seed + 1000
    set_seed(gate_seed)
    gate_model, gate_meta = train_risk_gate(
        x_train=x_gate_train,
        y_train=meta_helpful[gate_idx],
        x_val=x_gate_cal,
        y_val=meta_helpful[cal_idx],
    )
    raw_helpful_cal = infer_risk(gate_model, x_gate_cal)
    raw_helpful_test = infer_risk(gate_model, x_gate_test)
    use_local_cal, use_local, gate_trigger_meta = resolve_gate_trigger(
        args=args,
        raw_helpful_cal=raw_helpful_cal,
        raw_helpful_test=raw_helpful_test,
        meta_disagree=meta_disagree[cal_idx].astype(bool),
        test_disagree=test_disagree.astype(bool),
    )
    final_logits, inference_meta = apply_local_inference_mode(
        args=args,
        test_logits=test_logits,
        test_local_logits=test_local_logits,
        raw_helpful_test=raw_helpful_test,
        use_local=use_local,
    )

    curve = alpha_sweep(
        1.0 / (1.0 + np.exp(-final_logits)),
        payload["test_perf"],
        payload["test_cost"],
        payload["test_meta"],
    )
    summary = summarize_curve(curve, refs)

    result = {
        "method_name": "RARE",
        "setting": "LLMRouterBench performance-cost",
        "config_path": str(CONFIG_PATH),
        "split_protocol": {
            "name": "official_prompt_split",
            "train_ratio": TRAIN_RATIO,
            "split_seed": split_seed,
        },
        "embedding_model": EMBEDDING_MODEL,
        "references": refs,
        "alpha_grid": ALPHAS,
        "curve": curve,
        "summary": summary,
        "training": {
            "retrieval_cfg": ret_cfg,
            "stage1_meta": stage1_meta,
            "backbone_loss_preset": args.backbone_loss_preset,
            "backbone_loss_weights": backbone_loss_weights,
            "local_cfg": local_cfg,
            "gate_meta": gate_meta,
            "gate_seed": gate_seed,
            "gate_policy": args.gate_policy,
            "gate_threshold_quantile": gate_trigger_meta["threshold_quantile"],
            "gate_target_rate": gate_trigger_meta["target_rate"],
            "gate_threshold": gate_trigger_meta["threshold"],
            "gate_use_local_cal_rate": float(use_local_cal.mean()),
            "test_use_local_rate": float(use_local.mean()),
            "local_inference_mode": args.local_inference_mode,
            "inference_meta": inference_meta,
            "gate_trigger_meta": gate_trigger_meta,
            "knn_chunk_size": knn_chunk_size,
        },
        "frontier_source": str(BASE_COST_RESULT_JSON) if BASE_COST_RESULT_JSON.exists() else None,
    }

    result_json = args.result_json or default_result_path(split_seed)
    result_json.write_text(json.dumps(result, indent=2))
    print(json.dumps({"result_json": str(result_json), "summary": result["summary"]}, indent=2))
    return result


def run_response_override_variant(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    split_seed = int(args.split_seed)
    set_seed(split_seed)
    payload, refs = load_official_split(split_seed)
    x_train, x_test = encode_queries_gpu(
        payload["train_queries"],
        payload["test_queries"],
        batch_size=int(os.getenv("RARE_EMBED_BATCH_SIZE", "64")),
        embedding_model_name=EMBEDDING_MODEL,
        cache_prefix=f"llmrouterbench_performance_cost_prompt_seed{split_seed}",
    )

    models = list(payload["models"])
    base_idx, meta_idx = split_indices(len(x_train), split_seed)
    x_base = x_train[base_idx]
    y_base = payload["train_perf"][base_idx]
    x_meta_all = x_train[meta_idx]
    y_meta_all = payload["train_perf"][meta_idx]
    train_cost_base = payload["train_cost"][base_idx]
    meta_queries = [payload["train_queries"][int(i)] for i in meta_idx.tolist()]
    meta_meta = {
        row_idx: payload["train_meta"][int(source_idx)]
        for row_idx, source_idx in enumerate(meta_idx.tolist())
    }

    model_train_idx, model_val_idx = split_indices(len(x_base), split_seed + 1)
    x_model_train = x_base[model_train_idx]
    y_model_train = y_base[model_train_idx]
    x_model_val = x_base[model_val_idx]
    y_model_val = y_base[model_val_idx]
    train_cost_model = train_cost_base[model_train_idx]

    xtr_ret, xmeta_ret, xtest_ret, ret_cfg = augment_with_cost_aware_retrieval_features(
        args=args,
        x_train=x_model_train,
        train_perf=y_model_train,
        train_cost=train_cost_model,
        x_val=x_meta_all,
        x_test=x_test,
    )
    knn_chunk_size = retrieval_knn_chunk_size()
    if args.retrieval_feature_preset == "perf_only":
        xval_profile = gpu_weighted_knn_scores(
            x_train=x_model_train,
            train_matrix=y_model_train,
            x_query=x_model_val,
            k=int(ret_cfg["k"]),
            tau=float(ret_cfg["tau"]),
            chunk_size=knn_chunk_size,
        )
    elif args.retrieval_feature_preset == "perf_cost_concat":
        train_retrieval_profile = np.concatenate(
            [y_model_train.astype(np.float32), cost_score_matrix(train_cost_model.astype(np.float32))],
            axis=1,
        ).astype(np.float32)
        xval_profile = gpu_weighted_knn_scores(
            x_train=x_model_train,
            train_matrix=train_retrieval_profile,
            x_query=x_model_val,
            k=int(ret_cfg["k"]),
            tau=float(ret_cfg["tau"]),
            chunk_size=knn_chunk_size,
        )
    elif args.retrieval_feature_preset == "utility_0.8":
        train_retrieval_profile = utility_profile_matrix(y_model_train, train_cost_model, alpha=0.8)
        xval_profile = gpu_weighted_knn_scores(
            x_train=x_model_train,
            train_matrix=train_retrieval_profile,
            x_query=x_model_val,
            k=int(ret_cfg["k"]),
            tau=float(ret_cfg["tau"]),
            chunk_size=knn_chunk_size,
        )
    else:
        raise ValueError(f"Unsupported retrieval feature preset: {args.retrieval_feature_preset}")
    xval_ret = np.concatenate([x_model_val, xval_profile], axis=1).astype(np.float32)
    backbone_loss_weights = backbone_loss_weights_from_args(args)
    model, stage1_meta = train_global_backbone(
        x_train=xtr_ret,
        y_train=y_model_train,
        x_val=xval_ret,
        y_val=y_model_val,
        batch_size=int(os.getenv("RARE_TRAIN_BATCH_SIZE", "2048")),
        loss_weights=backbone_loss_weights,
    )

    train_logits = infer_logits(model, xtr_ret)
    meta_logits = infer_logits(model, xmeta_ret)
    test_logits = infer_logits(model, xtest_ret)
    local_cfg = {
        "k": int(args.local_k),
        "alpha": float(args.local_alpha),
        "tau": float(args.local_tau),
        "uncertainty_threshold": float(args.local_uncertainty_threshold),
    }
    meta_local_logits = local_residual_rectify(
        train_embeddings=x_model_train,
        train_targets=y_model_train,
        train_logits=train_logits,
        query_embeddings=x_meta_all,
        query_logits=meta_logits,
        **local_cfg,
    )
    test_local_logits = local_residual_rectify(
        train_embeddings=x_model_train,
        train_targets=y_model_train,
        train_logits=train_logits,
        query_embeddings=x_test,
        query_logits=test_logits,
        **local_cfg,
    )

    shortlist_k = int(os.getenv("RARE_RESPONSE_TOPK", "3"))
    meta_bundle = build_response_feature_bundle(meta_meta, models=models, queries=meta_queries)
    test_bundle = build_response_feature_bundle(payload["test_meta"], models=models, queries=payload["test_queries"])
    meta_shortlists = union_shortlists([meta_logits, meta_local_logits], top_k=shortlist_k)
    test_shortlists = union_shortlists([test_logits, test_local_logits], top_k=shortlist_k)
    meta_features, meta_labels, meta_group_ids, meta_map, shortlist_meta = _response_candidate_features(
        meta_shortlists,
        meta_bundle["features"],
        meta_bundle["predictions"],
        [meta_logits, meta_local_logits],
        y_meta_all,
        include_relative=False,
    )
    test_features, _test_labels, _test_group_ids, test_map, shortlist_test = _response_candidate_features(
        test_shortlists,
        test_bundle["features"],
        test_bundle["predictions"],
        [test_logits, test_local_logits],
        payload["test_perf"],
        include_relative=False,
    )

    rng = np.random.default_rng(split_seed + 101)
    unique_groups = np.unique(meta_group_ids)
    rng.shuffle(unique_groups)
    n_val_groups = max(64, int(len(unique_groups) * 0.2))
    val_groups = set(int(v) for v in unique_groups[:n_val_groups].tolist())
    train_idx = np.asarray([i for i, gid in enumerate(meta_group_ids.tolist()) if int(gid) not in val_groups], dtype=np.int64)
    val_idx = np.asarray([i for i, gid in enumerate(meta_group_ids.tolist()) if int(gid) in val_groups], dtype=np.int64)
    x_rr_train, x_rr_val = standardize_fit(meta_features[train_idx], meta_features[val_idx])
    _, x_rr_test = standardize_fit(meta_features[train_idx], test_features)

    risk_model, response_meta = train_risk_gate(
        x_train=x_rr_train,
        y_train=meta_labels[train_idx],
        x_val=x_rr_val,
        y_val=meta_labels[val_idx],
        hidden_dim=128,
        dropout=0.1,
        epochs=160,
        batch_size=1024,
        lr=1e-3,
        weight_decay=1e-4,
    )
    meta_candidate_probs = infer_risk(risk_model, standardize_fit(meta_features[train_idx], meta_features)[1])
    test_candidate_probs = infer_risk(risk_model, x_rr_test)
    meta_prob_matrix = _candidate_prob_matrix(len(meta_shortlists), len(models), meta_candidate_probs, meta_map)
    test_prob_matrix = _candidate_prob_matrix(len(test_shortlists), len(models), test_candidate_probs, test_map)

    meta_override_x, meta_override_target_idx = _build_override_features(
        base_logits=meta_local_logits,
        aux_logits=meta_logits,
        candidate_prob_matrix=meta_prob_matrix,
        shortlist_lists=meta_shortlists,
        response_features=meta_bundle["features"],
        predictions=meta_bundle["predictions"],
    )
    test_override_x, test_override_target_idx = _build_override_features(
        base_logits=test_local_logits,
        aux_logits=test_logits,
        candidate_prob_matrix=test_prob_matrix,
        shortlist_lists=test_shortlists,
        response_features=test_bundle["features"],
        predictions=test_bundle["predictions"],
    )
    base_choice_meta = np.argmax(meta_local_logits, axis=1)
    override_success = y_meta_all[np.arange(len(y_meta_all)), meta_override_target_idx] > y_meta_all[np.arange(len(y_meta_all)), base_choice_meta]
    val_group_mask = np.asarray([gid in val_groups for gid in range(len(meta_shortlists))], dtype=bool)
    x_gate_train, x_gate_val = standardize_fit(meta_override_x[~val_group_mask], meta_override_x[val_group_mask])
    _, x_gate_test = standardize_fit(meta_override_x[~val_group_mask], test_override_x)
    gate_model, gate_meta = train_risk_gate(
        x_train=x_gate_train,
        y_train=override_success[~val_group_mask].astype(np.float32),
        x_val=x_gate_val,
        y_val=override_success[val_group_mask].astype(np.float32),
        hidden_dim=96,
        dropout=0.1,
        epochs=120,
        batch_size=512,
        lr=1e-3,
        weight_decay=1e-4,
    )
    val_gate_prob = infer_risk(gate_model, x_gate_val)
    best_cfg = None
    val_perf = y_meta_all[val_group_mask]
    val_cost = payload["train_cost"][meta_idx][val_group_mask]
    val_meta = {
        row_idx: meta_meta[int(source_idx)]
        for row_idx, source_idx in enumerate(np.flatnonzero(val_group_mask).tolist())
    }
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
        chosen = base_choice_meta[val_group_mask].copy()
        replace = val_gate_prob >= threshold
        chosen[replace] = meta_override_target_idx[val_group_mask][replace]
        val_curve = alpha_sweep(
            np.eye(len(models), dtype=np.float32)[chosen],
            val_perf,
            val_cost,
            val_meta,
        )
        best_val_point = max(val_curve.values(), key=lambda row: row["dataset_avg"])
        score = float(best_val_point["dataset_avg"])
        if best_cfg is None or score > best_cfg["val_best_accuracy_dataset_avg"]:
            best_cfg = {
                "threshold": float(threshold),
                "val_best_accuracy_dataset_avg": score,
                "val_best_accuracy_sample_avg": float(best_val_point["sample_avg"]),
                "override_rate": float(replace.mean()),
            }
    assert best_cfg is not None

    test_gate_prob = infer_risk(gate_model, x_gate_test)
    base_choice_test = np.argmax(test_local_logits, axis=1)
    final_idx = base_choice_test.copy()
    replace = test_gate_prob >= float(best_cfg["threshold"])
    final_idx[replace] = test_override_target_idx[replace]
    final_scores = np.eye(len(models), dtype=np.float32)[final_idx]

    curve = alpha_sweep(
        final_scores,
        payload["test_perf"],
        payload["test_cost"],
        payload["test_meta"],
    )
    summary = summarize_curve(curve, refs)
    base_curve = alpha_sweep(
        1.0 / (1.0 + np.exp(-test_local_logits)),
        payload["test_perf"],
        payload["test_cost"],
        payload["test_meta"],
    )
    base_summary = summarize_curve(base_curve, refs)
    response_hit = _candidate_list_hit_rate(test_shortlists, payload["test_perf"])

    result = {
        "method_name": "RARE-RESP-OVERRIDE",
        "setting": "LLMRouterBench performance-cost",
        "config_path": str(CONFIG_PATH),
        "split_protocol": {
            "name": "official_prompt_split",
            "train_ratio": TRAIN_RATIO,
            "split_seed": split_seed,
        },
        "embedding_model": EMBEDDING_MODEL,
        "references": refs,
        "alpha_grid": ALPHAS,
        "curve": curve,
        "summary": summary,
        "training": {
            "family": "rare_response_override",
            "retrieval_cfg": ret_cfg,
            "stage1_meta": stage1_meta,
            "backbone_loss_preset": args.backbone_loss_preset,
            "backbone_loss_weights": backbone_loss_weights,
            "local_cfg": local_cfg,
            "response_gate_meta": response_meta,
            "override_gate_meta": gate_meta,
            "override_cfg": best_cfg,
            "shortlist_top_k": shortlist_k,
            "shortlist_experts": ["global", "local"],
            "response_shortlist_hit_rate": float(response_hit),
            "always_local_summary": base_summary,
            "test_override_rate": float(replace.mean()),
            "avg_candidate_size_meta": shortlist_meta["avg_candidate_size"],
            "avg_candidate_size_test": shortlist_test["avg_candidate_size"],
            "knn_chunk_size": knn_chunk_size,
        },
        "frontier_source": str(BASE_COST_RESULT_JSON) if BASE_COST_RESULT_JSON.exists() else None,
        "model_usage": model_usage_stats([models[int(idx)] for idx in final_idx]),
    }

    result_json = args.result_json or response_override_result_path(split_seed)
    result_json.write_text(json.dumps(result, indent=2))
    print(json.dumps({"result_json": str(result_json), "summary": result["summary"]}, indent=2))
    return result
