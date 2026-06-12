#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_rare_performance_cost import (
    ALPHAS,
    alpha_sweep,
    apply_local_inference_mode,
    augment_with_cost_aware_retrieval_features,
    backbone_loss_weights_from_args,
    resolve_gate_trigger,
    summarize_curve,
)
from src.glider_router import (
    encode_queries_gpu,
    infer_factor_features,
    infer_logits,
    local_residual_rectify,
    train_global_backbone,
)
from src.glider_v2_router import gpu_weighted_knn_scores
from src.rare_shared import (
    EMBEDDING_MODEL,
    LLMROUTERBENCH_ROOT,
    POOL_EXP_ROOT,
    ROOT,
    infer_risk,
    set_seed,
    split_indices,
    stage1_gate_features,
    standardize_fit,
    train_risk_gate,
)
from run_rare_performance import (
    local_delta_features,
    local_delta_labels,
    retrieval_knn_chunk_size,
)


RESULT_JSON = ROOT / "results" / "result_rare_cost_suite_aligned.json"
TRAIN_PATH = LLMROUTERBENCH_ROOT / "baselines" / "AvengersPro" / "data" / "proprietary_models" / "seed42_split0.7" / "train.jsonl"
TEST_PATH = LLMROUTERBENCH_ROOT / "baselines" / "AvengersPro" / "data" / "proprietary_models" / "seed42_split0.7" / "test.jsonl"
SEED = 42
TRAIN_RATIO = 0.7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-aligned RARE cost-suite evaluation.")
    parser.add_argument("--result-json", type=Path, default=RESULT_JSON)
    parser.add_argument("--split-seed", type=int, default=SEED)
    parser.add_argument("--local-k", type=int, default=24)
    parser.add_argument("--local-alpha", type=float, default=1.0)
    parser.add_argument("--local-tau", type=float, default=0.05)
    parser.add_argument("--local-uncertainty-threshold", type=float, default=2.0)
    parser.add_argument(
        "--gate-policy",
        type=str,
        default="quantile_masked",
        choices=["quantile_masked", "quantile_nonzero_masked", "quantile_raw_disagree", "topk_raw_disagree"],
    )
    parser.add_argument("--gate-threshold-quantile", type=float, default=0.98)
    parser.add_argument("--gate-target-rate", type=float, default=None)
    parser.add_argument(
        "--backbone-loss-preset",
        type=str,
        default="listwise_onlyish",
        choices=["baseline", "margin_heavy", "listwise_onlyish"],
    )
    parser.add_argument(
        "--retrieval-feature-preset",
        type=str,
        default="perf_only",
        choices=["perf_only", "perf_cost_concat", "utility_0.8"],
    )
    parser.add_argument(
        "--local-inference-mode",
        type=str,
        default="soft_blend",
        choices=["hard_switch", "masked_blend", "soft_blend"],
    )
    parser.add_argument("--blend-beta", type=float, default=0.75)
    parser.add_argument("--blend-gamma", type=float, default=0.5)
    return parser.parse_args()


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_to_arrays(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    model_set = set()
    for row in train_rows + test_rows:
        model_set.update(row["records"].keys())
    models = sorted(model_set)
    model_to_idx = {name: idx for idx, name in enumerate(models)}

    def rows_to_payload(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[str], dict[int, dict[str, Any]]]:
        perf = np.zeros((len(rows), len(models)), dtype=np.float32)
        cost = np.zeros((len(rows), len(models)), dtype=np.float32)
        queries: list[str] = []
        meta: dict[int, dict[str, Any]] = {}
        for row_idx, row in enumerate(rows):
            queries.append(str(row["query"]))
            meta[row_idx] = {
                "dataset": str(row["dataset"]).lower(),
                "index": int(row["index"]),
            }
            for model_name, score in row["records"].items():
                col = model_to_idx[model_name]
                perf[row_idx, col] = float(score)
                usage = row.get("usages", {}).get(model_name, {})
                cost[row_idx, col] = float(usage.get("cost", 0.0) or 0.0)
        return perf, cost, queries, meta

    train_perf, train_cost, train_queries, train_meta = rows_to_payload(train_rows)
    test_perf, test_cost, test_queries, test_meta = rows_to_payload(test_rows)
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


def model_reference_table(
    test_perf: np.ndarray,
    test_cost: np.ndarray,
    models: list[str],
) -> dict[str, Any]:
    per_model = {}
    for col, model_name in enumerate(models):
        per_model_dataset = []
        for row_name in DATASET_ORDER:
            ds_indices = DATASET_SLICES.get(row_name)
            if ds_indices is None:
                continue
            start, end = ds_indices
            if end > start:
                per_model_dataset.append(float(test_perf[start:end, col].mean()))
        per_model[model_name] = {
            "dataset_avg": float(np.mean(per_model_dataset)) if per_model_dataset else 0.0,
            "sample_avg": float(test_perf[:, col].mean()),
            "avg_cost": float(test_cost[:, col].mean()),
        }
    best_single_dataset_model = max(models, key=lambda model: per_model[model]["dataset_avg"])
    best_single_sample_model = max(models, key=lambda model: per_model[model]["sample_avg"])
    gpt5_key = "gpt-5" if "gpt-5" in per_model else best_single_dataset_model
    return {
        "per_model": per_model,
        "best_single_dataset_model": best_single_dataset_model,
        "best_single_dataset_avg": per_model[best_single_dataset_model]["dataset_avg"],
        "best_single_dataset_cost": per_model[best_single_dataset_model]["avg_cost"],
        "best_single_sample_model": best_single_sample_model,
        "best_single_sample_avg": per_model[best_single_sample_model]["sample_avg"],
        "best_single_sample_cost": per_model[best_single_sample_model]["avg_cost"],
        "gpt5_reference_model": gpt5_key,
        "gpt5_reference_dataset_avg": per_model[gpt5_key]["dataset_avg"],
        "gpt5_reference_sample_avg": per_model[gpt5_key]["sample_avg"],
        "gpt5_reference_cost": per_model[gpt5_key]["avg_cost"],
        "oracle_sample_avg": float(test_perf.max(axis=1).mean()),
        "random_router_sample_avg": float(test_perf.mean(axis=1).mean()),
    }


def global_pareto_frontier(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier = []
    for i, point in enumerate(points):
        dominated = False
        for j, other in enumerate(points):
            if i == j:
                continue
            if (
                other["dataset_avg"] >= point["dataset_avg"]
                and other["avg_cost"] <= point["avg_cost"]
                and (
                    other["dataset_avg"] > point["dataset_avg"]
                    or other["avg_cost"] < point["avg_cost"]
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(point)
    frontier.sort(key=lambda row: (row["avg_cost"], -row["dataset_avg"]))
    return frontier


def pareto_distance(point: dict[str, Any], frontier: list[dict[str, Any]]) -> float:
    best = float("inf")
    for candidate in frontier:
        extra_cost = max(0.0, point["avg_cost"] - candidate["avg_cost"])
        missing_acc = max(0.0, candidate["dataset_avg"] - point["dataset_avg"])
        best = min(best, extra_cost + missing_acc)
    return 0.0 if best == float("inf") else float(best)


def build_frontier_points(curve: dict[str, Any], refs: dict[str, Any]) -> list[dict[str, Any]]:
    points = []
    for alpha in ALPHAS:
        row = curve[str(alpha)]
        points.append({
            "method": "rare",
            "alpha": alpha,
            "dataset_avg": row["dataset_avg"],
            "sample_avg": row["sample_avg"],
            "avg_cost": row["avg_cost"],
        })
    for model_name, row in refs["per_model"].items():
        points.append({
            "method": f"single::{model_name}",
            "alpha": None,
            "dataset_avg": row["dataset_avg"],
            "sample_avg": row["sample_avg"],
            "avg_cost": row["avg_cost"],
        })
    return points


def summarize_curve_with_frontier(
    curve: dict[str, Any],
    refs: dict[str, Any],
    frontier: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = summarize_curve(curve, refs)
    points = [curve[str(alpha)] | {"alpha": alpha} for alpha in ALPHAS]
    summary["pareto_dist"] = float(np.mean([pareto_distance(row, frontier) for row in points]))
    return summary


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for paper-aligned RARE cost-suite evaluation.")

    split_seed = int(args.split_seed)
    set_seed(split_seed)
    if split_seed != 42:
        raise ValueError("Paper-aligned cost-suite currently supports only seed42, matching the paper protocol.")
    train_rows = load_jsonl(TRAIN_PATH)
    test_rows = load_jsonl(TEST_PATH)
    payload = split_to_arrays(train_rows, test_rows)
    refs = model_reference_table(payload["test_perf"], payload["test_cost"], payload["models"])
    x_train, x_test = encode_queries_gpu(
        payload["train_queries"],
        payload["test_queries"],
        batch_size=int(os.getenv("GLIDER_EMBED_BATCH_SIZE", "64")),
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
    xval_profile = gpu_weighted_knn_scores(
        x_train=x_model_train,
        train_matrix=y_model_train,
        x_query=x_model_val,
        k=int(ret_cfg["k"]),
        tau=float(ret_cfg["tau"]),
        chunk_size=knn_chunk_size,
    )
    xval_ret = np.concatenate([x_model_val, xval_profile], axis=1).astype(np.float32)

    model, stage1_meta = train_global_backbone(
        x_train=xtr_ret,
        y_train=y_model_train,
        x_val=xval_ret,
        y_val=y_model_val,
        batch_size=int(os.getenv("GLIDER_TRAIN_BATCH_SIZE", "2048")),
        loss_weights=backbone_loss_weights_from_args(args),
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
    frontier = global_pareto_frontier(build_frontier_points(curve, refs))
    summary = summarize_curve_with_frontier(curve, refs, frontier)

    result = {
        "method_name": "RARE",
        "setting": "LLMRouterBench performance-cost",
        "split_protocol": {
            "name": "paper_cost_suite_aligned",
            "train_ratio": TRAIN_RATIO,
            "split_seed": split_seed,
            "train_path": str(TRAIN_PATH),
            "test_path": str(TEST_PATH),
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
        "frontier": frontier,
    }
    args.result_json.write_text(json.dumps(result, indent=2))
    print(json.dumps({"result_json": str(args.result_json), "summary": summary}, indent=2))
    release_cuda_memory()


if __name__ == "__main__":
    main()
