#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_rare_performance import augment_with_retrieval_features, local_delta_features, local_delta_labels
from src.glider_router import encode_queries_gpu, infer_factor_features, infer_logits, local_residual_rectify, train_global_backbone
from src.glider_v2_router import gpu_weighted_knn_scores
from src.rare_shared import (
    EMBEDDING_MODEL,
    ROOT,
    infer_risk,
    load_jsonl,
    set_seed,
    split_cost_rows,
    split_indices,
    stage1_gate_features,
    standardize_fit,
    train_risk_gate,
)


RESULT_JSON = ROOT / "results" / "result_rare_performance_cost.json"
TRAIN_PATH = ROOT / "data" / "performance_cost_split" / "train.jsonl"
TEST_PATH = ROOT / "data" / "performance_cost_split" / "test.jsonl"
ALPHAS = [0.0, 0.25, 0.39, 0.53, 0.8, 1.0]


def model_reference_table(test_perf: np.ndarray, test_cost: np.ndarray, models: list[str]) -> dict[str, Any]:
    per_model = {}
    for col, model_name in enumerate(models):
        per_model[model_name] = {"sample_avg": float(test_perf[:, col].mean()), "avg_cost": float(test_cost[:, col].mean())}
    best_single = max(models, key=lambda model: per_model[model]["sample_avg"])
    return {
        "per_model": per_model,
        "best_single_sample_model": best_single,
        "best_single_sample_avg": per_model[best_single]["sample_avg"],
        "best_single_sample_cost": per_model[best_single]["avg_cost"],
        "gpt5_reference_model": "gpt-5",
        "gpt5_reference_acc": per_model["gpt-5"]["sample_avg"],
        "gpt5_reference_cost": per_model["gpt-5"]["avg_cost"],
    }


def per_dataset_accuracy_from_idx(selected_idx: np.ndarray, matrix: np.ndarray, meta: dict[int, dict[str, Any]]) -> dict[str, float]:
    correct = {}
    total = {}
    for row_idx, model_idx in enumerate(selected_idx):
        ds = meta[row_idx]["dataset"]
        total[ds] = total.get(ds, 0) + 1
        if matrix[row_idx, int(model_idx)] > 0:
            correct[ds] = correct.get(ds, 0) + 1
    return {ds: correct.get(ds, 0) / total[ds] for ds in sorted(total)}


def evaluate_selected(selected_idx: np.ndarray, test_perf: np.ndarray, test_cost: np.ndarray, test_meta: dict[int, dict[str, Any]]) -> dict[str, Any]:
    per_dataset = per_dataset_accuracy_from_idx(selected_idx, test_perf, test_meta)
    return {
        "sample_avg": float(test_perf[np.arange(len(test_perf)), selected_idx].mean()),
        "dataset_avg": float(np.mean(list(per_dataset.values()))),
        "avg_cost": float(test_cost[np.arange(len(test_cost)), selected_idx].mean()),
        "total_cost": float(test_cost[np.arange(len(test_cost)), selected_idx].sum()),
        "per_dataset": per_dataset,
    }


def normalize_perf_scores(perf_scores: np.ndarray) -> np.ndarray:
    row_min = perf_scores.min(axis=1, keepdims=True)
    row_max = perf_scores.max(axis=1, keepdims=True)
    return (perf_scores - row_min) / np.clip(row_max - row_min, 1e-8, None)


def cost_score_matrix(cost_matrix: np.ndarray) -> np.ndarray:
    row_max = np.clip(cost_matrix.max(axis=1, keepdims=True), 1e-8, None)
    return 1.0 - (cost_matrix / row_max)


def alpha_sweep(perf_scores: np.ndarray, test_perf: np.ndarray, test_cost: np.ndarray, test_meta: dict[int, dict[str, Any]]) -> dict[str, Any]:
    perf_norm = normalize_perf_scores(perf_scores.astype(np.float32))
    cost_norm = cost_score_matrix(test_cost)
    curve = {}
    for alpha in ALPHAS:
        utility = alpha * perf_norm + (1.0 - alpha) * cost_norm
        selected_idx = np.argmax(utility, axis=1)
        curve[str(alpha)] = evaluate_selected(selected_idx, test_perf, test_cost, test_meta) | {"selected_indices": selected_idx.tolist()}
    return curve


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    set_seed(42)
    payload = split_cost_rows(load_jsonl(TRAIN_PATH), load_jsonl(TEST_PATH))
    refs = model_reference_table(payload["test_perf"], payload["test_cost"], payload["models"])
    x_train, x_test = encode_queries_gpu(
        payload["train_queries"],
        payload["test_queries"],
        batch_size=int(os.getenv("GLIDER_EMBED_BATCH_SIZE", "64")),
        embedding_model_name=EMBEDDING_MODEL,
        cache_prefix="llmrouterbench_performance_cost_prompt_seed42",
    )

    base_idx, meta_idx = split_indices(len(x_train), 42)
    x_base = x_train[base_idx]
    y_base = payload["train_perf"][base_idx]
    x_meta_all = x_train[meta_idx]
    y_meta_all = payload["train_perf"][meta_idx]

    model_train_idx, model_val_idx = split_indices(len(x_base), 43)
    x_model_train = x_base[model_train_idx]
    y_model_train = y_base[model_train_idx]
    x_model_val = x_base[model_val_idx]
    y_model_val = y_base[model_val_idx]

    xtr_ret, xmeta_ret, xtest_ret, ret_cfg = augment_with_retrieval_features(x_model_train, y_model_train, x_meta_all, x_test)
    xval_ret = np.concatenate([
        x_model_val,
        gpu_weighted_knn_scores(x_train=x_model_train, train_matrix=y_model_train, x_query=x_model_val, k=int(ret_cfg["k"]), tau=float(ret_cfg["tau"]), chunk_size=1024),
    ], axis=1).astype(np.float32)
    model, stage1_meta = train_global_backbone(
        x_train=xtr_ret,
        y_train=y_model_train,
        x_val=xval_ret,
        y_val=y_model_val,
        batch_size=int(os.getenv("GLIDER_TRAIN_BATCH_SIZE", "2048")),
    )

    train_logits = infer_logits(model, xtr_ret)
    meta_logits = infer_logits(model, xmeta_ret)
    test_logits = infer_logits(model, xtest_ret)
    meta_mix = infer_factor_features(model, xmeta_ret, feature_kind="mixture")
    test_mix = infer_factor_features(model, xtest_ret, feature_kind="mixture")

    local_cfg = {"k": 24, "alpha": 1.0, "tau": 0.03, "uncertainty_threshold": 2.0}
    meta_local_logits = local_residual_rectify(train_embeddings=x_model_train, train_targets=y_model_train, train_logits=train_logits, query_embeddings=x_meta_all, query_logits=meta_logits, **local_cfg)
    test_local_logits = local_residual_rectify(train_embeddings=x_model_train, train_targets=y_model_train, train_logits=train_logits, query_embeddings=x_test, query_logits=test_logits, **local_cfg)

    meta_gate_features = np.concatenate([stage1_gate_features(meta_logits, meta_mix), local_delta_features(meta_logits, meta_local_logits)], axis=1).astype(np.float32)
    test_gate_features = np.concatenate([stage1_gate_features(test_logits, test_mix), local_delta_features(test_logits, test_local_logits)], axis=1).astype(np.float32)
    meta_helpful, _, meta_disagree = local_delta_labels(meta_logits, meta_local_logits, y_meta_all)
    _, _, test_disagree = local_delta_labels(test_logits, test_local_logits, payload["test_perf"])

    rng = np.random.default_rng(42)
    idx = np.arange(len(meta_gate_features))
    rng.shuffle(idx)
    n_cal = max(128, int(len(idx) * 0.25))
    cal_idx = idx[:n_cal]
    gate_idx = idx[n_cal:]

    x_gate_train, x_gate_cal = standardize_fit(meta_gate_features[gate_idx], meta_gate_features[cal_idx])
    _, x_gate_test = standardize_fit(meta_gate_features[gate_idx], test_gate_features)
    gate_model, gate_meta = train_risk_gate(x_train=x_gate_train, y_train=meta_helpful[gate_idx], x_val=x_gate_cal, y_val=meta_helpful[cal_idx])
    helpful_cal = infer_risk(gate_model, x_gate_cal) * meta_disagree[cal_idx].astype(np.float32)
    helpful_test = infer_risk(gate_model, x_gate_test) * test_disagree.astype(np.float32)
    threshold = float(np.quantile(helpful_cal, 0.95))
    use_local = helpful_test >= threshold
    final_logits = test_logits.copy()
    final_logits[use_local] = test_local_logits[use_local]

    curve = alpha_sweep(1.0 / (1.0 + np.exp(-final_logits)), payload["test_perf"], payload["test_cost"], payload["test_meta"])
    best = max((curve[str(alpha)] | {"alpha": alpha} for alpha in ALPHAS), key=lambda row: row["sample_avg"])

    result = {
        "method_name": "RARE",
        "setting": "LLMRouterBench performance-cost",
        "embedding_model": EMBEDDING_MODEL,
        "references": refs,
        "alpha_grid": ALPHAS,
        "curve": curve,
        "summary": {
            "best_accuracy_alpha": best["alpha"],
            "best_accuracy_sample_avg": best["sample_avg"],
            "best_accuracy_avg_cost": best["avg_cost"],
            "perf_gain_vs_gpt5": best["sample_avg"] - refs["gpt5_reference_acc"],
            "perf_gain_vs_best_single_sample": best["sample_avg"] - refs["best_single_sample_avg"],
            "cost_save_vs_gpt5": (refs["gpt5_reference_cost"] - best["avg_cost"]) / refs["gpt5_reference_cost"],
        },
        "training": {
            "retrieval_cfg": ret_cfg,
            "stage1_meta": stage1_meta,
            "local_cfg": local_cfg,
            "gate_meta": gate_meta,
            "gate_threshold_quantile": 0.95,
            "test_use_local_rate": float(use_local.mean()),
        },
    }
    RESULT_JSON.write_text(json.dumps(result, indent=2))
    print(json.dumps({"result_json": str(RESULT_JSON), "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
