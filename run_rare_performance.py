#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

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
    ROOT,
    accuracy_from_logits,
    dataset_accuracy_from_logits,
    infer_risk,
    load_official_cached_split,
    set_seed,
    split_indices,
    stage1_gate_features,
    standardize_fit,
    train_risk_gate,
)


RESULT_JSON = ROOT / "results" / "result_rare_performance.json"


def augment_with_retrieval_features(
    x_train: np.ndarray,
    train_matrix: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    best = None
    best_payload = None
    for k in [16, 24, 32]:
        for tau in [0.03, 0.05]:
            tr_scores = knn_scores_leave_self(x_train, train_matrix, k=k, tau=tau)
            val_scores = gpu_weighted_knn_scores(x_train=x_train, train_matrix=train_matrix, x_query=x_val, k=k, tau=tau, chunk_size=1024)
            score = float(val_scores.max(axis=1).mean())
            if best is None or score > best:
                best = score
                test_scores = gpu_weighted_knn_scores(x_train=x_train, train_matrix=train_matrix, x_query=x_test, k=k, tau=tau, chunk_size=1024)
                best_payload = (
                    np.concatenate([x_train, tr_scores], axis=1).astype(np.float32),
                    np.concatenate([x_val, val_scores], axis=1).astype(np.float32),
                    np.concatenate([x_test, test_scores], axis=1).astype(np.float32),
                    {"k": k, "tau": tau, "neighbor_conf_mean": score},
                )
    assert best_payload is not None
    return best_payload


def knn_scores_leave_self(
    x_train: np.ndarray,
    train_matrix: np.ndarray,
    *,
    k: int = 24,
    tau: float = 0.05,
    chunk_size: int = 1024,
) -> np.ndarray:
    device = torch.device("cuda")
    xtr = torch.tensor(x_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(train_matrix, dtype=torch.float32, device=device)
    top_k = min(k + 1, xtr.shape[0])
    outs = []
    for start in range(0, xtr.shape[0], chunk_size):
        q = xtr[start:start + chunk_size]
        sim = q @ xtr.T
        nn_scores, nn_idx = torch.topk(sim, k=top_k, dim=1)
        row_ids = torch.arange(start, start + q.shape[0], device=device).unsqueeze(1)
        keep = nn_idx != row_ids
        filtered_scores = []
        filtered_idx = []
        for i in range(nn_idx.shape[0]):
            valid_scores = nn_scores[i][keep[i]][:k]
            valid_idx = nn_idx[i][keep[i]][:k]
            if valid_scores.numel() < k:
                pad = k - valid_scores.numel()
                valid_scores = torch.cat([valid_scores, torch.full((pad,), -1e9, device=device)])
                valid_idx = torch.cat([valid_idx, torch.zeros(pad, dtype=torch.long, device=device)])
            filtered_scores.append(valid_scores)
            filtered_idx.append(valid_idx)
        top_scores = torch.stack(filtered_scores, dim=0)
        top_idx = torch.stack(filtered_idx, dim=0)
        nn_perf = ytr[top_idx]
        weights = torch.softmax(top_scores / tau, dim=1)
        outs.append((nn_perf * weights.unsqueeze(-1)).sum(dim=1).cpu())
    return torch.cat(outs, dim=0).numpy().astype(np.float32)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def top_margin_np(logits: np.ndarray) -> np.ndarray:
    probs = softmax_np(logits)
    top2 = np.partition(probs, -2, axis=1)[:, -2:]
    top2.sort(axis=1)
    return (top2[:, 1] - top2[:, 0]).astype(np.float32)


def local_delta_features(global_logits: np.ndarray, local_logits: np.ndarray) -> np.ndarray:
    g_probs = softmax_np(global_logits)
    l_probs = softmax_np(local_logits)
    g_top = np.argmax(global_logits, axis=1)
    l_top = np.argmax(local_logits, axis=1)
    row = np.arange(len(global_logits))
    n_models = global_logits.shape[1]
    dense = np.column_stack([
        g_probs[row, g_top],
        l_probs[row, l_top],
        top_margin_np(global_logits),
        top_margin_np(local_logits),
        l_probs[row, l_top] - g_probs[row, g_top],
        global_logits[row, l_top] - global_logits[row, g_top],
        local_logits[row, l_top] - local_logits[row, g_top],
        (g_top == l_top).astype(np.float32),
        np.linalg.norm(local_logits - global_logits, axis=1),
        (local_logits - global_logits).mean(axis=1),
        (local_logits - global_logits).std(axis=1),
    ]).astype(np.float32)
    g_onehot = np.eye(n_models, dtype=np.float32)[g_top]
    l_onehot = np.eye(n_models, dtype=np.float32)[l_top]
    return np.concatenate([dense, g_onehot, l_onehot], axis=1).astype(np.float32)


def local_delta_labels(global_logits: np.ndarray, local_logits: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row = np.arange(len(targets))
    g_sel = np.argmax(global_logits, axis=1)
    l_sel = np.argmax(local_logits, axis=1)
    g_score = targets[row, g_sel]
    l_score = targets[row, l_sel]
    helpful = (l_score > g_score).astype(np.float32)
    harmful = (g_score > l_score).astype(np.float32)
    disagree = g_sel != l_sel
    return helpful, harmful, disagree


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    split_seed = 42
    set_seed(split_seed)
    train_matrix, train_queries, train_meta, test_matrix, test_queries, test_meta, models = load_official_cached_split()
    x_train, x_test = encode_queries_gpu(
        train_queries,
        test_queries,
        batch_size=int(os.getenv("GLIDER_EMBED_BATCH_SIZE", "256")),
        embedding_model_name=EMBEDDING_MODEL,
        cache_prefix=f"llmrouterbench_performance_v2_prompt_seed{split_seed}",
    )

    base_idx, meta_idx = split_indices(len(x_train), split_seed)
    x_base = x_train[base_idx]
    y_base = train_matrix[base_idx]
    x_meta_all = x_train[meta_idx]
    y_meta_all = train_matrix[meta_idx]

    model_train_idx, model_val_idx = split_indices(len(x_base), split_seed + 1)
    x_model_train = x_base[model_train_idx]
    y_model_train = y_base[model_train_idx]
    x_model_val = x_base[model_val_idx]
    y_model_val = y_base[model_val_idx]

    xtr_ret, xmeta_ret, xtest_ret, ret_cfg = augment_with_retrieval_features(x_model_train, y_model_train, x_meta_all, x_test)
    xval_ret = np.concatenate([
        x_model_val,
        gpu_weighted_knn_scores(x_train=x_model_train, train_matrix=y_model_train, x_query=x_model_val, k=int(ret_cfg["k"]), tau=float(ret_cfg["tau"]), chunk_size=1024),
    ], axis=1).astype(np.float32)
    model, train_meta_info = train_global_backbone(
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

    meta_gate_features = np.concatenate([stage1_gate_features(meta_logits, meta_mix), local_delta_features(meta_logits, meta_local_logits)], axis=1).astype(np.float32)
    test_gate_features = np.concatenate([stage1_gate_features(test_logits, test_mix), local_delta_features(test_logits, test_local_logits)], axis=1).astype(np.float32)
    meta_helpful, _, meta_disagree = local_delta_labels(meta_logits, meta_local_logits, y_meta_all)
    _, _, test_disagree = local_delta_labels(test_logits, test_local_logits, test_matrix)

    rng = np.random.default_rng(split_seed)
    idx = np.arange(len(meta_gate_features))
    rng.shuffle(idx)
    n_cal = max(128, int(len(idx) * 0.25))
    cal_idx = idx[:n_cal]
    gate_idx = idx[n_cal:]

    x_gate_train, x_gate_cal = standardize_fit(meta_gate_features[gate_idx], meta_gate_features[cal_idx])
    _, x_gate_test = standardize_fit(meta_gate_features[gate_idx], test_gate_features)
    gate_model, gate_meta = train_risk_gate(
        x_train=x_gate_train,
        y_train=meta_helpful[gate_idx],
        x_val=x_gate_cal,
        y_val=meta_helpful[cal_idx],
    )
    helpful_cal = infer_risk(gate_model, x_gate_cal)
    helpful_test = infer_risk(gate_model, x_gate_test)
    helpful_cal = helpful_cal * meta_disagree[cal_idx].astype(np.float32)
    helpful_test = helpful_test * test_disagree.astype(np.float32)
    threshold = float(np.quantile(helpful_cal, 0.95))
    use_local = helpful_test >= threshold
    final_logits = test_logits.copy()
    final_logits[use_local] = test_local_logits[use_local]

    payload = {
        "method_name": "RARE",
        "setting": "LLMRouterBench performance",
        "embedding_model": EMBEDDING_MODEL,
        "retrieval_cfg": ret_cfg,
        "stage1_training": train_meta_info,
        "local_candidate_cfg": local_cfg,
        "gate_meta": gate_meta,
        "gate_target": "helpful_prob_disagree_only",
        "gate_threshold_quantile": 0.95,
        "stage1": {
            "sample_avg": accuracy_from_logits(test_logits, test_matrix),
            "dataset_avg": dataset_accuracy_from_logits(test_logits, test_matrix, models, test_meta)[1],
        },
        "always_local": {
            "sample_avg": accuracy_from_logits(test_local_logits, test_matrix),
            "dataset_avg": dataset_accuracy_from_logits(test_local_logits, test_matrix, models, test_meta)[1],
        },
        "best_test": {
            "sample_avg": accuracy_from_logits(final_logits, test_matrix),
            "dataset_avg": dataset_accuracy_from_logits(final_logits, test_matrix, models, test_meta)[1],
            "use_local_rate": float(use_local.mean()),
        },
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"result_json": str(RESULT_JSON), "best_test": payload["best_test"]}, indent=2))


if __name__ == "__main__":
    main()
