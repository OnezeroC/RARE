from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config.paths import artifacts_results_root
from src.data.adaptors.response_records import build_response_feature_bundle
from src.evaluation.performance_shared import (
    base_result_payload,
    build_per_dataset_rows,
    load_official_split as load_standardized_split,
    model_usage_stats,
)
from src.methods.rare.response_rerank import expert_rank_features, union_shortlists
from src.models.expert_fusion_router import gpu_weighted_knn_scores
from src.models.global_local_router import (
    encode_queries_gpu,
    infer_logits,
    local_residual_rectify,
    train_global_backbone,
)
from src.shared import (
    EMBEDDING_MODEL,
    accuracy_from_logits,
    dataset_accuracy_from_logits,
    infer_risk,
    load_official_cached_split,
    set_seed,
    split_indices,
    standardize_fit,
    train_risk_gate,
)


RESULT_JSON = artifacts_results_root() / "result_rare_performance.json"


def retrieval_knn_chunk_size() -> int:
    return int(os.getenv("RARE_KNN_CHUNK_SIZE", "1024"))


def response_override_seed_result_path(split_seed: int) -> Path:
    return artifacts_results_root() / f"result_rare_performance_seed{split_seed}_response_override.json"


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


def augment_with_retrieval_features(
    x_train: np.ndarray,
    train_matrix: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    best = None
    best_payload = None
    chunk_size = retrieval_knn_chunk_size()
    for k in [16, 24, 32]:
        for tau in [0.03, 0.05]:
            tr_scores = knn_scores_leave_self(
                x_train,
                train_matrix,
                k=k,
                tau=tau,
                chunk_size=chunk_size,
            )
            val_scores = gpu_weighted_knn_scores(
                x_train=x_train,
                train_matrix=train_matrix,
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
                    train_matrix=train_matrix,
                    x_query=x_test,
                    k=k,
                    tau=tau,
                    chunk_size=chunk_size,
                )
                best_payload = (
                    np.concatenate([x_train, tr_scores], axis=1).astype(np.float32),
                    np.concatenate([x_val, val_scores], axis=1).astype(np.float32),
                    np.concatenate([x_test, test_scores], axis=1).astype(np.float32),
                    {"k": k, "tau": tau, "neighbor_conf_mean": score},
                )
    assert best_payload is not None
    return best_payload


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def top_margin_np(logits: np.ndarray) -> np.ndarray:
    probs = softmax_np(logits)
    top2 = np.partition(probs, -2, axis=1)[:, -2:]
    top2.sort(axis=1)
    return (top2[:, 1] - top2[:, 0]).astype(np.float32)


def _candidate_list_hit_rate(
    candidate_lists: list[list[int]],
    targets: np.ndarray,
) -> float:
    hits = 0
    for row_idx, candidates in enumerate(candidate_lists):
        if any(targets[row_idx, cand] > 0 for cand in candidates):
            hits += 1
    return float(hits / max(1, len(candidate_lists)))


def _response_candidate_features(
    candidate_lists: list[list[int]],
    response_features: np.ndarray,
    predictions: np.ndarray,
    expert_logits: list[np.ndarray],
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]], dict[str, float]]:
    expert_ranks = [expert_rank_features(logits) for logits in expert_logits]
    rows: list[np.ndarray] = []
    labels: list[float] = []
    group_ids: list[int] = []
    mapping: list[tuple[int, int]] = []
    sizes: list[int] = []
    for row_idx, candidates in enumerate(candidate_lists):
        sizes.append(len(candidates))
        cand_arr = np.asarray(candidates, dtype=np.int64)
        shortlist_resp = response_features[row_idx, cand_arr]
        shortlist_resp_mean = shortlist_resp.mean(axis=0)
        counts: dict[object, int] = {}
        for cand in candidates:
            key = predictions[row_idx, cand]
            counts[key] = counts.get(key, 0) + 1
        max_count = max(counts.values()) if counts else 1
        vote_counts = np.zeros((len(candidates),), dtype=np.float32)
        mean_rank_pct = np.zeros((len(candidates),), dtype=np.float32)
        mean_gap_to_best = np.zeros((len(candidates),), dtype=np.float32)
        for expert_logits_single, expert_ranks_single in zip(expert_logits, expert_ranks):
            shortlist_logits = expert_logits_single[row_idx, cand_arr]
            shortlist_best = float(shortlist_logits.max())
            shortlist_len = max(1, len(candidates) - 1)
            for local_idx, cand in enumerate(candidates):
                vote_counts[local_idx] += float(expert_ranks_single[row_idx, cand] == 0)
                mean_rank_pct[local_idx] += float(expert_ranks_single[row_idx, cand] / shortlist_len)
                mean_gap_to_best[local_idx] += float(expert_logits_single[row_idx, cand] - shortlist_best)
        mean_rank_pct /= max(1, len(expert_logits))
        mean_gap_to_best /= max(1, len(expert_logits))
        for local_idx, cand in enumerate(candidates):
            pred_text = predictions[row_idx, cand]
            expert_dense: list[float] = []
            for logits, ranks in zip(expert_logits, expert_ranks):
                expert_dense.extend(
                    [
                        float(logits[row_idx, cand]),
                        float(ranks[row_idx, cand]),
                        float(ranks[row_idx, cand] == 0),
                    ]
                )
            row = np.concatenate(
                [
                    response_features[row_idx, cand],
                    np.zeros_like(response_features[row_idx, cand] - shortlist_resp_mean),
                    np.asarray(
                        expert_dense
                        + [
                            float(counts.get(pred_text, 0) / max(1, len(candidates))),
                            float(counts.get(pred_text, 0) == max_count),
                            float(vote_counts[local_idx]),
                            float(mean_rank_pct[local_idx]),
                            float(mean_gap_to_best[local_idx]),
                        ],
                        dtype=np.float32,
                    ),
                ],
                axis=0,
            ).astype(np.float32)
            rows.append(row)
            labels.append(float(targets[row_idx, cand] > 0))
            group_ids.append(int(row_idx))
            mapping.append((row_idx, cand))
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(group_ids, dtype=np.int64),
        mapping,
        {
            "avg_candidate_size": float(np.mean(sizes)) if sizes else 0.0,
            "max_candidate_size": float(max(sizes) if sizes else 0),
        },
    )


def _candidate_prob_matrix(
    n_queries: int,
    n_models: int,
    candidate_probs: np.ndarray,
    candidate_map: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full((n_queries, n_models), -1e9, dtype=np.float32)
    for prob, (row_idx, cand) in zip(candidate_probs.tolist(), candidate_map):
        out[row_idx, cand] = float(prob)
    return out


def _build_override_features(
    *,
    base_logits: np.ndarray,
    aux_logits: np.ndarray,
    candidate_prob_matrix: np.ndarray,
    shortlist_lists: list[list[int]],
    response_features: np.ndarray,
    predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base_idx = np.argmax(base_logits, axis=1)
    top_idx = np.argmax(candidate_prob_matrix, axis=1)
    features: list[np.ndarray] = []
    for row_idx in range(len(base_logits)):
        shortlist = shortlist_lists[row_idx]
        shortlist_size = float(len(shortlist))
        base_model = int(base_idx[row_idx])
        top_model = int(top_idx[row_idx])
        base_prob = float(candidate_prob_matrix[row_idx, base_model]) if base_model in shortlist else -1e9
        top_prob = float(candidate_prob_matrix[row_idx, top_model])
        base_feat = response_features[row_idx, base_model]
        top_feat = response_features[row_idx, top_model]
        answer_same = float(predictions[row_idx, base_model] == predictions[row_idx, top_model])
        row = np.concatenate(
            [
                np.asarray(
                    [
                        base_prob,
                        top_prob,
                        top_prob - base_prob,
                        float(base_logits[row_idx, base_model]),
                        float(aux_logits[row_idx, top_model]),
                        float(top_margin_np(base_logits[row_idx: row_idx + 1])[0]),
                        float(top_margin_np(aux_logits[row_idx: row_idx + 1])[0]),
                        float(np.argmax(base_logits[row_idx]) == np.argmax(aux_logits[row_idx])),
                        float(base_model == top_model),
                        shortlist_size,
                        answer_same,
                    ],
                    dtype=np.float32,
                ),
                base_feat,
                top_feat,
                top_feat - base_feat,
            ]
        ).astype(np.float32)
        features.append(row)
    return np.asarray(features, dtype=np.float32), top_idx.astype(np.int64)


def _reference_context(seed: int) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str], list[str]]:
    (
        _train_matrix,
        std_train_queries,
        _train_meta,
        _test_matrix,
        std_test_queries,
        _test_meta,
        std_models,
        reference,
        per_dataset_ref,
    ) = load_standardized_split(query_field="prompt", split_seed=seed)
    reference = {
        **reference,
        "embedding_model": EMBEDDING_MODEL,
        "split_seed": int(seed),
    }
    return reference, per_dataset_ref, std_train_queries, std_test_queries, std_models


def run_response_override_variant(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    split_seed = int(args.split_seed)
    set_seed(split_seed)
    train_matrix, train_queries, train_meta, test_matrix, test_queries, test_meta, models = load_official_cached_split(split_seed)
    x_train, x_test = encode_queries_gpu(
        train_queries,
        test_queries,
        batch_size=int(os.getenv("RARE_EMBED_BATCH_SIZE", "256")),
        embedding_model_name=EMBEDDING_MODEL,
        cache_prefix=f"llmrouterbench_performance_v2_prompt_seed{split_seed}",
    )

    base_idx, meta_idx = split_indices(len(x_train), split_seed)
    x_base = x_train[base_idx]
    y_base = train_matrix[base_idx]
    x_meta_all = x_train[meta_idx]
    y_meta_all = train_matrix[meta_idx]
    meta_queries = [train_queries[int(i)] for i in meta_idx.tolist()]
    meta_meta = {row_idx: train_meta[int(source_idx)] for row_idx, source_idx in enumerate(meta_idx.tolist())}

    model_train_idx, model_val_idx = split_indices(len(x_base), split_seed + 1)
    x_model_train = x_base[model_train_idx]
    y_model_train = y_base[model_train_idx]
    x_model_val = x_base[model_val_idx]
    y_model_val = y_base[model_val_idx]

    xtr_ret, xmeta_ret, xtest_ret, ret_cfg = augment_with_retrieval_features(x_model_train, y_model_train, x_meta_all, x_test)
    xval_ret = np.concatenate(
        [
            x_model_val,
            gpu_weighted_knn_scores(
                x_train=x_model_train,
                train_matrix=y_model_train,
                x_query=x_model_val,
                k=int(ret_cfg["k"]),
                tau=float(ret_cfg["tau"]),
                chunk_size=1024,
            ),
        ],
        axis=1,
    ).astype(np.float32)
    model, train_meta_info = train_global_backbone(
        x_train=xtr_ret,
        y_train=y_model_train,
        x_val=xval_ret,
        y_val=y_model_val,
        batch_size=int(os.getenv("RARE_TRAIN_BATCH_SIZE", "2048")),
    )

    train_logits = infer_logits(model, xtr_ret)
    meta_logits = infer_logits(model, xmeta_ret)
    test_logits = infer_logits(model, xtest_ret)

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

    shortlist_k = int(os.getenv("RARE_RESPONSE_TOPK", "3"))
    meta_bundle = build_response_feature_bundle(meta_meta, models=models, queries=meta_queries)
    test_bundle = build_response_feature_bundle(test_meta, models=models, queries=test_queries)
    meta_shortlists = union_shortlists([meta_logits, meta_local_logits], top_k=shortlist_k)
    test_shortlists = union_shortlists([test_logits, test_local_logits], top_k=shortlist_k)

    meta_features, meta_labels, meta_group_ids, meta_map, shortlist_meta = _response_candidate_features(
        meta_shortlists,
        meta_bundle["features"],
        meta_bundle["predictions"],
        [meta_logits, meta_local_logits],
        y_meta_all,
    )
    test_features, _test_labels, _test_group_ids, test_map, shortlist_test = _response_candidate_features(
        test_shortlists,
        test_bundle["features"],
        test_bundle["predictions"],
        [test_logits, test_local_logits],
        test_matrix,
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
    best_cfg: dict[str, float] | None = None
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
        chosen = base_choice_meta[val_group_mask].copy()
        replace = val_gate_prob >= threshold
        chosen[replace] = meta_override_target_idx[val_group_mask][replace]
        val_acc = float((y_meta_all[val_group_mask][np.arange(replace.shape[0]), chosen] > 0).mean())
        if best_cfg is None or val_acc > best_cfg["val_sample_avg"]:
            best_cfg = {
                "threshold": float(threshold),
                "val_sample_avg": float(val_acc),
                "override_rate": float(replace.mean()),
            }
    assert best_cfg is not None

    test_gate_prob = infer_risk(gate_model, x_gate_test)
    base_choice_test = np.argmax(test_local_logits, axis=1)
    final_idx = base_choice_test.copy()
    replace = test_gate_prob >= float(best_cfg["threshold"])
    final_idx[replace] = test_override_target_idx[replace]
    final_logits = np.full_like(test_local_logits, -1e9)
    final_logits[np.arange(len(final_logits)), final_idx] = 1.0

    result_json = args.result_json or response_override_seed_result_path(split_seed)
    final_per_dataset, final_dataset_avg = dataset_accuracy_from_logits(final_logits, test_matrix, models, test_meta)
    final_sample_avg = accuracy_from_logits(final_logits, test_matrix)
    local_per_dataset, local_dataset_avg = dataset_accuracy_from_logits(test_local_logits, test_matrix, models, test_meta)
    response_hit = _candidate_list_hit_rate(test_shortlists, test_matrix)

    reference, per_dataset_ref, std_train_queries, std_test_queries, std_models = _reference_context(split_seed)
    if set(std_models) != set(models):
        raise RuntimeError("Model set mismatch between cached split and LLMRouterBench split loader.")
    if std_models != models:
        reorder_idx = np.asarray([models.index(model_name) for model_name in std_models], dtype=np.int64)
        models = list(std_models)
        test_matrix = test_matrix[:, reorder_idx]
        test_local_logits = test_local_logits[:, reorder_idx]
        final_logits = final_logits[:, reorder_idx]
        final_per_dataset, final_dataset_avg = dataset_accuracy_from_logits(final_logits, test_matrix, models, test_meta)
        final_sample_avg = accuracy_from_logits(final_logits, test_matrix)
        local_per_dataset, local_dataset_avg = dataset_accuracy_from_logits(test_local_logits, test_matrix, models, test_meta)

    payload: dict[str, Any] = base_result_payload(
        method_name="RARE-RESP-OVERRIDE",
        split_seed=split_seed,
        query_field="prompt",
        train_queries=std_train_queries,
        test_queries=std_test_queries,
        models=models,
        reference=reference,
    )
    payload["baseline"] = {
        "family": "rare_response_override",
        "retrieval_cfg": ret_cfg,
        "stage1_training": train_meta_info,
        "local_candidate_cfg": local_cfg,
        "response_gate_meta": response_meta,
        "override_gate_meta": gate_meta,
        "override_cfg": best_cfg,
        "shortlist_top_k": shortlist_k,
        "shortlist_experts": ["global", "local"],
    }
    payload["overall"] = {
        "rare_sample_avg": final_sample_avg,
        "rare_dataset_avg": final_dataset_avg,
        "best_single_sample_avg": reference["best_single_sample_avg"],
        "dataset_best_single_avg": reference["dataset_best_single_avg"],
        "oracle_sample_avg": reference["oracle_sample_avg"],
        "dataset_oracle_avg": reference["dataset_oracle_avg"],
    }
    payload["model_usage"] = model_usage_stats([models[int(idx)] for idx in final_idx])
    payload["per_dataset"] = build_per_dataset_rows(
        scores=final_per_dataset,
        per_dataset_ref=per_dataset_ref,
        method_key="rare",
    )
    payload["diagnostics"] = {
        "always_local": {
            "sample_avg": accuracy_from_logits(test_local_logits, test_matrix),
            "dataset_avg": local_dataset_avg,
            "per_dataset": local_per_dataset,
        },
        "response_shortlist": {
            "top_k_per_expert": shortlist_k,
            "avg_candidate_size_meta": shortlist_meta["avg_candidate_size"],
            "avg_candidate_size_test": shortlist_test["avg_candidate_size"],
            "test_candidate_hit_rate": float(response_hit),
        },
        "override": {
            **best_cfg,
            "test_override_rate": float(replace.mean()),
        },
        "best_test": {
            "sample_avg": final_sample_avg,
            "dataset_avg": final_dataset_avg,
            "per_dataset": final_per_dataset,
        },
    }
    payload["details"] = {
        "method_name": "RARE-RESP-OVERRIDE",
        "setting": "LLMRouterBench performance",
        "split_protocol": {
            "name": "official_prompt_split",
            "train_ratio": 0.7,
            "split_seed": split_seed,
        },
        "embedding_model": EMBEDDING_MODEL,
        "retrieval_cfg": ret_cfg,
        "stage1_training": train_meta_info,
        "local_candidate_cfg": local_cfg,
        "response_gate_meta": response_meta,
        "override_gate_meta": gate_meta,
        "override_cfg": best_cfg,
        "response_feature_dim": int(meta_features.shape[1]),
        "shortlist_experts": ["global", "local"],
    }
    result_json.write_text(json.dumps(payload, indent=2))
    if split_seed == 42:
        RESULT_JSON.write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            {
                "result_json": str(result_json),
                "overall": payload["overall"],
                "local_dataset_avg": local_dataset_avg,
            },
            indent=2,
        )
    )
    return payload
