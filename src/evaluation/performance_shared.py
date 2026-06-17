from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import yaml

from src.config.paths import third_party_root
from src.shared import EMBEDDING_MODEL


LLMROUTERBENCH_ROOT = third_party_root()
CONFIG_PATH = LLMROUTERBENCH_ROOT / "config" / "baseline_config.yaml"
TRAIN_RATIO = 0.7


def resolve_loader():
    import sys

    sys.path.insert(0, str(LLMROUTERBENCH_ROOT))
    from baselines.data_loader import BaselineDataLoader

    cfg = yaml.safe_load(CONFIG_PATH.read_text())["baseline"]
    resolved = dict(cfg)
    resolved["results_dir"] = str((LLMROUTERBENCH_ROOT / cfg["results_dir"]).resolve())
    loader = BaselineDataLoader(config=resolved)
    return loader, resolved


def matrix_from_records(
    records: list[Any],
    models: list[str],
    *,
    query_field: str,
) -> tuple[np.ndarray, list[str], dict[int, dict[str, Any]]]:
    model_to_idx = {name: idx for idx, name in enumerate(models)}
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        ds = str(record.dataset_id).lower()
        record_index = int(record.record_index)
        key = (ds, record_index)
        if key not in grouped:
            grouped[key] = {
                "dataset": ds,
                "index": record_index,
                "query": str(getattr(record, query_field) or record.prompt or ""),
                "scores": {},
            }
        grouped[key]["scores"][str(record.model_name)] = float(record.score)

    ordered_keys = sorted(grouped.keys(), key=lambda x: (x[0], x[1]))
    matrix = np.zeros((len(ordered_keys), len(models)), dtype=np.float32)
    queries: list[str] = []
    meta: dict[int, dict[str, Any]] = {}
    for row_idx, key in enumerate(ordered_keys):
        row = grouped[key]
        queries.append(row["query"])
        meta[row_idx] = {
            "dataset": row["dataset"],
            "index": row["index"],
        }
        for model_name, score in row["scores"].items():
            matrix[row_idx, model_to_idx[model_name]] = score
    return matrix, queries, meta


def model_usage_stats(selected_models: list[str]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for model_name in selected_models:
        counts[model_name] = counts.get(model_name, 0) + 1
    total = max(len(selected_models), 1)
    return {
        "selected_counts": counts,
        "selected_ratios": {k: v / total for k, v in counts.items()},
        "total_queries": len(selected_models),
    }


def summarize_reference_split(
    test_matrix: np.ndarray,
    models: list[str],
    test_meta: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    sample_means = test_matrix.mean(axis=0)
    best_single_sample_idx = int(np.argmax(sample_means))
    best_single_sample_model = models[best_single_sample_idx]
    best_single_sample_avg = float(sample_means[best_single_sample_idx])

    dataset_model_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    oracle_scores: dict[str, list[float]] = defaultdict(list)
    for row_idx, row in enumerate(test_matrix):
        ds = str(test_meta[row_idx]["dataset"])
        oracle_scores[ds].append(float(np.max(row)))
        for model_name, score in zip(models, row.tolist()):
            dataset_model_scores[ds][model_name].append(float(score))

    per_dataset_ref: dict[str, dict[str, Any]] = {}
    best_dataset_scores = []
    oracle_dataset_scores = []
    dataset_model_avgs: dict[str, dict[str, float]] = {}
    for ds in sorted(dataset_model_scores):
        per_model_avg = {
            model_name: float(np.mean(scores))
            for model_name, scores in dataset_model_scores[ds].items()
        }
        dataset_model_avgs[ds] = per_model_avg
        best_model = max(per_model_avg, key=per_model_avg.get)
        best_score = per_model_avg[best_model]
        oracle_avg = float(np.mean(oracle_scores[ds]))
        best_dataset_scores.append(best_score)
        oracle_dataset_scores.append(oracle_avg)
        per_dataset_ref[ds] = {
            "n_queries": int(sum(len(v) for v in dataset_model_scores[ds].values()) / len(models)),
            "best_single_model": best_model,
            "best_single_accuracy": best_score,
            "oracle_accuracy": oracle_avg,
        }

    model_dataset_avgs = {
        model_name: float(np.mean([dataset_model_avgs[ds].get(model_name, 0.0) for ds in sorted(dataset_model_avgs)]))
        for model_name in models
    }
    best_single_dataset_model = max(model_dataset_avgs, key=model_dataset_avgs.get)
    reference = {
        "best_single_sample_avg_model": best_single_sample_model,
        "best_single_sample_avg": best_single_sample_avg,
        "best_single_dataset_avg_model": best_single_dataset_model,
        "dataset_best_single_avg": float(np.mean(best_dataset_scores)),
        "oracle_sample_avg": float(np.mean(np.max(test_matrix, axis=1))),
        "dataset_oracle_avg": float(np.mean(oracle_dataset_scores)),
    }
    return reference, per_dataset_ref


def load_official_split(
    *,
    query_field: str,
    split_seed: int,
) -> tuple[np.ndarray, list[str], dict[int, dict[str, Any]], np.ndarray, list[str], dict[int, dict[str, Any]], list[str], dict[str, Any], dict[str, dict[str, Any]]]:
    loader, resolved_config = resolve_loader()
    all_records = loader.load_all_records()
    train_records, test_records = loader.split_by_dataset_then_prompt(
        records=all_records,
        train_ratio=TRAIN_RATIO,
        random_seed=split_seed,
    )

    configured_models = resolved_config.get("filters", {}).get("models") or []
    observed_models = {str(record.model_name) for record in all_records}
    models = [model_name for model_name in configured_models if model_name in observed_models]
    if not models:
        models = sorted(observed_models)

    train_matrix, train_queries, train_meta = matrix_from_records(
        train_records,
        models,
        query_field=query_field,
    )
    test_matrix, test_queries, test_meta = matrix_from_records(
        test_records,
        models,
        query_field=query_field,
    )
    reference, per_dataset_ref = summarize_reference_split(test_matrix, models, test_meta)
    return (
        train_matrix,
        train_queries,
        train_meta,
        test_matrix,
        test_queries,
        test_meta,
        models,
        reference,
        per_dataset_ref,
    )


def base_result_payload(
    *,
    method_name: str,
    split_seed: int,
    query_field: str,
    train_queries: list[str],
    test_queries: list[str],
    models: list[str],
    reference: dict[str, Any],
) -> dict[str, Any]:
    return {
        "setting": "LLMRouterBench performance",
        "config_path": str(CONFIG_PATH),
        "split_protocol": {
            "name": "official_prompt_split",
            "train_ratio": TRAIN_RATIO,
            "split_seed": split_seed,
        },
        "query_field": query_field,
        "embedding_model": EMBEDDING_MODEL,
        "method_name": method_name,
        "n_models": len(models),
        "n_train_queries": len(train_queries),
        "n_test_queries": len(test_queries),
        "references": reference,
    }


def build_per_dataset_rows(
    *,
    scores: dict[str, float],
    per_dataset_ref: dict[str, dict[str, Any]],
    method_key: str,
) -> dict[str, dict[str, Any]]:
    rows = {}
    for dataset in sorted(per_dataset_ref):
        rows[dataset] = {
            **per_dataset_ref[dataset],
            method_key: scores[dataset],
        }
    return rows
