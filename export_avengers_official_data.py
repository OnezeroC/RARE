#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from src.rare_shared import LLMROUTERBENCH_ROOT, PAPER_PERFORMANCE_SEEDS, POOL_EXP_ROOT, ROOT


DEFAULT_LLMROUTERBENCH_ROOT = LLMROUTERBENCH_ROOT
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "avengers_official_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export official LLMRouterBench performance splits to Avengers input files."
    )
    parser.add_argument(
        "--llmrouterbench-root",
        type=Path,
        default=DEFAULT_LLMROUTERBENCH_ROOT,
        help="Path to the official LLMRouterBench workspace.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Optional baseline config path. Defaults to <llmrouterbench-root>/config/baseline_config.yaml.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for exported Avengers data/config files.",
    )
    parser.add_argument(
        "--split-seeds",
        type=int,
        nargs="+",
        default=list(PAPER_PERFORMANCE_SEEDS),
        help="Official prompt-split seeds to export.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Official performance setting uses 0.7 train ratio.",
    )
    return parser.parse_args()


def import_official_loader(llmrouterbench_root: Path):
    sys.path.insert(0, str(llmrouterbench_root))
    from baselines.data_loader import BaselineDataLoader  # type: ignore

    return BaselineDataLoader


def split_by_dataset_then_prompt(
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
    for dataset_records in dataset_groups.values():
        prompt_to_records: dict[str, list[Any]] = defaultdict(list)
        for record in dataset_records:
            prompt_to_records[str(record.prompt)].append(record)

        unique_prompts = list(prompt_to_records.keys())
        unique_prompts.sort(key=lambda prompt: min(r.record_index for r in prompt_to_records[prompt]))

        n_train = int(len(unique_prompts) * train_ratio)
        indices = list(range(len(unique_prompts)))
        rng.shuffle(indices)
        train_indices = set(indices[:n_train])

        for idx, prompt in enumerate(unique_prompts):
            if idx in train_indices:
                train_records.extend(prompt_to_records[prompt])
            else:
                test_records.extend(prompt_to_records[prompt])

    return train_records, test_records


def records_to_jsonl(records: list[Any], all_models: list[str]) -> list[dict[str, Any]]:
    prompt_groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for record in records:
        prompt_groups[(str(record.dataset_id), str(record.prompt))].append(record)

    rows: list[dict[str, Any]] = []
    for (dataset_id, prompt), prompt_records in prompt_groups.items():
        records_dict: dict[str, float] = {}
        usages_dict: dict[str, dict[str, float | int]] = {}
        index = min(int(r.record_index) for r in prompt_records)

        for record in prompt_records:
            model_name = str(record.model_name)
            records_dict[model_name] = float(record.score)
            usages_dict[model_name] = {
                "completion_tokens": int(record.completion_tokens),
                "cost": float(record.cost),
                "prompt_tokens": int(record.prompt_tokens),
            }

        for model_name in all_models:
            if model_name in records_dict:
                continue
            records_dict[model_name] = 0.0
            usages_dict[model_name] = {
                "completion_tokens": 0,
                "cost": 0.0,
                "prompt_tokens": 0,
            }

        rows.append(
            {
                "query": prompt,
                "dataset": dataset_id,
                "index": index,
                "records": records_dict,
                "usages": usages_dict,
            }
        )

    rows.sort(key=lambda row: (row["dataset"], row["index"]))
    return rows


def baseline_scores(records: list[Any]) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        scores[str(record.model_name)][str(record.dataset_id)].append(float(record.score))

    result: dict[str, dict[str, float]] = {}
    for model_name, per_ds in scores.items():
        result[model_name] = {}
        for dataset_id, values in per_ds.items():
            result[model_name][dataset_id] = round(sum(values) / len(values) * 100.0, 2)
    return result


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_router_config(seed: int, train_path: Path, test_path: Path, baseline_path: Path) -> dict[str, Any]:
    return {
        "train_data_path": str(train_path),
        "test_data_path": str(test_path),
        "baseline_scores_path": str(baseline_path),
        "n_clusters": 30,
        "seed": seed,
        "max_router": 1,
        "top_k": 1,
        "beta": 9.0,
        "max_workers": 1,
        "cluster_batch_size": 2048,
        "embedding_batch_size": 16,
        "embedding_model": "gte_Qwen2-7B-instruct",
        "embedding_base_url": "https://sd3g8mam6uj313gtp0fbg.apigateway-cn-beijing.volceapi.com/v1",
        "embedding_api_key": "inplaceholder",
        "embedding_config_path": "config/embedding_config.yaml",
        "shared_embedding_cache_path": str(POOL_EXP_ROOT / "hs_cache" / "shared_embedding_cache.sqlite"),
        "excluded_models": [],
        "ood_datasets": [],
    }


def export_seed(records: list[Any], all_models: list[str], output_root: Path, seed: int, train_ratio: float) -> dict[str, Any]:
    seed_dir = output_root / f"seed{seed}_split{train_ratio:.1f}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    train_records, test_records = split_by_dataset_then_prompt(
        records,
        train_ratio=train_ratio,
        random_seed=seed,
    )

    train_rows = records_to_jsonl(train_records, all_models)
    test_rows = records_to_jsonl(test_records, all_models)
    score_rows = baseline_scores(test_records)

    train_path = seed_dir / "train.jsonl"
    test_path = seed_dir / "test.jsonl"
    baseline_path = seed_dir / "baseline_scores.json"
    config_path = output_root / f"simple_config_small_models_seed{seed}.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(test_path, test_rows)
    baseline_path.write_text(json.dumps(score_rows, indent=2, ensure_ascii=False))
    config_path.write_text(
        json.dumps(build_router_config(seed, train_path, test_path, baseline_path), indent=2, ensure_ascii=False)
    )

    return {
        "seed": seed,
        "train_queries": len(train_rows),
        "test_queries": len(test_rows),
        "train_records": len(train_records),
        "test_records": len(test_records),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "baseline_scores_path": str(baseline_path),
        "config_path": str(config_path),
    }


def main() -> None:
    args = parse_args()
    llmrouterbench_root = args.llmrouterbench_root.resolve()
    config_path = (args.config_path or (llmrouterbench_root / "config" / "baseline_config.yaml")).resolve()
    output_root = args.output_root.resolve()

    loader_cls = import_official_loader(llmrouterbench_root)
    config = yaml.safe_load(config_path.read_text())
    config["baseline"]["results_dir"] = str((llmrouterbench_root / "results" / "bench").resolve())
    loader = loader_cls(config=config["baseline"])
    records = list(loader.load_records_iter())
    all_models = sorted({str(record.model_name) for record in records})

    summaries = []
    for seed in args.split_seeds:
        summaries.append(export_seed(records, all_models, output_root, seed, args.train_ratio))

    print(json.dumps({"exported": summaries}, indent=2))


if __name__ == "__main__":
    main()
