#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.rare_shared import LLMROUTERBENCH_ROOT, PAPER_PERFORMANCE_SEEDS, performance_split_cache_paths


DEFAULT_LLMROUTERBENCH_ROOT = LLMROUTERBENCH_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build official LLMRouterBench performance prompt-split caches for one or more seeds."
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
        "--split-seeds",
        type=int,
        nargs="+",
        default=list(PAPER_PERFORMANCE_SEEDS),
        help="Prompt-split seeds to generate.",
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


def build_matrix_payload(records: list[Any], models: list[str]) -> tuple[np.ndarray, list[str], dict[int, dict[str, object]]]:
    model_to_col = {name: idx for idx, name in enumerate(models)}
    grouped: OrderedDict[tuple[str, int], dict[str, Any]] = OrderedDict()

    for record in records:
        key = (record.dataset_id, int(record.record_index))
        bucket = grouped.setdefault(
            key,
            {
                "dataset": str(record.dataset_id),
                "record_index": int(record.record_index),
                "prompt": str(record.prompt),
                "scores": {},
            },
        )
        bucket["scores"][str(record.model_name)] = float(record.score)

    matrix = np.zeros((len(grouped), len(models)), dtype=np.float32)
    queries: list[str] = []
    meta: dict[int, dict[str, object]] = {}

    for row_idx, payload in enumerate(grouped.values()):
        queries.append(payload["prompt"])
        meta[row_idx] = {
            "dataset": payload["dataset"],
            "record_index": payload["record_index"],
        }
        for model_name, score in payload["scores"].items():
            matrix[row_idx, model_to_col[model_name]] = score

    return matrix, queries, meta


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

    for dataset_id, dataset_records in dataset_groups.items():
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


def build_cache_for_seed(
    loader,
    records: list[Any],
    llmrouterbench_root: Path,
    config_path: Path,
    split_seed: int,
    train_ratio: float,
) -> dict[str, Any]:
    train_records, test_records = split_by_dataset_then_prompt(records, train_ratio=train_ratio, random_seed=split_seed)

    models = sorted({record.model_name for record in records})
    train_matrix, train_queries, train_meta = build_matrix_payload(train_records, models)
    test_matrix, test_queries, test_meta = build_matrix_payload(test_records, models)

    out_npz, out_meta = performance_split_cache_paths(split_seed)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        train_matrix=train_matrix,
        test_matrix=test_matrix,
        train_queries=np.asarray(train_queries, dtype=object),
        test_queries=np.asarray(test_queries, dtype=object),
    )
    out_meta.write_text(
        json.dumps(
            {
                "models": models,
                "train_meta": {str(k): v for k, v in train_meta.items()},
                "test_meta": {str(k): v for k, v in test_meta.items()},
                "split_protocol": {
                    "name": "official_prompt_split",
                    "train_ratio": train_ratio,
                    "split_seed": split_seed,
                },
                "source": {
                    "llmrouterbench_root": str(llmrouterbench_root),
                    "config_path": str(config_path),
                },
            },
            indent=2,
        )
    )

    return {
        "split_seed": split_seed,
        "n_models": len(models),
        "n_train_queries": len(train_queries),
        "n_test_queries": len(test_queries),
        "cache_npz": str(out_npz),
        "cache_meta": str(out_meta),
    }


def main() -> None:
    args = parse_args()
    llmrouterbench_root = args.llmrouterbench_root.resolve()
    config_path = (args.config_path or (llmrouterbench_root / "config" / "baseline_config.yaml")).resolve()

    if not llmrouterbench_root.exists():
        raise FileNotFoundError(f"Missing LLMRouterBench root: {llmrouterbench_root}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing baseline config: {config_path}")

    loader_cls = import_official_loader(llmrouterbench_root)
    config = yaml.safe_load(config_path.read_text())
    config["baseline"]["results_dir"] = str((llmrouterbench_root / "results" / "bench").resolve())
    loader = loader_cls(config=config["baseline"])
    records = list(loader.load_records_iter())

    summaries = []
    for split_seed in args.split_seeds:
        summaries.append(
            build_cache_for_seed(
                loader=loader,
                records=records,
                llmrouterbench_root=llmrouterbench_root,
                config_path=config_path,
                split_seed=split_seed,
                train_ratio=args.train_ratio,
            )
        )

    print(json.dumps({"generated": summaries}, indent=2))


if __name__ == "__main__":
    main()
