#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_rare_performance_cost import (
    ALPHAS,
    CONFIG_PATH,
    TRAIN_RATIO,
    alpha_sweep,
    load_official_split,
    summarize_curve,
)
from src.glider_router import encode_queries_gpu
from src.rare_shared import EMBEDDING_MODEL, LLMROUTERBENCH_ROOT, POOL_EXP_ROOT, ROOT, set_seed


LLMRB_ROOT = LLMROUTERBENCH_ROOT
sys.path.insert(0, str(POOL_EXP_ROOT))
sys.path.insert(0, str(LLMRB_ROOT))

from LLMRouterBench.baselines.AvengersPro.balance_cluster_router import BalanceClusterRouter
from LLMRouterBench.baselines.AvengersPro.config import SimpleClusterConfig
from run_baselines_llmrouterbench_performance import infer_routerbench_mlp, train_routerbench_mlp


RESULTS_DIR = ROOT / "results"
AVENGERS_SPLIT_DIR = ROOT / "data" / "performance_cost_avengers_official"
EMBED_CFG = LLMRB_ROOT / "config" / "embedding_config.yaml"
SHARED_CACHE = ROOT / "data" / "hs_cache" / "shared_embedding_cache.sqlite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official-split performance-cost baselines.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["avengers", "knn", "routerbench_mlp"],
        choices=["avengers", "knn", "routerbench_mlp"],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 999, 2024, 2025, 3407],
    )
    return parser.parse_args()


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def knn_chunk_size() -> int:
    return int(os.getenv("KNN_CHUNK_SIZE", os.getenv("RARE_KNN_CHUNK_SIZE", "64")))


def knn_scores_gpu(
    x_train: np.ndarray,
    train_perf: np.ndarray,
    x_test: np.ndarray,
    *,
    k: int = 10,
) -> np.ndarray:
    device = torch.device("cuda")
    xtr = torch.tensor(x_train, dtype=torch.float32, device=device)
    xte = torch.tensor(x_test, dtype=torch.float32, device=device)
    ytr = torch.tensor(train_perf, dtype=torch.float32, device=device)
    top_k = min(k, xtr.shape[0])
    out = []
    for start in range(0, xte.shape[0], knn_chunk_size()):
        chunk = xte[start:start + knn_chunk_size()]
        sim = chunk @ xtr.T
        nn_scores, nn_idx = torch.topk(sim, k=top_k, dim=1, largest=True, sorted=False)
        nn_perf = ytr[nn_idx]
        weights = nn_scores.clamp_min(0.0)
        denom = weights.sum(dim=1, keepdim=True)
        weighted_avg = (nn_perf * weights.unsqueeze(-1)).sum(dim=1)
        mean_perf = nn_perf.mean(dim=1)
        scores = torch.where(
            denom > 0,
            weighted_avg / denom.clamp_min(1e-8),
            mean_perf,
        )
        out.append(scores.cpu())
    return torch.cat(out, dim=0).numpy().astype(np.float32)


def result_path(method_name: str, seed: int) -> Path:
    return RESULTS_DIR / f"result_baseline_{method_name}_llmrouterbench_performance_cost_seed{seed}.json"


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))
    if path.name.endswith("_seed42.json"):
        unsuffixed = path.with_name(path.name.replace("_seed42", ""))
        unsuffixed.write_text(json.dumps(payload, indent=2))


def load_seed_embeddings(seed: int, payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return encode_queries_gpu(
        payload["train_queries"],
        payload["test_queries"],
        batch_size=int(os.getenv("GLIDER_EMBED_BATCH_SIZE", "32")),
        embedding_model_name=EMBEDDING_MODEL,
        cache_prefix=f"llmrouterbench_performance_cost_prompt_seed{seed}",
    )


def run_knn(seed: int, payload: dict[str, Any], refs: dict[str, Any], x_train: np.ndarray, x_test: np.ndarray) -> dict[str, Any]:
    curve = alpha_sweep(
        knn_scores_gpu(x_train, payload["train_perf"], x_test, k=10),
        payload["test_perf"],
        payload["test_cost"],
        payload["test_meta"],
    )
    summary = summarize_curve(curve, refs)
    result = {
        "method_name": "Standard-kNN",
        "setting": "LLMRouterBench performance-cost",
        "config_path": str(CONFIG_PATH),
        "split_protocol": {
            "name": "official_prompt_split",
            "train_ratio": TRAIN_RATIO,
            "split_seed": seed,
        },
        "embedding_model": EMBEDDING_MODEL,
        "references": refs,
        "alpha_grid": ALPHAS,
        "curve": curve,
        "summary": summary,
        "baseline": {
            "family": "standard_knn",
            "k": 10,
            "weighted": True,
            "chunk_size": knn_chunk_size(),
        },
    }
    write_result(result_path("knn", seed), result)
    release_cuda_memory()
    return result


def run_routerbench_mlp(
    seed: int,
    payload: dict[str, Any],
    refs: dict[str, Any],
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x_train))
    rng.shuffle(idx)
    n_val = max(256, int(len(idx) * 0.1))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    model, train_meta = train_routerbench_mlp(
        x_train=x_train[tr_idx],
        y_train=payload["train_perf"][tr_idx],
        x_val=x_train[val_idx],
        y_val=payload["train_perf"][val_idx],
        batch_size=int(os.getenv("BASELINE_TRAIN_BATCH_SIZE", "1024")),
    )
    logits = infer_routerbench_mlp(model, x_test)
    curve = alpha_sweep(
        1.0 / (1.0 + np.exp(-logits)),
        payload["test_perf"],
        payload["test_cost"],
        payload["test_meta"],
    )
    summary = summarize_curve(curve, refs)
    result = {
        "method_name": "RouterBench-MLP",
        "setting": "LLMRouterBench performance-cost",
        "config_path": str(CONFIG_PATH),
        "split_protocol": {
            "name": "official_prompt_split",
            "train_ratio": TRAIN_RATIO,
            "split_seed": seed,
        },
        "embedding_model": EMBEDDING_MODEL,
        "references": refs,
        "alpha_grid": ALPHAS,
        "curve": curve,
        "summary": summary,
        "baseline": {
            "family": "routerbench_mlp",
            "hidden_layers": [100, 100],
            "dropout": 0.1,
            "batch_size": int(os.getenv("BASELINE_TRAIN_BATCH_SIZE", "1024")),
            "val_acc": train_meta["val_acc"],
            "val_loss": train_meta["val_loss"],
            "history_tail": train_meta["history_tail"],
        },
    }
    write_result(result_path("routerbench_mlp", seed), result)
    release_cuda_memory()
    return result


def ensure_avengers_split(seed: int, payload: dict[str, Any]) -> dict[str, str]:
    split_dir = AVENGERS_SPLIT_DIR / f"seed{seed}_split{TRAIN_RATIO:.1f}"
    split_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, queries: list[str], meta: dict[int, dict[str, Any]], perf: np.ndarray, cost: np.ndarray, models: list[str]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row_idx, query in enumerate(queries):
                row = {
                    "query": query,
                    "dataset": meta[row_idx]["dataset"],
                    "index": int(meta[row_idx]["index"]),
                    "records": {
                        model: float(perf[row_idx, col])
                        for col, model in enumerate(models)
                    },
                    "usages": {
                        model: {
                            "completion_tokens": 0,
                            "cost": float(cost[row_idx, col]),
                            "prompt_tokens": 0,
                        }
                        for col, model in enumerate(models)
                    },
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def write_baseline_scores(path: Path, meta: dict[int, dict[str, Any]], perf: np.ndarray, models: list[str]) -> None:
        by_model_ds: dict[str, dict[str, list[float]]] = {}
        for model in models:
            by_model_ds[model] = {}
        for row_idx in range(perf.shape[0]):
            dataset = str(meta[row_idx]["dataset"])
            for col, model in enumerate(models):
                by_model_ds[model].setdefault(dataset, []).append(float(perf[row_idx, col]))
        payload_scores = {
            model: {
                dataset: round((sum(scores) / len(scores)) * 100.0, 2)
                for dataset, scores in ds_map.items()
            }
            for model, ds_map in by_model_ds.items()
        }
        path.write_text(json.dumps(payload_scores, indent=2, ensure_ascii=False))

    train_path = split_dir / "train.jsonl"
    test_path = split_dir / "test.jsonl"
    baseline_scores_path = split_dir / "baseline_scores.json"
    write_jsonl(
        train_path,
        payload["train_queries"],
        payload["train_meta"],
        payload["train_perf"],
        payload["train_cost"],
        payload["models"],
    )
    write_jsonl(
        test_path,
        payload["test_queries"],
        payload["test_meta"],
        payload["test_perf"],
        payload["test_cost"],
        payload["models"],
    )
    write_baseline_scores(
        baseline_scores_path,
        payload["test_meta"],
        payload["test_perf"],
        payload["models"],
    )
    return {
        "train": str(train_path),
        "test": str(test_path),
        "baseline_scores": str(baseline_scores_path),
    }


def run_avengers(seed: int, refs: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    split_files = ensure_avengers_split(seed, payload)
    curve = {}
    for alpha in ALPHAS:
        cfg = SimpleClusterConfig(
            train_data_path=split_files["train"],
            test_data_path=split_files["test"],
            baseline_scores_path=split_files["baseline_scores"],
            n_clusters=16,
            seed=seed,
            max_router=1,
            top_k=1,
            beta=9.0,
            max_workers=1,
            cluster_batch_size=2048,
            embedding_batch_size=16,
            embedding_model=EMBEDDING_MODEL,
            embedding_base_url="local",
            embedding_api_key="local",
            embedding_config_path=str(EMBED_CFG),
            shared_embedding_cache_path=str(SHARED_CACHE),
            performance_weight=float(alpha),
            cost_sensitivity=float(1.0 - alpha),
            min_accuracy_threshold=0.0,
            excluded_models=[],
            excluded_datasets=[],
            ood_datasets=[],
            dataset_exclusion_mode="hard",
        )
        router = BalanceClusterRouter(cfg)
        result = router.run_routing()
        curve[str(alpha)] = {
            "sample_avg": float(result.get("all_sample_avg", result["accuracy"])) / 100.0
            if float(result.get("all_sample_avg", result["accuracy"])) > 1.0
            else float(result.get("all_sample_avg", result["accuracy"])),
            "dataset_avg": float(np.mean([
                (float(row["accuracy"]) / 100.0) if float(row["accuracy"]) > 1.0 else float(row["accuracy"])
                for row in result["dataset_performance"].values()
            ])),
            "avg_cost": float(result["cost_analysis"]["avg_cost_per_query"]),
            "total_cost": float(result["cost_analysis"]["total_cost"]),
            "per_dataset": {
                dataset: ((float(row["accuracy"]) / 100.0) if float(row["accuracy"]) > 1.0 else float(row["accuracy"]))
                for dataset, row in result["dataset_performance"].items()
            },
        }
        del router
        del result
        release_cuda_memory()
    summary = summarize_curve(curve, refs)
    result = {
        "method_name": "Avengers-Pro",
        "setting": "LLMRouterBench performance-cost",
        "config_path": str(CONFIG_PATH),
        "split_protocol": {
            "name": "official_prompt_split",
            "train_ratio": TRAIN_RATIO,
            "split_seed": seed,
        },
        "embedding_model": EMBEDDING_MODEL,
        "references": refs,
        "alpha_grid": ALPHAS,
        "curve": curve,
        "summary": summary,
        "baseline": {
            "family": "avengers_pro",
            "n_clusters": 16,
            "top_k": 1,
            "beta": 9.0,
            "train_data_path": split_files["train"],
            "test_data_path": split_files["test"],
            "baseline_scores_path": split_files["baseline_scores"],
        },
    }
    write_result(result_path("avengers", seed), result)
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for performance-cost baseline evaluation.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    AVENGERS_SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        set_seed(seed)
        print(f"===== seed {seed} =====", flush=True)
        payload, refs = load_official_split(seed)
        x_train, x_test = load_seed_embeddings(seed, payload)
        if "knn" in args.methods:
            print("Running official-split Standard-kNN...", flush=True)
            run_knn(seed, payload, refs, x_train, x_test)
        if "routerbench_mlp" in args.methods:
            print("Running official-split RouterBench-MLP...", flush=True)
            run_routerbench_mlp(seed, payload, refs, x_train, x_test)
        if "avengers" in args.methods:
            print("Running official-split Avengers-Pro...", flush=True)
            run_avengers(seed, refs, payload)


if __name__ == "__main__":
    main()
