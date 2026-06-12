#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
import pandas as pd

from run_rare_performance_cost import CONFIG_PATH, TRAIN_RATIO, load_official_split, summarize_curve
from src.glider_router import encode_queries_gpu
from src.rare_shared import EMBEDDING_MODEL, LLMROUTERBENCH_ROOT, POOL_EXP_ROOT, ROOT, set_seed


LLMRB_ROOT = LLMROUTERBENCH_ROOT
sys.path.insert(0, str(POOL_EXP_ROOT))
sys.path.insert(0, str(LLMRB_ROOT))

from baselines.adaptors.embedllm_adaptor import EmbedLLMAdaptor
from baselines.adaptors.graphrouter_adaptor import GraphRouterAdaptor


RESULTS_DIR = ROOT / "results"
EXTRA_BASELINE_DIR = ROOT / "data" / "extra_baselines"
EMBEDDING_CFG = LLMRB_ROOT / "config" / "embedding_config.yaml"
GRAPHROUTER_ADAPTOR_CFG = LLMRB_ROOT / "baselines" / "GraphRouter" / "configs" / "adaptor_config.yaml"
GRAPHROUTER_BASE_CFG = LLMRB_ROOT / "baselines" / "GraphRouter" / "configs" / "config.yaml"
EMBEDLLM_ALGO = LLMRB_ROOT / "baselines" / "EmbedLLM" / "algorithm" / "mf.py"
GRAPHROUTER_RUN = LLMRB_ROOT / "baselines" / "GraphRouter" / "run_exp.py"
EMBEDDING_DEVICE = os.getenv("RARE_BASELINE_EMBED_CUDA_VISIBLE_DEVICES", "7")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run additional official-split performance-cost baselines.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["embedllm", "graphrouter"],
        choices=["embedllm", "graphrouter"],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
    )
    parser.add_argument("--embedllm-epochs", type=int, default=120)
    parser.add_argument("--graphrouter-epochs", type=int, default=300)
    return parser.parse_args()


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def result_path(method_name: str, seed: int) -> Path:
    return RESULTS_DIR / f"result_baseline_{method_name}_llmrouterbench_performance_cost_seed{seed}.json"


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))
    if path.name.endswith("_seed42.json"):
        unsuffixed = path.with_name(path.name.replace("_seed42", ""))
        unsuffixed.write_text(json.dumps(payload, indent=2))


def run_cmd(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    return proc.stdout


def build_query_embeddings(queries: list[str], cache_prefix: str) -> np.ndarray:
    old_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = EMBEDDING_DEVICE
    try:
        emb, _ = encode_queries_gpu(
            queries,
            queries,
            batch_size=8,
            embedding_model_name=EMBEDDING_MODEL,
            cache_prefix=cache_prefix,
        )
        return emb.astype(np.float32)
    finally:
        if old_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_visible


def build_refs(seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, refs = load_official_split(seed)
    return payload, refs


def build_embedllm_curve(output: str) -> dict[str, Any]:
    dataset_avg = None
    sample_avg = None
    per_dataset: dict[str, float] = {}
    total_cost = None
    avg_cost = None

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if "Dataset-Level Average Accuracy:" in line:
            dataset_avg = float(line.rsplit(":", 1)[1].strip())
        elif "Sample-Level Average Accuracy:" in line:
            sample_avg = float(line.rsplit(":", 1)[1].strip())
        elif line.startswith("All datasets:"):
            chunks = line.replace(",", " ").split()
            for chunk in chunks:
                if chunk.startswith("total_cost="):
                    total_cost = float(chunk.split("=", 1)[1])
                elif chunk.startswith("avg_cost="):
                    avg_cost = float(chunk.split("=", 1)[1])
        elif ": acc=" in line and "prompts=" in line and "total_cost=" in line:
            name, rest = line.split(":", 1)
            fields = rest.strip().split()
            acc_field = next((x for x in fields if x.startswith("acc=")), None)
            if acc_field is not None:
                per_dataset[name.strip()] = float(acc_field.split("=", 1)[1])

    if dataset_avg is None or sample_avg is None or avg_cost is None or total_cost is None:
        raise RuntimeError("Failed to parse EmbedLLM output.")

    return {
        "1.0": {
            "dataset_avg": dataset_avg,
            "sample_avg": sample_avg,
            "avg_cost": avg_cost,
            "total_cost": total_cost,
            "per_dataset": per_dataset,
        }
    }


def prepare_embedllm(seed: int) -> dict[str, Path]:
    out_dir = EXTRA_BASELINE_DIR / "embedllm" / f"seed{seed}_split{TRAIN_RATIO:.1f}"
    old_cwd = Path.cwd()
    adaptor = EmbedLLMAdaptor(
        config_path=str(CONFIG_PATH),
        random_seed=seed,
        train_ratio=TRAIN_RATIO,
    )
    try:
        os.chdir(LLMRB_ROOT)
        files = adaptor.convert(output_dir=str(out_dir.parent))
        question_order = Path(files["question_order"])
        embedding_path = question_order.parent / "question_embeddings.pth"
        if not embedding_path.exists():
            df = pd.read_csv(question_order)
            questions = df["prompt"].tolist()
            embeddings = build_query_embeddings(questions, cache_prefix=f"embedllm_seed{seed}_questions")
            torch.save(torch.from_numpy(embeddings), embedding_path)
    finally:
        os.chdir(old_cwd)
    return {
        "train": Path(files["train"]),
        "test": Path(files["test"]),
        "question_embeddings": embedding_path,
        "output_dir": question_order.parent,
    }


def run_embedllm(seed: int, refs: dict[str, Any], epochs: int) -> dict[str, Any]:
    files = prepare_embedllm(seed)
    model_save = files["output_dir"] / "embedllm_router_seed.pt"
    emb_save = files["output_dir"] / "embedllm_model_embeddings.pth"
    env = {
        "WANDB_MODE": "offline",
        "WANDB_SILENT": "true",
    }
    output = run_cmd(
        [
            sys.executable,
            str(EMBEDLLM_ALGO),
            "--train-data-path",
            str(files["train"]),
            "--test-data-path",
            str(files["test"]),
            "--question-embedding-path",
            str(files["question_embeddings"]),
            "--embedding-save-path",
            str(emb_save),
            "--model-save-path",
            str(model_save),
            "--eval-mode",
            "router",
            "--model-num",
            "13",
            "--embedding-dim",
            "256",
            "--batch-size",
            "4096",
            "--num-epochs",
            str(epochs),
            "--learning-rate",
            "1e-4",
            "--wandb-run-name",
            f"embedllm-llmrouterbench-seed{seed}",
        ],
        cwd=LLMRB_ROOT / "baselines" / "EmbedLLM" / "algorithm",
        env=env,
    )
    curve = build_embedllm_curve(output)
    summary = summarize_curve(curve, refs)
    result = {
        "method_name": "EmbedLLM",
        "setting": "LLMRouterBench performance-cost",
        "config_path": str(CONFIG_PATH),
        "split_protocol": {
            "name": "official_prompt_split",
            "train_ratio": TRAIN_RATIO,
            "split_seed": seed,
        },
        "embedding_model": str(EMBEDDING_CFG),
        "references": refs,
        "alpha_grid": [1.0],
        "curve": curve,
        "summary": summary,
        "baseline": {
            "family": "embedllm_mf",
            "train_data_path": str(files["train"]),
            "test_data_path": str(files["test"]),
            "question_embedding_path": str(files["question_embeddings"]),
            "epochs": epochs,
            "embedding_dim": 256,
            "batch_size": 4096,
        },
        "raw_stdout_tail": output.splitlines()[-120:],
    }
    write_result(result_path("embedllm", seed), result)
    release_cuda_memory()
    return result


def prepare_graphrouter(seed: int, epochs: int) -> tuple[Path, Path]:
    out_dir = EXTRA_BASELINE_DIR / "graphrouter" / f"seed{seed}_split{TRAIN_RATIO:.1f}"
    old_cwd = Path.cwd()
    adaptor = GraphRouterAdaptor(
        baseline_config_path=str(CONFIG_PATH),
        graphrouter_config_path=str(GRAPHROUTER_ADAPTOR_CFG),
        embedding_config_path=str(EMBEDDING_CFG),
        random_seed=seed,
    )
    try:
        os.chdir(LLMRB_ROOT)
        files = adaptor.convert(output_dir=str(out_dir))
    finally:
        os.chdir(old_cwd)

    csv_path = Path(files["router_data_csv"])
    parquet_path = Path(files["router_data_parquet"])
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        raise RuntimeError("GraphRouter adaptor did not produce router_data.csv")

    df = pd.read_csv(csv_path)
    if "query_embedding" not in df.columns or df["query_embedding"].isna().all():
        unique_queries = list(dict.fromkeys(df["query"].tolist()))
        query_emb = build_query_embeddings(unique_queries, cache_prefix=f"graphrouter_seed{seed}_queries")
        query_map = {q: e for q, e in zip(unique_queries, query_emb)}
        df["query_embedding"] = df["query"].map(lambda q: "[[" + " ".join(map(str, query_map[q].tolist())) + "]]")

    if "task_description_embedding" not in df.columns or df["task_description_embedding"].isna().all():
        unique_tasks = list(dict.fromkeys(df["task_description"].tolist()))
        task_emb = build_query_embeddings(unique_tasks, cache_prefix=f"graphrouter_seed{seed}_tasks")
        task_map = {q: e for q, e in zip(unique_tasks, task_emb)}
        df["task_description_embedding"] = df["task_description"].map(lambda q: "[[" + " ".join(map(str, task_map[q].tolist())) + "]]")

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    cfg = yaml.safe_load(GRAPHROUTER_BASE_CFG.read_text())
    cfg["saved_router_data_path"] = str(Path(files["router_data_parquet"]))
    cfg["llm_description_path"] = str(Path(files["llm_descriptions_json"]))
    cfg["llm_embedding_path"] = str(Path(files["llm_embedding_pkl"]))
    cfg["wandb_key"] = ""
    cfg["train_epoch"] = int(epochs)
    cfg["seed"] = int(seed)
    cfg["llm_num"] = 13
    cfg["split_ratio"] = [TRAIN_RATIO, 0.0, 1.0 - TRAIN_RATIO]
    cfg["model_path"] = str(out_dir / "best_model.pth")
    cfg_path = out_dir / "config_runtime.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out_dir, cfg_path


def build_graphrouter_curve(output: str) -> dict[str, Any]:
    dataset_avg = None
    sample_avg = None
    total_cost = None
    per_dataset: dict[str, float] = {}

    section = None
    for raw_line in output.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped == "[Summary Metrics]":
            section = "summary"
            continue
        if stripped == "[Per-Task Metrics]":
            section = "tasks"
            continue
        if stripped.startswith("="):
            continue
        if section == "summary":
            if "Dataset-Level Average Accuracy:" in stripped:
                dataset_avg = float(stripped.rsplit(":", 1)[1].strip())
            elif "Sample-Level Average Accuracy:" in stripped:
                sample_avg = float(stripped.rsplit(":", 1)[1].strip())
            elif "Total Cost:" in stripped:
                total_cost = float(stripped.rsplit(":", 1)[1].strip())
        elif section == "tasks" and "Acc:" in stripped and "Total Cost:" in stripped:
            task_name, rest = stripped.split("Acc:", 1)
            acc_val = float(rest.split("|", 1)[0].strip())
            per_dataset[task_name.strip()] = acc_val

    if dataset_avg is None or sample_avg is None or total_cost is None:
        raise RuntimeError("Failed to parse GraphRouter output.")

    avg_cost = total_cost / 3740.0
    return {
        "1.0": {
            "dataset_avg": dataset_avg,
            "sample_avg": sample_avg,
            "avg_cost": avg_cost,
            "total_cost": total_cost,
            "per_dataset": per_dataset,
        }
    }


def run_graphrouter(seed: int, refs: dict[str, Any], epochs: int) -> dict[str, Any]:
    out_dir, cfg_path = prepare_graphrouter(seed, epochs)
    env = {
        "WANDB_MODE": "offline",
        "WANDB_SILENT": "true",
    }
    output = run_cmd(
        [
            sys.executable,
            str(GRAPHROUTER_RUN),
            "--config_file",
            str(cfg_path),
        ],
        cwd=LLMRB_ROOT / "baselines" / "GraphRouter",
        env=env,
    )
    curve = build_graphrouter_curve(output)
    summary = summarize_curve(curve, refs)
    result = {
        "method_name": "GraphRouter",
        "setting": "LLMRouterBench performance-cost",
        "config_path": str(CONFIG_PATH),
        "split_protocol": {
            "name": "official_prompt_split",
            "train_ratio": TRAIN_RATIO,
            "split_seed": seed,
        },
        "embedding_model": str(EMBEDDING_CFG),
        "references": refs,
        "alpha_grid": [1.0],
        "curve": curve,
        "summary": summary,
        "baseline": {
            "family": "graphrouter",
            "runtime_config": str(cfg_path),
            "epochs": epochs,
            "output_dir": str(out_dir),
        },
        "raw_stdout_tail": output.splitlines()[-160:],
    }
    write_result(result_path("graphrouter", seed), result)
    release_cuda_memory()
    return result


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EXTRA_BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        set_seed(seed)
        print(f"===== seed {seed} =====", flush=True)
        _, refs = build_refs(seed)
        if "embedllm" in args.methods:
            print("Running EmbedLLM...", flush=True)
            run_embedllm(seed, refs, args.embedllm_epochs)
        if "graphrouter" in args.methods:
            print("Running GraphRouter...", flush=True)
            run_graphrouter(seed, refs, args.graphrouter_epochs)


if __name__ == "__main__":
    main()
