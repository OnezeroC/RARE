from __future__ import annotations

import gc
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models.global_local_router import encode_queries_gpu
from src.config.paths import artifacts_results_root
from src.shared import EMBEDDING_MODEL, ROOT


RESULTS_DIR = artifacts_results_root()


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def result_path(method_name: str, seed: int) -> Path:
    return RESULTS_DIR / f"result_baseline_{method_name}_llmrouterbench_performance_cost_seed{seed}.json"


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def build_query_embeddings(
    queries: list[str],
    cache_prefix: str,
    *,
    batch_size: int = 8,
    embedding_device: str = "7",
) -> np.ndarray:
    old_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = embedding_device
    try:
        emb, _ = encode_queries_gpu(
            queries,
            queries,
            batch_size=batch_size,
            embedding_model_name=EMBEDDING_MODEL,
            cache_prefix=cache_prefix,
        )
        return emb.astype(np.float32)
    finally:
        if old_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_visible


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
