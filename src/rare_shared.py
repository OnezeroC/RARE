from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def resolve_poolexp_root() -> Path:
    env_override = os.getenv("RARE_POOL_EXP_ROOT")
    if env_override:
        return Path(env_override).expanduser().resolve()

    candidate_paths = [
        ROOT.parent / "PoolExp",
        ROOT / "PoolExp",
    ]
    for path in candidate_paths:
        if path.exists():
            return path.resolve()
    return candidate_paths[0].resolve()


def resolve_llmrouterbench_root() -> Path:
    env_override = os.getenv("RARE_LLMROUTERBENCH_ROOT")
    if env_override:
        return Path(env_override).expanduser().resolve()

    candidate_paths = [
        resolve_poolexp_root() / "LLMRouterBench",
        ROOT.parent / "LLMRouterBench",
    ]
    for path in candidate_paths:
        if path.exists():
            return path.resolve()
    return candidate_paths[0].resolve()


def resolve_embedding_model() -> str:
    env_override = os.getenv("RARE_EMBEDDING_MODEL")
    if env_override:
        return env_override

    candidate_paths = [
        resolve_poolexp_root() / "models" / "gte_Qwen2-7B-instruct",
        ROOT.parent / "models" / "gte_Qwen2-7B-instruct",
    ]
    for path in candidate_paths:
        if path.exists():
            return str(path)
    return str(candidate_paths[0])


POOL_EXP_ROOT = resolve_poolexp_root()
LLMROUTERBENCH_ROOT = resolve_llmrouterbench_root()
EMBEDDING_MODEL = resolve_embedding_model()
PAPER_PERFORMANCE_SEEDS = (42, 999, 2024, 2025, 3407)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def performance_split_cache_paths(split_seed: int) -> tuple[Path, Path]:
    cache_dir = ROOT / "data" / "hs_cache"
    return (
        cache_dir / f"llmrouterbench_performance_prompt_split_seed{split_seed}_cache.npz",
        cache_dir / f"llmrouterbench_performance_prompt_split_seed{split_seed}_meta.json",
    )


def load_official_cached_split(
    split_seed: int = 42,
) -> tuple[np.ndarray, list[str], dict[int, dict[str, object]], np.ndarray, list[str], dict[int, dict[str, object]], list[str]]:
    perf_split_cache_npz, perf_split_cache_meta = performance_split_cache_paths(split_seed)
    if not perf_split_cache_npz.exists() or not perf_split_cache_meta.exists():
        raise FileNotFoundError(
            f"Missing official split cache for seed {split_seed}: {perf_split_cache_npz} / {perf_split_cache_meta}"
        )
    with np.load(perf_split_cache_npz, allow_pickle=True) as data:
        train_matrix = data["train_matrix"].astype(np.float32)
        test_matrix = data["test_matrix"].astype(np.float32)
        train_queries = data["train_queries"].tolist()
        test_queries = data["test_queries"].tolist()
    meta = json.loads(perf_split_cache_meta.read_text())
    models = meta["models"]
    train_meta = {int(k): v for k, v in meta["train_meta"].items()}
    test_meta = {int(k): v for k, v in meta["test_meta"].items()}
    return train_matrix, train_queries, train_meta, test_matrix, test_queries, test_meta, models


def split_indices(n_items: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n_items)
    rng.shuffle(idx)
    n_val = max(256, int(n_items * 0.1))
    return idx[n_val:], idx[:n_val]


def accuracy_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    pred = np.argmax(logits, axis=1)
    return float((targets[np.arange(len(targets)), pred] > 0).mean())


def per_dataset_accuracy(
    pred_idx: np.ndarray,
    targets: np.ndarray,
    models: list[str],
    query_meta: dict[int, dict[str, object]],
) -> dict[str, float]:
    correct: dict[str, int] = {}
    total: dict[str, int] = {}
    for row_idx, model_idx in enumerate(pred_idx):
        ds = str(query_meta[row_idx]["dataset"])
        total[ds] = total.get(ds, 0) + 1
        if targets[row_idx, int(model_idx)] > 0:
            correct[ds] = correct.get(ds, 0) + 1
    return {ds: correct.get(ds, 0) / total[ds] for ds in sorted(total)}


def dataset_accuracy_from_logits(
    logits: np.ndarray,
    targets: np.ndarray,
    models: list[str],
    query_meta: dict[int, dict[str, object]],
) -> tuple[dict[str, float], float]:
    pred = np.argmax(logits, axis=1)
    per_ds = per_dataset_accuracy(pred, targets, models, query_meta)
    return per_ds, float(np.mean(list(per_ds.values())))


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def stage1_gate_features(logits: np.ndarray, mixtures: np.ndarray) -> np.ndarray:
    probs = softmax_np(logits)
    top3 = np.partition(probs, -3, axis=1)[:, -3:]
    top3.sort(axis=1)
    p1 = top3[:, 2]
    p2 = top3[:, 1]
    p3 = top3[:, 0]
    margin12 = p1 - p2
    margin13 = p1 - p3
    entropy = -(probs * np.log(np.clip(probs, 1e-8, None))).sum(axis=1)

    mix_top2 = np.partition(mixtures, -2, axis=1)[:, -2:]
    mix_top2.sort(axis=1)
    mix_max = mix_top2[:, 1]
    mix_gap = mix_top2[:, 1] - mix_top2[:, 0]
    mix_entropy = -(mixtures * np.log(np.clip(mixtures, 1e-8, None))).sum(axis=1)

    pred_idx = np.argmax(logits, axis=1)
    pred_onehot = np.eye(logits.shape[1], dtype=np.float32)[pred_idx]

    dense = np.column_stack([
        p1,
        p2,
        p3,
        margin12,
        margin13,
        entropy,
        logits.max(axis=1),
        logits.mean(axis=1),
        logits.std(axis=1),
        mix_max,
        mix_gap,
        mix_entropy,
        mixtures.std(axis=1),
    ]).astype(np.float32)
    return np.concatenate([dense, pred_onehot], axis=1).astype(np.float32)


def standardize_fit(train_x: np.ndarray, other_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0, keepdims=True)
    sigma = train_x.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    return ((train_x - mu) / sigma).astype(np.float32), ((other_x - mu) / sigma).astype(np.float32)


class RiskGateMLP(nn.Module):
    def __init__(self, d_in: int, hidden_dim: int = 96, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def train_risk_gate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden_dim: int = 96,
    dropout: float = 0.1,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
) -> tuple[RiskGateMLP, dict[str, Any]]:
    device = torch.device("cuda")
    model = RiskGateMLP(d_in=x_train.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    xt = torch.tensor(x_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    xv = torch.tensor(x_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)

    pos = float(y_train.sum())
    neg = float(len(y_train) - y_train.sum())
    pos_weight = torch.tensor(max(1.0, neg / max(pos, 1.0)), dtype=torch.float32, device=device)

    best_state = None
    best_val_loss = float("inf")
    best_val_auc = -1.0
    history = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), device=device)
        for start in range(0, len(xt), batch_size):
            idx = perm[start:start + batch_size]
            logits = model(xt[idx])
            loss = F.binary_cross_entropy_with_logits(logits, yt[idx], pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(xv)
            val_loss = F.binary_cross_entropy_with_logits(val_logits, yv, pos_weight=pos_weight).item()
            val_probs = torch.sigmoid(val_logits).detach().cpu().numpy()
            val_pred = (val_probs >= 0.5).astype(np.float32)
            val_acc = float((val_pred == y_val).mean())
            if y_val.sum() > 0 and y_val.sum() < len(y_val):
                rank = np.argsort(np.argsort(val_probs))
                pos_ranks = rank[y_val > 0]
                n_pos = int(y_val.sum())
                n_neg = int(len(y_val) - n_pos)
                auc = float((pos_ranks.sum() - n_pos * (n_pos - 1) / 2) / max(1, n_pos * n_neg))
            else:
                auc = 0.5

        history.append({
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": auc,
        })
        if (auc > best_val_auc + 1e-12) or (np.isclose(auc, best_val_auc) and val_loss < best_val_loss):
            best_val_auc = auc
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {
        "val_loss": best_val_loss,
        "val_auc": best_val_auc,
        "history_tail": history[-5:],
    }


def infer_risk(model: RiskGateMLP, x: np.ndarray) -> np.ndarray:
    device = torch.device("cuda")
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32, device=device)
        probs = torch.sigmoid(model(xt)).detach().cpu().numpy().astype(np.float32)
    return probs


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_cost_rows(
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
