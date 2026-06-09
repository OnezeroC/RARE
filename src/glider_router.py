"""GLIDER router core components.

GLIDER = Global-Local Inference with Dynamic Evidence Rectification

This module packages the final main-method version distilled from the
method7_v* exploration line.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from common.shared_embedding_cache import SharedEmbeddingCache


EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / 'data' / 'hs_cache'
SHARED_CACHE_DB = CACHE_DIR / 'shared_embedding_cache.sqlite'


def _resolve_sentence_transformer_path(model_name_or_path: str) -> str:
    model_path = Path(model_name_or_path).expanduser()
    if model_path.exists():
        return str(model_path)

    cache_root = Path('/data/yuqihang/.cache/huggingface/hub')
    repo_dir = cache_root / f"models--{model_name_or_path.replace('/', '--')}"
    snapshots_dir = repo_dir / 'snapshots'
    if snapshots_dir.exists():
        snapshots = sorted(p for p in snapshots_dir.iterdir() if p.is_dir())
        if snapshots:
            return str(snapshots[-1])

    return model_name_or_path


def split_cache_paths() -> tuple[Path, Path]:
    return (
        CACHE_DIR / 'exp5_origin_split_cache.npz',
        CACHE_DIR / 'exp5_origin_split_meta.json',
    )


def load_cached_exp5_split():
    cache_npz, cache_meta = split_cache_paths()
    if not cache_npz.exists() or not cache_meta.exists():
        raise FileNotFoundError('exp5 split cache missing.')
    meta = json.loads(cache_meta.read_text())
    with np.load(cache_npz, allow_pickle=True) as data:
        tm = data['tm'].astype(np.float32)
        em = data['em'].astype(np.float32)
        tq = data['tq'].tolist()
        eq = data['eq'].tolist()
    models = meta['models']
    tm_meta = {int(k): v for k, v in meta['tm_meta'].items()}
    em_meta = {int(k): v for k, v in meta['em_meta'].items()}
    return tm, tq, models, tm_meta, em, eq, em_meta


def _embedding_cache_slug(model_name_or_path: str) -> str:
    model_id = Path(model_name_or_path).name or model_name_or_path
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in model_id)


def embedding_cache_paths(
    train_queries: list[str],
    test_queries: list[str],
    *,
    embedding_model_name: str = EMBEDDING_MODEL,
    cache_prefix: str = 'cap_simplex',
) -> tuple[Path, Path]:
    model_slug = _embedding_cache_slug(embedding_model_name)
    train_hash = hashlib.md5(
        (
            f"{cache_prefix}_{model_slug}_train_"
            + "".join(train_queries[:100])
            + str(len(train_queries))
        ).encode()
    ).hexdigest()[:8]
    test_hash = hashlib.md5(
        (
            f"{cache_prefix}_{model_slug}_test_"
            + "".join(test_queries[:100])
            + str(len(test_queries))
        ).encode()
    ).hexdigest()[:8]
    return (
        CACHE_DIR / f'{cache_prefix}_{model_slug}_train_{train_hash}.npz',
        CACHE_DIR / f'{cache_prefix}_{model_slug}_test_{test_hash}.npz',
    )


def encode_queries_gpu(
    train_queries: list[str],
    test_queries: list[str],
    batch_size: int = 256,
    *,
    embedding_model_name: str = EMBEDDING_MODEL,
    cache_prefix: str = 'cap_simplex',
) -> tuple[np.ndarray, np.ndarray]:
    train_cache, test_cache = embedding_cache_paths(
        train_queries,
        test_queries,
        embedding_model_name=embedding_model_name,
        cache_prefix=cache_prefix,
    )
    if train_cache.exists() and test_cache.exists():
        with np.load(train_cache, allow_pickle=True) as data:
            xtr = data['embeddings'].astype(np.float32)
        with np.load(test_cache, allow_pickle=True) as data:
            xte = data['embeddings'].astype(np.float32)
        return xtr, xte

    shared_cache = SharedEmbeddingCache(SHARED_CACHE_DB)
    resolved = _resolve_sentence_transformer_path(embedding_model_name)
    model = SentenceTransformer(resolved, local_files_only=True, device='cuda')

    def fetch_or_encode(queries: list[str]) -> np.ndarray:
        cached = shared_cache.get_many(
            queries,
            model_name=embedding_model_name,
            normalized=True,
        )
        missing = [query for query in queries if query not in cached]
        if missing:
            encoded = model.encode(
                missing,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
                device='cuda',
            ).astype(np.float32)
            shared_cache.put_many(
                missing,
                encoded,
                model_name=embedding_model_name,
                normalized=True,
            )
            for query, embedding in zip(missing, encoded):
                cached[query] = embedding
        return np.stack([cached[query] for query in queries]).astype(np.float32)

    xtr = fetch_or_encode(train_queries)
    xte = fetch_or_encode(test_queries)
    np.savez_compressed(train_cache, embeddings=xtr)
    np.savez_compressed(test_cache, embeddings=xte)
    return xtr, xte


def split_train_val(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    rng.shuffle(idx)
    n_val = max(256, int(len(idx) * 0.1))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return x[tr_idx], y[tr_idx], x[val_idx], y[val_idx]


def per_dataset_accuracy(selections: list[str], matrix: np.ndarray, models: list[str], query_meta: dict) -> dict[str, float]:
    per_correct = defaultdict(int)
    per_total = defaultdict(int)
    for i, sel in enumerate(selections):
        ds = query_meta[i]['dataset']
        per_total[ds] += 1
        if matrix[i, models.index(sel)] > 0:
            per_correct[ds] += 1
    return {ds: per_correct[ds] / per_total[ds] for ds in sorted(per_total)}


class GlobalCapabilityBackbone(nn.Module):
    def __init__(
        self,
        d_in: int,
        n_models: int,
        n_factors: int = 12,
        hidden_dim: int = 640,
        bottleneck_dim: int = 320,
        dropout: float = 0.1,
        temperature: float = 1.0,
        residual_scale: float = 0.4,
    ):
        super().__init__()
        self.temperature = temperature
        self.residual_scale = residual_scale
        self.backbone = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.factor_proj = nn.Linear(hidden_dim, n_factors)
        self.factor_to_model = nn.Parameter(torch.empty(n_factors, n_models))
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim + n_factors, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, n_models),
        )
        self.residual_gate = nn.Sequential(
            nn.Linear(hidden_dim + n_factors, bottleneck_dim // 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim // 2, n_models),
        )
        nn.init.xavier_uniform_(self.factor_to_model)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(x)
        factor_logits = self.factor_proj(h)
        mixture = F.softmax(factor_logits / self.temperature, dim=-1)
        base_logits = mixture @ self.factor_to_model
        residual_in = torch.cat([h, mixture], dim=1)
        residual_raw = self.residual_head(residual_in)
        residual_gate = torch.sigmoid(self.residual_gate(residual_in))
        residual_logits = self.residual_scale * residual_gate * residual_raw
        final_logits = base_logits + residual_logits
        return {
            'hidden': h,
            'mixture': mixture,
            'base_logits': base_logits,
            'final_logits': final_logits,
        }


def routing_margin_loss(logits: torch.Tensor, targets: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    pos_mask = targets > 0.5
    neg_mask = ~pos_mask
    pos_logits = logits.masked_fill(~pos_mask, -1e9).max(dim=1).values
    neg_logits = logits.masked_fill(~neg_mask, -1e9).max(dim=1).values
    valid = pos_mask.any(dim=1)
    if valid.any():
        return F.relu(margin - pos_logits[valid] + neg_logits[valid]).mean()
    return logits.new_tensor(0.0)


def listwise_target_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    denom = targets.sum(dim=1, keepdim=True).clamp(min=1.0)
    soft_targets = targets / denom
    return -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def factor_orthogonality_penalty(factor_to_model: torch.Tensor) -> torch.Tensor:
    w = F.normalize(factor_to_model, dim=1)
    gram = w @ w.T
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return ((gram - eye) ** 2).mean()


def anti_collapse_penalty(mix: torch.Tensor, entropy_floor: float) -> tuple[torch.Tensor, torch.Tensor]:
    ent = -(mix * torch.log(torch.clamp(mix, min=1e-8))).sum(dim=1)
    ent_pen = F.relu(entropy_floor - ent).mean()
    mean_mix = mix.mean(dim=0)
    uniform = torch.full_like(mean_mix, 1.0 / mean_mix.numel())
    balance_pen = F.kl_div(torch.log(torch.clamp(mean_mix, min=1e-8)), uniform, reduction='batchmean')
    return ent_pen, balance_pen


def train_global_backbone(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    n_factors: int = 12,
    hidden_dim: int = 640,
    bottleneck_dim: int = 320,
    temperature: float = 1.0,
    residual_scale: float = 0.4,
    entropy_floor: float = 0.75,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
) -> tuple[GlobalCapabilityBackbone, dict]:
    device = torch.device('cuda')
    model = GlobalCapabilityBackbone(
        d_in=x_train.shape[1],
        n_models=y_train.shape[1],
        n_factors=n_factors,
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
        temperature=temperature,
        residual_scale=residual_scale,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    xt = torch.tensor(x_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    xv = torch.tensor(x_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)

    best_state = None
    best_val_acc = -1.0
    best_val_loss = float('inf')
    history = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), device=device)
        for start in range(0, len(xt), batch_size):
            idx = perm[start:start + batch_size]
            out = model(xt[idx])
            logits = out['final_logits']
            base_logits = out['base_logits']
            mix = out['mixture']
            yb = yt[idx]
            bce = F.binary_cross_entropy_with_logits(logits, yb)
            listwise = listwise_target_loss(logits, yb)
            base_listwise = listwise_target_loss(base_logits, yb)
            margin = routing_margin_loss(logits, yb)
            ent_pen, balance_pen = anti_collapse_penalty(mix, entropy_floor=entropy_floor)
            ortho = factor_orthogonality_penalty(model.factor_to_model)
            loss = (
                0.5 * bce
                + 1.0 * listwise
                + 0.25 * base_listwise
                + 0.35 * margin
                + 0.04 * ent_pen
                + 0.05 * balance_pen
                + 0.03 * ortho
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            out = model(xv)
            val_logits = out['final_logits']
            val_sel = torch.argmax(val_logits, dim=1).cpu().numpy()
            val_acc = float((y_val[np.arange(len(y_val)), val_sel] > 0).mean())
            val_loss = listwise_target_loss(val_logits, yv).item()
            mean_mix = out['mixture'].mean(dim=0).detach().cpu().numpy().astype(np.float32)
        history.append({
            'epoch': epoch + 1,
            'val_acc': val_acc,
            'val_loss': val_loss,
            'max_factor_share': float(mean_mix.max()),
        })
        if val_acc > best_val_acc or (np.isclose(val_acc, best_val_acc) and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, {
        'val_acc': best_val_acc,
        'val_loss': best_val_loss,
        'history_tail': history[-5:],
    }


def infer_logits(model: GlobalCapabilityBackbone, x: np.ndarray) -> np.ndarray:
    device = torch.device('cuda')
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32, device=device)
        logits = model(xt)['final_logits'].detach().cpu().numpy().astype(np.float32)
    return logits


def infer_factor_features(
    model: GlobalCapabilityBackbone,
    x: np.ndarray,
    *,
    feature_kind: str = 'mixture',
    batch_size: int = 2048,
) -> np.ndarray:
    device = torch.device('cuda')
    model.eval()
    xt = torch.tensor(x, dtype=torch.float32, device=device)
    outs = []
    with torch.no_grad():
        for start in range(0, len(xt), batch_size):
            batch_out = model(xt[start:start + batch_size])
            if feature_kind == 'mixture':
                feat = batch_out['mixture']
            elif feature_kind == 'hidden':
                feat = batch_out['hidden']
            elif feature_kind == 'hidden_plus_mixture':
                feat = torch.cat([batch_out['hidden'], batch_out['mixture']], dim=1)
            else:
                raise ValueError(f'Unsupported feature_kind={feature_kind!r}')
            outs.append(feat.detach().cpu())
    return torch.cat(outs, dim=0).numpy().astype(np.float32)


def local_residual_rectify(
    train_embeddings: np.ndarray,
    train_targets: np.ndarray,
    train_logits: np.ndarray,
    query_embeddings: np.ndarray,
    query_logits: np.ndarray,
    *,
    k: int = 24,
    alpha: float = 0.8,
    tau: float = 0.05,
    uncertainty_threshold: float = 0.1,
) -> np.ndarray:
    device = torch.device('cuda')
    tr_emb = torch.tensor(train_embeddings, dtype=torch.float32, device=device)
    q_emb = torch.tensor(query_embeddings, dtype=torch.float32, device=device)
    tr_resid = torch.tensor(
        train_targets - 1.0 / (1.0 + np.exp(-train_logits)),
        dtype=torch.float32,
        device=device,
    )
    q_logits = torch.tensor(query_logits, dtype=torch.float32, device=device)

    sims = q_emb @ tr_emb.T
    top_sim, top_idx = torch.topk(sims, k=min(k, tr_emb.shape[0]), dim=1)
    weights = F.softmax(top_sim / tau, dim=1)
    gathered = tr_resid[top_idx]
    local_resid = (weights.unsqueeze(-1) * gathered).sum(dim=1)

    probs = F.softmax(q_logits, dim=1)
    top2 = torch.topk(probs, k=2, dim=1).values
    margins = top2[:, 0] - top2[:, 1]
    gate = (margins < uncertainty_threshold).float().unsqueeze(1)
    corrected = q_logits + gate * alpha * local_resid
    return corrected.detach().cpu().numpy().astype(np.float32)
