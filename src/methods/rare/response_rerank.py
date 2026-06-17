from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def expert_rank_features(logits: np.ndarray) -> np.ndarray:
    order = np.argsort(-logits, axis=1)
    ranks = np.empty_like(order)
    row = np.arange(logits.shape[0])[:, None]
    ranks[row, order] = np.arange(logits.shape[1])[None, :]
    return ranks.astype(np.int64)


def union_shortlists(expert_logits: list[np.ndarray], top_k: int) -> list[list[int]]:
    out: list[list[int]] = []
    for row_idx in range(expert_logits[0].shape[0]):
        cand: set[int] = set()
        for logits in expert_logits:
            cand.update(np.argsort(-logits[row_idx])[:top_k].tolist())
        out.append(sorted(cand))
    return out


class ShortlistListwiseRanker(nn.Module):
    def __init__(self, d_in: int, hidden_dim: int = 128, dropout: float = 0.1):
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


def _pack_groups(group_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = np.unique(group_ids)
    groups: list[np.ndarray] = []
    max_len = 0
    for gid in unique_groups:
        idx = np.where(group_ids == gid)[0].astype(np.int64)
        groups.append(idx)
        if len(idx) > max_len:
            max_len = len(idx)
    packed_idx = np.full((len(groups), max_len), -1, dtype=np.int64)
    packed_mask = np.zeros((len(groups), max_len), dtype=bool)
    for row_idx, idx in enumerate(groups):
        packed_idx[row_idx, : len(idx)] = idx
        packed_mask[row_idx, : len(idx)] = True
    return packed_idx, packed_mask


def _listwise_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    packed_idx: torch.Tensor,
    packed_mask: torch.Tensor,
) -> torch.Tensor:
    gather_idx = torch.clamp(packed_idx, min=0)
    grouped_logits = logits[gather_idx]
    grouped_targets = targets[gather_idx]
    masked_logits = grouped_logits.masked_fill(~packed_mask, -1e9)
    target_mass = (grouped_targets * packed_mask.float()).sum(dim=1, keepdim=True)
    valid_groups = target_mass.squeeze(1) > 0
    if not bool(valid_groups.any()):
        return torch.tensor(0.0, device=logits.device)
    weight = torch.where(
        packed_mask,
        grouped_targets / torch.clamp(target_mass, min=1.0),
        torch.zeros_like(grouped_targets),
    )
    log_probs = F.log_softmax(masked_logits, dim=1)
    losses = -(weight * log_probs).sum(dim=1)
    return losses[valid_groups].mean()


def _group_accuracy(
    scores: np.ndarray,
    targets: np.ndarray,
    packed_idx: np.ndarray,
    packed_mask: np.ndarray,
) -> float:
    if len(packed_idx) == 0:
        return 0.0
    gather_idx = np.clip(packed_idx, 0, None)
    grouped_scores = scores[gather_idx]
    grouped_scores = np.where(packed_mask, grouped_scores, -1e9)
    grouped_targets = targets[gather_idx]
    chosen_pos = np.argmax(grouped_scores, axis=1)
    chosen_idx = gather_idx[np.arange(len(gather_idx)), chosen_pos]
    valid = (grouped_targets * packed_mask).sum(axis=1) > 0
    if not valid.any():
        return 0.0
    correct = (targets[chosen_idx] > 0) & valid
    return float(correct[valid].mean())


def train_listwise_ranker(
    x_train: np.ndarray,
    y_train: np.ndarray,
    group_ids_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    group_ids_val: np.ndarray,
    *,
    hidden_dim: int = 128,
    dropout: float = 0.1,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> tuple[ShortlistListwiseRanker, dict[str, float]]:
    device = torch.device("cuda")
    model = ShortlistListwiseRanker(d_in=x_train.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    xt = torch.tensor(x_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    xv = torch.tensor(x_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)
    train_packed_idx_np, train_packed_mask_np = _pack_groups(group_ids_train)
    val_packed_idx_np, val_packed_mask_np = _pack_groups(group_ids_val)
    train_packed_idx = torch.tensor(train_packed_idx_np, dtype=torch.long, device=device)
    train_packed_mask = torch.tensor(train_packed_mask_np, dtype=torch.bool, device=device)
    val_packed_idx = torch.tensor(val_packed_idx_np, dtype=torch.long, device=device)
    val_packed_mask = torch.tensor(val_packed_mask_np, dtype=torch.bool, device=device)

    best_state = None
    best_val_acc = -1.0
    best_val_loss = float("inf")
    for _epoch in range(epochs):
        model.train()
        logits = model(xt)
        loss = _listwise_loss(logits, yt, train_packed_idx, train_packed_mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(xv)
            val_loss = float(_listwise_loss(val_logits, yv, val_packed_idx, val_packed_mask).item())
            val_scores = val_logits.detach().cpu().numpy()
            val_acc = _group_accuracy(val_scores, y_val, val_packed_idx_np, val_packed_mask_np)
        if (val_acc > best_val_acc + 1e-12) or (np.isclose(val_acc, best_val_acc) and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {
        "val_group_acc": float(best_val_acc),
        "val_listwise_loss": float(best_val_loss),
        "train_groups": float(len(train_packed_idx_np)),
        "val_groups": float(len(val_packed_idx_np)),
    }


def infer_listwise_ranker(
    model: ShortlistListwiseRanker,
    x: np.ndarray,
) -> np.ndarray:
    device = torch.device("cuda")
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32, device=device)
        return model(xt).detach().cpu().numpy().astype(np.float32)
