"""GLIDER-v2: expert-gated dynamic fusion for LLM routing.

This module keeps GLIDER's global-local structure, then adds two
complementary routing experts:

- weighted kNN expert over train embeddings
- RouterBench-style MLP expert

A lightweight fusion head learns when to trust each expert and applies a
final residual correction over the blended logits.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.glider_router import listwise_target_loss, routing_margin_loss


class RouterBenchMLPExpert(nn.Module):
    def __init__(self, d_in: int, n_models: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_models),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def routing_accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return float(
        (targets[torch.arange(len(targets), device=targets.device), pred] > 0)
        .float()
        .mean()
        .item()
    )


def train_routerbench_mlp_expert(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    epochs: int = 120,
    batch_size: int = 2048,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
) -> tuple[RouterBenchMLPExpert, dict[str, Any]]:
    device = torch.device('cuda')
    model = RouterBenchMLPExpert(
        d_in=x_train.shape[1],
        n_models=y_train.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    xt = torch.tensor(x_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    xv = torch.tensor(x_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)

    best_state = None
    best_val_acc = -1.0
    best_val_loss = float('inf')
    bad_epochs = 0
    history_tail: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), device=device)
        for start in range(0, len(xt), batch_size):
            idx = perm[start:start + batch_size]
            logits = model(xt[idx])
            loss = loss_fn(logits, yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(xv)
            val_loss = float(loss_fn(val_logits, yv).item())
            val_acc = routing_accuracy_from_logits(val_logits, yv)

        history_tail.append({
            'epoch': epoch + 1,
            'val_loss': val_loss,
            'val_acc': val_acc,
        })
        if len(history_tail) > 10:
            history_tail.pop(0)

        improved = (val_acc > best_val_acc) or (
            abs(val_acc - best_val_acc) < 1e-8 and val_loss < best_val_loss
        )
        if improved:
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_val_acc = val_acc
            best_val_loss = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return model, {
        'hidden_dim': hidden_dim,
        'dropout': dropout,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
        'patience': patience,
        'val_acc': best_val_acc,
        'val_loss': best_val_loss,
        'history_tail': history_tail,
    }


def infer_routerbench_mlp_expert(
    model: RouterBenchMLPExpert,
    x_test: np.ndarray,
    *,
    batch_size: int = 2048,
) -> np.ndarray:
    device = torch.device('cuda')
    xt = torch.tensor(x_test, dtype=torch.float32, device=device)
    outs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(xt), batch_size):
            outs.append(model(xt[start:start + batch_size]).cpu())
    return torch.cat(outs, dim=0).numpy().astype(np.float32)


def gpu_weighted_knn_scores(
    x_train: np.ndarray,
    train_matrix: np.ndarray,
    x_query: np.ndarray,
    *,
    k: int = 24,
    tau: float = 0.05,
    chunk_size: int = 1024,
) -> np.ndarray:
    device = torch.device('cuda')
    xtr = torch.tensor(x_train, dtype=torch.float32, device=device)
    xq = torch.tensor(x_query, dtype=torch.float32, device=device)
    ytr = torch.tensor(train_matrix, dtype=torch.float32, device=device)
    top_k = min(k, xtr.shape[0])

    outs: list[torch.Tensor] = []
    for start in range(0, xq.shape[0], chunk_size):
        q = xq[start:start + chunk_size]
        sim = q @ xtr.T
        nn_scores, nn_idx = torch.topk(sim, k=top_k, dim=1)
        nn_perf = ytr[nn_idx]
        weights = F.softmax(nn_scores / tau, dim=1)
        scores = (nn_perf * weights.unsqueeze(-1)).sum(dim=1)
        outs.append(scores.cpu())
    return torch.cat(outs, dim=0).numpy().astype(np.float32)


def _normalize_entropy(probs: torch.Tensor) -> torch.Tensor:
    entropy = -(probs * torch.log(torch.clamp(probs, min=1e-8))).sum(dim=-1)
    return entropy / np.log(probs.shape[-1])


class ExpertGatedFusionRouter(nn.Module):
    def __init__(
        self,
        d_in: int,
        n_models: int,
        n_experts: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        residual_scale: float = 0.35,
    ) -> None:
        super().__init__()
        self.n_models = n_models
        self.n_experts = n_experts
        self.residual_scale = residual_scale
        pair_dim = n_experts * (n_experts - 1) // 2
        stat_dim = n_experts * 3 + pair_dim
        self.gate_net = nn.Sequential(
            nn.Linear(d_in + stat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_experts),
        )
        self.residual_net = nn.Sequential(
            nn.Linear(d_in + n_experts * n_models, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_models),
        )
        self.residual_gate = nn.Sequential(
            nn.Linear(d_in + stat_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_models),
        )

    def forward(self, x: torch.Tensor, expert_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        probs = F.softmax(expert_logits, dim=-1)
        max_prob = probs.max(dim=-1).values
        top2 = torch.topk(probs, k=2, dim=-1).values
        margins = top2[..., 0] - top2[..., 1]
        entropy = _normalize_entropy(probs)
        top_idx = probs.argmax(dim=-1)
        agreements = []
        for i in range(self.n_experts):
            for j in range(i + 1, self.n_experts):
                agreements.append((top_idx[:, i] == top_idx[:, j]).float().unsqueeze(1))
        pair_agreement = torch.cat(agreements, dim=1) if agreements else x.new_zeros((x.shape[0], 0))
        stats = torch.cat([max_prob, margins, entropy, pair_agreement], dim=1)

        gate_input = torch.cat([x, stats], dim=1)
        gate_logits = self.gate_net(gate_input)
        gate_weights = F.softmax(gate_logits, dim=1)
        blended = (gate_weights.unsqueeze(-1) * expert_logits).sum(dim=1)

        residual_input = torch.cat([x, expert_logits.flatten(start_dim=1)], dim=1)
        residual = self.residual_net(residual_input)
        residual_gate = torch.sigmoid(self.residual_gate(gate_input))
        final_logits = blended + self.residual_scale * residual_gate * residual
        return {
            'gate_weights': gate_weights,
            'blended_logits': blended,
            'final_logits': final_logits,
        }


def _gate_supervision_targets(expert_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pred_idx = torch.argmax(expert_logits, dim=-1)
    batch_idx = torch.arange(targets.shape[0], device=targets.device).unsqueeze(1)
    expert_correct = targets[batch_idx, pred_idx].float()
    fallback = torch.full_like(expert_correct, 1.0 / expert_correct.shape[1])
    denom = expert_correct.sum(dim=1, keepdim=True)
    return torch.where(
        denom > 0,
        expert_correct / denom.clamp_min(1.0),
        fallback,
    )


def train_fusion_router(
    x_train: np.ndarray,
    expert_logits_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    expert_logits_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden_dim: int = 512,
    dropout: float = 0.1,
    residual_scale: float = 0.35,
    epochs: int = 160,
    batch_size: int = 1024,
    lr: float = 8e-4,
    weight_decay: float = 1e-4,
    patience: int = 24,
) -> tuple[ExpertGatedFusionRouter, dict[str, Any]]:
    device = torch.device('cuda')
    n_experts = expert_logits_train.shape[1]
    n_models = y_train.shape[1]
    model = ExpertGatedFusionRouter(
        d_in=x_train.shape[1],
        n_models=n_models,
        n_experts=n_experts,
        hidden_dim=hidden_dim,
        dropout=dropout,
        residual_scale=residual_scale,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    xt = torch.tensor(x_train, dtype=torch.float32, device=device)
    et = torch.tensor(expert_logits_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    xv = torch.tensor(x_val, dtype=torch.float32, device=device)
    ev = torch.tensor(expert_logits_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)

    best_state = None
    best_val_acc = -1.0
    best_val_loss = float('inf')
    bad_epochs = 0
    history_tail: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), device=device)
        for start in range(0, len(xt), batch_size):
            idx = perm[start:start + batch_size]
            out = model(xt[idx], et[idx])
            logits = out['final_logits']
            gate_weights = out['gate_weights']
            targets = yt[idx]
            gate_targets = _gate_supervision_targets(et[idx], targets)
            gate_loss = -(gate_targets * torch.log(torch.clamp(gate_weights, min=1e-8))).sum(dim=1).mean()
            bce = F.binary_cross_entropy_with_logits(logits, targets)
            listwise = listwise_target_loss(logits, targets)
            margin = routing_margin_loss(logits, targets)
            blended_listwise = listwise_target_loss(out['blended_logits'], targets)
            mean_gate = gate_weights.mean(dim=0)
            gate_balance = F.kl_div(
                torch.log(torch.clamp(mean_gate, min=1e-8)),
                torch.full_like(mean_gate, 1.0 / mean_gate.numel()),
                reduction='batchmean',
            )
            loss = (
                0.45 * bce
                + 1.0 * listwise
                + 0.30 * margin
                + 0.35 * gate_loss
                + 0.20 * blended_listwise
                + 0.02 * gate_balance
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            out = model(xv, ev)
            val_logits = out['final_logits']
            val_loss = float(listwise_target_loss(val_logits, yv).item())
            val_acc = routing_accuracy_from_logits(val_logits, yv)
            mean_gate = out['gate_weights'].mean(dim=0).detach().cpu().numpy().astype(np.float32)
        history_tail.append({
            'epoch': epoch + 1,
            'val_acc': val_acc,
            'val_loss': val_loss,
            'max_gate_share': float(mean_gate.max()),
        })
        if len(history_tail) > 10:
            history_tail.pop(0)

        improved = (val_acc > best_val_acc) or (
            abs(val_acc - best_val_acc) < 1e-8 and val_loss < best_val_loss
        )
        if improved:
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_val_acc = val_acc
            best_val_loss = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return model, {
        'hidden_dim': hidden_dim,
        'dropout': dropout,
        'residual_scale': residual_scale,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
        'patience': patience,
        'val_acc': best_val_acc,
        'val_loss': best_val_loss,
        'history_tail': history_tail,
    }


def infer_fusion_router(
    model: ExpertGatedFusionRouter,
    x_query: np.ndarray,
    expert_logits: np.ndarray,
    *,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device('cuda')
    xq = torch.tensor(x_query, dtype=torch.float32, device=device)
    eq = torch.tensor(expert_logits, dtype=torch.float32, device=device)
    logits_out: list[torch.Tensor] = []
    gates_out: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(xq), batch_size):
            out = model(xq[start:start + batch_size], eq[start:start + batch_size])
            logits_out.append(out['final_logits'].cpu())
            gates_out.append(out['gate_weights'].cpu())
    return (
        torch.cat(logits_out, dim=0).numpy().astype(np.float32),
        torch.cat(gates_out, dim=0).numpy().astype(np.float32),
    )
