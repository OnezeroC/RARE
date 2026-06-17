from __future__ import annotations

from typing import Any

import torch

import argparse

from src.config.paths import ensure_standard_layout, project_root
from src.pipelines.prepare import dispatch_prepare
from src.pipelines.run import run_performance_cost_pipeline, run_performance_pipeline


def dispatch(args: argparse.Namespace) -> dict[str, Any] | None:
    ensure_standard_layout()
    project_root().joinpath("artifacts", "results").mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    if args.command == "run":
        if args.setting == "performance":
            return run_performance_pipeline(args)
        return run_performance_cost_pipeline(args)

    if args.command == "prepare":
        return dispatch_prepare(args)

    raise ValueError(f"Unhandled command arguments: {args}")
