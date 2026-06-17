from __future__ import annotations

import argparse
from typing import Any

from src.config.paths import third_party_root
from src.data.prepare import build_performance_split_cache


def dispatch_prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.prepare_command == "split-cache":
        return build_performance_split_cache(
            llmrouterbench_root=args.llmrouterbench_root or third_party_root(),
            config_path=args.config_path,
            split_seeds=list(args.seeds),
            train_ratio=float(args.train_ratio),
        )
    raise ValueError(f"Unhandled prepare arguments: {args}")
