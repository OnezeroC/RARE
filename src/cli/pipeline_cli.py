from __future__ import annotations

import argparse
from pathlib import Path

from src.shared import PAPER_PERFORMANCE_SEEDS


def add_shared_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--setting", required=True, choices=["performance", "performance-cost"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_PERFORMANCE_SEEDS))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--result-json", type=Path, default=None)
    parser.add_argument("--local-k", type=int, default=24)
    parser.add_argument("--local-alpha", type=float, default=1.0)
    parser.add_argument("--local-tau", type=float, default=0.03)
    parser.add_argument("--local-uncertainty-threshold", type=float, default=2.0)
    parser.add_argument(
        "--gate-policy",
        type=str,
        default="quantile_masked",
        choices=["quantile_masked", "quantile_nonzero_masked", "quantile_raw_disagree", "topk_raw_disagree"],
    )
    parser.add_argument("--gate-threshold-quantile", type=float, default=0.95)
    parser.add_argument("--gate-target-rate", type=float, default=None)
    parser.add_argument(
        "--backbone-loss-preset",
        type=str,
        default="baseline",
        choices=["baseline", "margin_heavy", "listwise_onlyish"],
    )
    parser.add_argument(
        "--retrieval-feature-preset",
        type=str,
        default="perf_only",
        choices=["perf_only", "perf_cost_concat", "utility_0.8"],
    )
    parser.add_argument(
        "--local-inference-mode",
        type=str,
        default="hard_switch",
        choices=["hard_switch", "masked_blend", "soft_blend"],
    )
    parser.add_argument("--blend-beta", type=float, default=0.75)
    parser.add_argument("--blend-gamma", type=float, default=0.5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RARE unified pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run experiments.")
    add_shared_run_args(run_parser)

    prepare_parser = subparsers.add_parser("prepare", help="Prepare benchmark inputs and caches.")
    prepare_subparsers = prepare_parser.add_subparsers(dest="prepare_command", required=True)

    split_cache_parser = prepare_subparsers.add_parser("split-cache", help="Build official split caches.")
    split_cache_parser.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_PERFORMANCE_SEEDS))
    split_cache_parser.add_argument("--llmrouterbench-root", type=Path, default=None)
    split_cache_parser.add_argument("--config-path", type=Path, default=None)
    split_cache_parser.add_argument("--train-ratio", type=float, default=0.7)

    return parser
