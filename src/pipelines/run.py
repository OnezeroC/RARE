from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.shared import set_seed


def performance_rare_result_exists(args: argparse.Namespace, seed: int) -> bool:
    from src.methods.rare.performance import response_override_seed_result_path

    if args.result_json is not None and len(args.seeds) == 1:
        return Path(args.result_json).exists()
    return response_override_seed_result_path(seed).exists()


def performance_cost_rare_result_exists(args: argparse.Namespace, seed: int) -> bool:
    from src.methods.rare.performance_cost import response_override_result_path

    if args.result_json is not None and len(args.seeds) == 1:
        return Path(args.result_json).exists()
    return response_override_result_path(seed).exists()


def build_rare_perf_args(args: argparse.Namespace, seed: int) -> argparse.Namespace:
    return argparse.Namespace(
        split_seed=seed,
        result_json=args.result_json if len(args.seeds) == 1 else None,
    )


def build_rare_cost_args(args: argparse.Namespace, seed: int) -> argparse.Namespace:
    return argparse.Namespace(
        split_seed=seed,
        result_json=args.result_json if len(args.seeds) == 1 else None,
        local_k=args.local_k,
        local_alpha=args.local_alpha,
        local_tau=args.local_tau,
        local_uncertainty_threshold=args.local_uncertainty_threshold,
        gate_policy=args.gate_policy,
        gate_threshold_quantile=args.gate_threshold_quantile,
        gate_target_rate=args.gate_target_rate,
        backbone_loss_preset=args.backbone_loss_preset,
        retrieval_feature_preset=args.retrieval_feature_preset,
        local_inference_mode=args.local_inference_mode,
        blend_beta=args.blend_beta,
        blend_gamma=args.blend_gamma,
    )


def run_performance_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    from src.methods.rare import performance as rare_perf

    results: dict[str, Any] = {}

    for seed in args.seeds:
        set_seed(seed)
        print(f"===== setting=performance seed={seed} =====", flush=True)
        seed_results: dict[str, Any] = {}

        if args.skip_existing and performance_rare_result_exists(args, seed):
            print("Skipping RARE-Response-Override (result exists)...", flush=True)
        else:
            print("Running RARE-Response-Override...", flush=True)
            seed_results["rare_response_override"] = rare_perf.run_response_override_variant(build_rare_perf_args(args, seed))
        results[str(seed)] = seed_results
    return results


def run_performance_cost_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    from src.methods.rare import performance_cost as rare_cost

    results: dict[str, Any] = {}

    for seed in args.seeds:
        set_seed(seed)
        print(f"===== setting=performance-cost seed={seed} =====", flush=True)
        seed_results: dict[str, Any] = {}

        if args.skip_existing and performance_cost_rare_result_exists(args, seed):
            print("Skipping RARE-Response-Override (result exists)...", flush=True)
        else:
            print("Running RARE-Response-Override...", flush=True)
            seed_results["rare_response_override"] = rare_cost.run_response_override_variant(build_rare_cost_args(args, seed))
        results[str(seed)] = seed_results
    return results
