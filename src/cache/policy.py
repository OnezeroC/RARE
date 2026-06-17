from __future__ import annotations

from pathlib import Path

from src.cache.registry import stable_key_hash
from src.cache.keys import split_cache_key
from src.config.paths import artifacts_cache_root


def cache_category_dir(kind: str) -> Path:
    return artifacts_cache_root() / kind


def split_cache_paths(*, setting: str, split_seed: int, train_ratio: float) -> tuple[Path, Path]:
    key = split_cache_key(setting=setting, split_seed=split_seed, train_ratio=train_ratio)
    key_hash = stable_key_hash(key)
    base_dir = cache_category_dir("splits") / setting / f"seed{split_seed}"
    base_dir.mkdir(parents=True, exist_ok=True)
    return (
        base_dir / f"{key_hash}_cache.npz",
        base_dir / f"{key_hash}_meta.json",
    )


def resolve_split_cache_paths(*, setting: str, split_seed: int, train_ratio: float) -> tuple[Path, Path]:
    return split_cache_paths(setting=setting, split_seed=split_seed, train_ratio=train_ratio)
