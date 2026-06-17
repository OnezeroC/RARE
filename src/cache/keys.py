from __future__ import annotations


def embedding_cache_key(*, setting: str, split_seed: int | None, role: str, encoder: str, cache_prefix: str) -> dict[str, str | int | None]:
    return {
        "kind": "embeddings",
        "setting": setting,
        "split_seed": split_seed,
        "role": role,
        "encoder": encoder,
        "cache_prefix": cache_prefix,
    }


def split_cache_key(*, setting: str, split_seed: int, train_ratio: float) -> dict[str, str | int | float]:
    return {
        "kind": "splits",
        "setting": setting,
        "split_seed": split_seed,
        "train_ratio": train_ratio,
    }
