from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.paths import artifacts_cache_root


REGISTRY_PATH = artifacts_cache_root() / "registry.json"


@dataclass
class CacheRecord:
    key: dict[str, Any]
    key_hash: str
    created_at: str
    producer: str
    files: list[str]
    metadata: dict[str, Any]


def stable_key_hash(key: dict[str, Any]) -> str:
    payload = json.dumps(key, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"records": []}
    return json.loads(REGISTRY_PATH.read_text())


def save_registry(payload: dict[str, Any]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def register_cache(
    *,
    key: dict[str, Any],
    producer: str,
    files: list[Path],
    metadata: dict[str, Any] | None = None,
) -> CacheRecord:
    record = CacheRecord(
        key=key,
        key_hash=stable_key_hash(key),
        created_at=datetime.now(timezone.utc).isoformat(),
        producer=producer,
        files=[str(path.resolve()) for path in files],
        metadata=metadata or {},
    )
    registry = load_registry()
    records = [r for r in registry.get("records", []) if r.get("key_hash") != record.key_hash]
    records.append(asdict(record))
    registry["records"] = records
    save_registry(registry)
    return record


def find_cache_record(key: dict[str, Any]) -> dict[str, Any] | None:
    key_hash = stable_key_hash(key)
    registry = load_registry()
    for record in registry.get("records", []):
        if record.get("key_hash") == key_hash:
            return record
    return None
