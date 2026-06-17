from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_env_path(var_name: str, default: Path) -> Path:
    value = os.getenv(var_name)
    if value:
        return Path(value).expanduser().resolve()
    return default.resolve()


def project_root() -> Path:
    return PROJECT_ROOT


def artifacts_root() -> Path:
    return _resolve_env_path("RARE_ARTIFACTS_ROOT", PROJECT_ROOT / "artifacts")


def artifacts_cache_root() -> Path:
    return artifacts_root() / "cache"


def artifacts_prepared_root() -> Path:
    return artifacts_root() / "prepared"


def artifacts_runs_root() -> Path:
    return artifacts_root() / "runs"


def artifacts_results_root() -> Path:
    return artifacts_root() / "results"


def artifacts_reports_root() -> Path:
    return artifacts_root() / "reports"


def third_party_root() -> Path:
    env_override = os.getenv("RARE_LLMROUTERBENCH_ROOT")
    if env_override:
        return Path(env_override).expanduser().resolve()
    preferred = (PROJECT_ROOT / "third_party" / "LLMRouterBench").resolve()
    legacy = (PROJECT_ROOT / "LLMRouterBench").resolve()
    if preferred.exists():
        return preferred
    return legacy


def models_root() -> Path:
    return PROJECT_ROOT / "models"


def embedding_model_path() -> str:
    override = os.getenv("RARE_EMBEDDING_MODEL")
    if override:
        return override
    return str((models_root() / "gte_Qwen2-7B-instruct").resolve())


def hf_home_root() -> Path:
    return _resolve_env_path("HF_HOME", PROJECT_ROOT / ".hf_home")


def hf_modules_cache_root() -> Path:
    return _resolve_env_path("HF_MODULES_CACHE", hf_home_root() / "modules")


def ensure_standard_layout() -> None:
    for path in [
        artifacts_root(),
        artifacts_cache_root(),
        artifacts_prepared_root(),
        artifacts_runs_root(),
        artifacts_results_root(),
        artifacts_reports_root(),
    ]:
        path.mkdir(parents=True, exist_ok=True)
