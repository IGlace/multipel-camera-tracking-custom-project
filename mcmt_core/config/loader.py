"""YAML config loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schema import RuntimeConfig


def _read_yaml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError("Top-level config must be a mapping.")
    return data


def load_runtime_config(path: str | Path, app_name: str) -> RuntimeConfig:
    raw = _read_yaml(path)
    raw["app_name"] = app_name
    cfg = RuntimeConfig.model_validate(raw)
    cfg.config_path = Path(path)
    return cfg
