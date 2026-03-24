"""Run manifest writing helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from mcmt_core.config.schema import RuntimeConfig


def write_run_manifest(output_root: Path, cfg: RuntimeConfig, app_name: str, mode: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "app_name": app_name,
        "mode": mode,
        "config_path": str(cfg.config_path) if cfg.config_path else None,
        "require_gpu": cfg.system.require_gpu,
        "verbosity": cfg.logging.verbosity,
    }
    path = output_root / f"{app_name}_{mode}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
