"""Standardized checkpoint package helpers.

The platform uses a single package convention for exported model checkpoints so that
trainer modules and runtime adapters can share loading logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(slots=True)
class CheckpointPackage:
    state_dict: dict[str, torch.Tensor]
    metadata: dict[str, Any]
    model_config: dict[str, Any]
    class_names: list[str] | None = None
    raw_package: dict[str, Any] | None = None


def extract_state_dict(package: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ["state_dict", "model_state_dict", "model", "net"]:
        value = package.get(key)
        if isinstance(value, dict):
            return value
    if all(isinstance(v, torch.Tensor) for v in package.values()):
        return package  # raw torch save(state_dict)
    raise KeyError("Could not find a state_dict-like payload in the checkpoint package.")


def load_checkpoint_package(path: str | Path, map_location: str | torch.device = "cpu") -> CheckpointPackage:
    loaded = torch.load(Path(path), map_location=map_location)
    if isinstance(loaded, dict):
        state_dict = extract_state_dict(loaded)
        metadata = dict(loaded.get("metadata", {}))
        model_config = dict(loaded.get("model_config", {}))
        class_names = loaded.get("class_names")
        if class_names is not None:
            class_names = list(class_names)
        return CheckpointPackage(
            state_dict=state_dict,
            metadata=metadata,
            model_config=model_config,
            class_names=class_names,
            raw_package=loaded,
        )
    raise TypeError(f"Unsupported checkpoint payload type: {type(loaded)!r}")


def save_checkpoint_package(
    path: str | Path,
    *,
    state_dict: dict[str, torch.Tensor],
    metadata: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
    class_names: list[str] | None = None,
    extras: dict[str, Any] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "state_dict": state_dict,
        "metadata": metadata or {},
        "model_config": model_config or {},
    }
    if class_names is not None:
        payload["class_names"] = class_names
    if extras:
        payload.update(extras)
    torch.save(payload, target)
    return target
