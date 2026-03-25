"""Export helpers for ReID-MGN checkpoint packages."""

from __future__ import annotations

from pathlib import Path

import torch

from mcmt_core.io import load_checkpoint_package, save_checkpoint_package


def export_reid_mgn_package(
    checkpoint_path: str | Path,
    output_package: str | Path,
    *,
    num_classes: int,
    image_size: tuple[int, int] = (384, 128),
) -> Path:
    source = load_checkpoint_package(checkpoint_path, map_location="cpu")
    metadata = dict(source.metadata)
    metadata.update({"backend": "mgn", "package_type": "reid_extractor"})
    model_config = dict(source.model_config)
    model_config.update(
        {
            "model_name": "MGN",
            "num_classes": int(num_classes),
            "pretrained_backbone": False,
            "image_size": list(image_size),
        }
    )
    return save_checkpoint_package(
        output_package,
        state_dict=source.state_dict,
        metadata=metadata,
        model_config=model_config,
    )
