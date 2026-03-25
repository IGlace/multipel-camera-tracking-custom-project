"""Export helpers for attribute checkpoint packages."""

from __future__ import annotations

from pathlib import Path

from mcmt_core.io import load_checkpoint_package, save_checkpoint_package


def export_attribute_package(
    checkpoint_path: str | Path,
    output_package: str | Path,
    *,
    num_classes: int,
    class_names: list[str] | None = None,
    backbone: str = "resnet50",
    image_size: tuple[int, int] = (256, 128),
    low_threshold: float = 0.3,
    high_threshold: float = 0.7,
) -> Path:
    source = load_checkpoint_package(checkpoint_path, map_location="cpu")
    metadata = dict(source.metadata)
    metadata.update({"backend": "person_attribute_recognition", "package_type": "attribute_extractor"})
    model_config = dict(source.model_config)
    model_config.update(
        {
            "model_name": "AttributeBaseline",
            "num_classes": int(num_classes),
            "backbone": backbone,
            "pretrained_backbone": False,
            "image_size": list(image_size),
            "low_threshold": float(low_threshold),
            "high_threshold": float(high_threshold),
        }
    )
    return save_checkpoint_package(
        output_package,
        state_dict=source.state_dict,
        metadata=metadata,
        model_config=model_config,
        class_names=class_names,
    )
