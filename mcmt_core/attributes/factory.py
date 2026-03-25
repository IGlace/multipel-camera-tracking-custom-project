"""Factory helpers for building attribute extractors from runtime config."""

from __future__ import annotations

from mcmt_core.config.schema import AttributesConfig
from .person_attributes import PersonAttributeExtractor
from .simple_model import AttributeBaseline


def build_attribute_extractor(config: AttributesConfig, *, device: str = "cuda"):
    if config.backend == "person_attribute_recognition" and config.weights_path:
        return PersonAttributeExtractor.from_checkpoint(config.weights_path, device=device)
    if config.backend == "person_attribute_recognition":
        names = []
        if config.attribute_names_path:
            names = [line.strip() for line in config.attribute_names_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not names:
            names = [f"attr_{idx}" for idx in range(26)]
        model = AttributeBaseline(num_classes=len(names), backbone="resnet50", pretrained_backbone=False)
        return PersonAttributeExtractor(model=model, attribute_names=names, device=device)
    raise NotImplementedError(f"Unsupported attributes backend: {config.backend}")
