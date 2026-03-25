"""Checkpoint-backed attribute extractor."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch

from mcmt_core.io import load_checkpoint_package
from .base import AttributeExtractor
from .simple_model import AttributeBaseline


class PersonAttributeExtractor(AttributeExtractor):
    def __init__(
        self,
        model: AttributeBaseline,
        *,
        attribute_names: list[str],
        device: str | torch.device = "cuda",
        image_size: tuple[int, int] = (256, 128),
        low_threshold: float = 0.3,
        high_threshold: float = 0.7,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.attribute_names = attribute_names
        self.image_size = image_size
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cuda",
    ) -> "PersonAttributeExtractor":
        package = load_checkpoint_package(checkpoint_path, map_location="cpu")
        model_config = package.model_config or {}
        num_classes = int(model_config.get("num_classes", len(package.class_names or [])))
        if num_classes <= 0:
            raise ValueError("Attribute checkpoint package must provide num_classes or class_names.")
        backbone = str(model_config.get("backbone", "resnet50"))
        pretrained_backbone = bool(model_config.get("pretrained_backbone", False))
        dropout = float(model_config.get("dropout", 0.0))
        model = AttributeBaseline(
            num_classes=num_classes,
            backbone=backbone,
            pretrained_backbone=pretrained_backbone,
            dropout=dropout,
        )
        model.load_state_dict(package.state_dict, strict=False)
        attribute_names = package.class_names or [f"attr_{idx}" for idx in range(num_classes)]
        image_size = tuple(model_config.get("image_size", [256, 128]))
        low_threshold = float(model_config.get("low_threshold", 0.3))
        high_threshold = float(model_config.get("high_threshold", 0.7))
        return cls(
            model=model,
            attribute_names=attribute_names,
            device=device,
            image_size=(int(image_size[0]), int(image_size[1])),
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )

    def _preprocess(self, images: Iterable[np.ndarray]) -> torch.Tensor:
        processed = []
        for image in images:
            if image is None:
                continue
            resized = cv2.resize(image, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            processed.append(tensor)
        if not processed:
            return torch.zeros((0, 3, self.image_size[0], self.image_size[1]), dtype=torch.float32, device=self.device)
        batch = torch.stack(processed).to(self.device)
        batch = (batch - self.mean) / self.std
        return batch

    def extract(self, images: Iterable[np.ndarray]) -> Sequence[dict[str, float | int | bool]]:
        batch = self._preprocess(images)
        if batch.shape[0] == 0:
            return []
        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
        outputs = []
        for row in probs:
            item: dict[str, float | int | bool] = {}
            for idx, score in enumerate(row.tolist()):
                name = self.attribute_names[idx] if idx < len(self.attribute_names) else f"attr_{idx}"
                item[f"{name}_score"] = float(score)
                if score >= self.high_threshold:
                    item[name] = True
                elif score <= self.low_threshold:
                    item[name] = False
                else:
                    item[name] = -1
            outputs.append(item)
        return outputs
