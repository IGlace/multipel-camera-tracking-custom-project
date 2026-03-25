"""MGN extractor with standardized checkpoint-package loading."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from mcmt_core.io import load_checkpoint_package
from .base import ReIDExtractor
from .mgn_model import MGN


class MGNReIDExtractor(ReIDExtractor):
    def __init__(
        self,
        model: MGN | None = None,
        *,
        device: str | torch.device = "cuda",
        normalize_embeddings: bool = True,
        image_size: tuple[int, int] = (384, 128),
    ) -> None:
        self.device = torch.device(device)
        self.model = model or MGN()
        self.model.to(self.device)
        self.model.eval()
        self.normalize_embeddings = normalize_embeddings
        self.image_size = image_size
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cuda",
        normalize_embeddings: bool = True,
    ) -> "MGNReIDExtractor":
        package = load_checkpoint_package(checkpoint_path, map_location="cpu")
        model_config = package.model_config or {}
        model = MGN(
            num_classes=int(model_config.get("num_classes", 751)),
            pretrained_backbone=bool(model_config.get("pretrained_backbone", False)),
        )
        model.load_state_dict(package.state_dict, strict=False)
        image_size = tuple(model_config.get("image_size", [384, 128]))
        return cls(
            model=model,
            device=device,
            normalize_embeddings=normalize_embeddings,
            image_size=(int(image_size[0]), int(image_size[1])),
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

    def extract(self, images: Iterable[np.ndarray]) -> np.ndarray:
        batch = self._preprocess(images)
        if batch.shape[0] == 0:
            return np.zeros((0, 2048), dtype=np.float32)
        with torch.no_grad():
            embeddings = self.model.forward_features(batch)
            if self.normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().numpy().astype(np.float32)
