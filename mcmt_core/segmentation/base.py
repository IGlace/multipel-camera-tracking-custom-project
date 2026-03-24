"""Segmentation interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class SegmentationRecord:
    xyxy: tuple[float, float, float, float] | None
    confidence: float | None
    class_id: int | None
    mask: Any
    metadata: dict[str, Any] | None = None


class Segmentor(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray) -> list[SegmentationRecord]:
        """Run segmentation on a single image."""
