"""Detection interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class DetectionRecord:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str | None = None
    metadata: dict[str, Any] | None = None


class Detector(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray) -> list[DetectionRecord]:
        """Run object detection on a single image."""
