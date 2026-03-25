"""Local tracker interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class LocalTrackObservation:
    camera_id: str
    frame_index: int
    timestamp: str
    local_track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    center_norm: tuple[float, float]
    area: float
    confidence: float
    class_id: int
    class_name: str | None = None
    metadata: dict[str, Any] | None = None


class LocalTracker(ABC):
    @abstractmethod
    def track(self, frame: np.ndarray, *, camera_id: str, frame_index: int, timestamp: str) -> list[LocalTrackObservation]:
        """Track objects in a single frame and return normalized local-track observations."""
