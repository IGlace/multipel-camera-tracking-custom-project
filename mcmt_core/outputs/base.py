"""Output sink interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class TrackObservation:
    camera_id: str
    frame_index: int
    timestamp: str
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str | None = None


class OutputSink(ABC):
    @abstractmethod
    def write(
        self,
        *,
        output_root: Path,
        timestamp: str,
        frame_index: int,
        images_by_camera: dict[str, np.ndarray],
        observations_by_camera: dict[str, list[TrackObservation]],
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Write output artifacts for the current multi-camera batch."""

    def close(self) -> None:
        """Optional cleanup hook."""
