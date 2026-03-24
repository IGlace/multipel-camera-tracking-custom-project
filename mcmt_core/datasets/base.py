"""Common dataset dataclasses for multi-camera input handling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class CameraFrame:
    camera_id: str
    image_path: Path
    image: np.ndarray


@dataclass(slots=True)
class MultiCameraFrameBatch:
    frame_index: int
    timestamp: str
    frames: list[CameraFrame]
