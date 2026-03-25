"""Annotated frame output sink."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mcmt_core.visualization.annotate import annotate_frame
from .base import OutputSink, TrackObservation


class AnnotatedFrameSink(OutputSink):
    def __init__(self, subdir: str = "annotated_frames") -> None:
        self.subdir = subdir

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
        del frame_index, payload
        base_dir = output_root / self.subdir
        base_dir.mkdir(parents=True, exist_ok=True)
        for camera_id, image in images_by_camera.items():
            camera_dir = base_dir / camera_id
            camera_dir.mkdir(parents=True, exist_ok=True)
            annotated = annotate_frame(image, observations_by_camera.get(camera_id, []))
            cv2.imwrite(str(camera_dir / f"{timestamp}.jpg"), annotated)
