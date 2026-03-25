"""Annotated video output sink."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from mcmt_core.visualization.annotate import annotate_frame
from .base import OutputSink, TrackObservation


class AnnotatedVideoSink(OutputSink):
    def __init__(self, subdir: str = "annotated_videos", fps: int = 20) -> None:
        self.subdir = subdir
        self.fps = fps
        self._writers: dict[str, cv2.VideoWriter] = {}

    def _writer_for_camera(self, output_root: Path, camera_id: str, frame_shape: tuple[int, int, int]):
        base_dir = output_root / self.subdir
        base_dir.mkdir(parents=True, exist_ok=True)
        if camera_id not in self._writers:
            height, width = frame_shape[:2]
            path = base_dir / f"{camera_id}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writers[camera_id] = cv2.VideoWriter(str(path), fourcc, self.fps, (width, height))
        return self._writers[camera_id]

    def write(
        self,
        *,
        output_root: Path,
        timestamp: str,
        frame_index: int,
        images_by_camera: dict[str, np.ndarray],
        observations_by_camera: dict[str, list[TrackObservation]],
    ) -> None:
        del timestamp, frame_index
        for camera_id, image in images_by_camera.items():
            writer = self._writer_for_camera(output_root, camera_id, image.shape)
            annotated = annotate_frame(image, observations_by_camera.get(camera_id, []))
            writer.write(annotated)

    def close(self) -> None:
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()
