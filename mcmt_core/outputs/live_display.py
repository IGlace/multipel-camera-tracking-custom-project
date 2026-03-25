"""Live grid display sink for realtime experimentation."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from mcmt_core.visualization.annotate import annotate_frame
from .base import OutputSink, TrackObservation


class LiveGridDisplaySink(OutputSink):
    def __init__(self, window_name: str = "MCMT Live Grid", fullscreen: bool = True) -> None:
        self.window_name = window_name
        self.fullscreen = fullscreen
        self._initialized = False

    def _ensure_window(self) -> None:
        if self._initialized:
            return
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self._initialized = True

    def _compose_grid(self, images_by_camera: dict[str, np.ndarray], observations_by_camera: dict[str, list[TrackObservation]]) -> np.ndarray:
        annotated_items: list[np.ndarray] = []
        target_h = 0
        target_w = 0
        for camera_id, image in images_by_camera.items():
            annotated = annotate_frame(image, observations_by_camera.get(camera_id, []))
            cv2.putText(annotated, camera_id, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            annotated_items.append(annotated)
            target_h = max(target_h, annotated.shape[0])
            target_w = max(target_w, annotated.shape[1])
        if not annotated_items:
            return np.zeros((64, 64, 3), dtype=np.uint8)
        resized = [cv2.resize(img, (target_w, target_h)) for img in annotated_items]
        cols = math.ceil(math.sqrt(len(resized)))
        rows = math.ceil(len(resized) / cols)
        blank = np.zeros_like(resized[0])
        while len(resized) < rows * cols:
            resized.append(blank.copy())
        row_images = []
        for row_idx in range(rows):
            row = resized[row_idx * cols : (row_idx + 1) * cols]
            row_images.append(np.hstack(row))
        return np.vstack(row_images)

    def write(
        self,
        *,
        output_root: Path,
        timestamp: str,
        frame_index: int,
        images_by_camera: dict[str, np.ndarray],
        observations_by_camera: dict[str, list[TrackObservation]],
    ) -> None:
        del output_root, timestamp, frame_index
        self._ensure_window()
        grid = self._compose_grid(images_by_camera, observations_by_camera)
        cv2.imshow(self.window_name, grid)
        cv2.waitKey(1)

    def close(self) -> None:
        if self._initialized:
            cv2.destroyWindow(self.window_name)
            self._initialized = False
