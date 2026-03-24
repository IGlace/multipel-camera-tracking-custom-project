"""MOT output sink."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import OutputSink, TrackObservation


class MOTSink(OutputSink):
    def __init__(self, subdir: str = "mot") -> None:
        self.subdir = subdir
        self._opened_files: dict[str, object] = {}

    def _file_for_camera(self, output_root: Path, camera_id: str):
        mot_dir = output_root / self.subdir
        mot_dir.mkdir(parents=True, exist_ok=True)
        if camera_id not in self._opened_files:
            self._opened_files[camera_id] = (mot_dir / f"{camera_id}.txt").open("a", encoding="utf-8")
        return self._opened_files[camera_id]

    def write(
        self,
        *,
        output_root: Path,
        timestamp: str,
        frame_index: int,
        images_by_camera: dict[str, np.ndarray],
        observations_by_camera: dict[str, list[TrackObservation]],
    ) -> None:
        del timestamp, images_by_camera
        for camera_id, observations in observations_by_camera.items():
            handle = self._file_for_camera(output_root, camera_id)
            for obs in observations:
                x1, y1, x2, y2 = obs.bbox_xyxy
                width = x2 - x1
                height = y2 - y1
                handle.write(
                    f"{frame_index + 1},{obs.track_id},{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},{obs.confidence:.4f},-1,-1,-1\n"
                )
                handle.flush()

    def close(self) -> None:
        for handle in self._opened_files.values():
            handle.close()
        self._opened_files.clear()
