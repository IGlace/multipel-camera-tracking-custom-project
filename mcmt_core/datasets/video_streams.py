"""Video-based multi-camera input adapters for realtime processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from mcmt_core.config.schema import DatasetConfig
from .base import CameraFrame, MultiCameraFrameBatch


@dataclass(slots=True)
class VideoSource:
    camera_id: str
    path: Path


class MultiCameraVideoDataset:
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.sources = self._discover_sources()

    def _discover_sources(self) -> list[VideoSource]:
        if self.config.input_type == "video_file":
            root = Path(self.config.root_dir)
            if not root.is_file():
                raise FileNotFoundError(f"Configured video file does not exist: {root}")
            return [VideoSource(camera_id=root.stem, path=root)]

        if self.config.input_type != "video_folder":
            raise ValueError(
                f"MultiCameraVideoDataset only supports video_file or video_folder, got: {self.config.input_type}"
            )

        root = Path(self.config.root_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"Configured video folder does not exist: {root}")

        if self.config.camera_dirs:
            candidates = [root / rel for rel in self.config.camera_dirs]
        else:
            allowed = {ext.lower() for ext in self.config.video_extensions}
            candidates = sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in allowed])

        sources: list[VideoSource] = []
        allowed = {ext.lower() for ext in self.config.video_extensions}
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in allowed:
                sources.append(VideoSource(camera_id=candidate.stem, path=candidate))
        if not sources:
            raise FileNotFoundError(f"No supported video files were found under {root}")
        return sources

    def __iter__(self):
        captures = [(source, cv2.VideoCapture(str(source.path))) for source in self.sources]
        try:
            frame_index = 0
            while True:
                batch_frames: list[CameraFrame] = []
                for source, cap in captures:
                    ok, image = cap.read()
                    if not ok or image is None:
                        return
                    batch_frames.append(CameraFrame(camera_id=source.camera_id, image_path=source.path, image=image))
                yield MultiCameraFrameBatch(
                    frame_index=frame_index,
                    timestamp=f"{frame_index:06d}",
                    frames=batch_frames,
                )
                frame_index += 1
        finally:
            for _, cap in captures:
                cap.release()
