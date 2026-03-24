"""Synchronized multi-camera frame-folder adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from mcmt_core.config.schema import DatasetConfig
from .base import CameraFrame, MultiCameraFrameBatch


@dataclass(slots=True)
class CameraFolder:
    camera_id: str
    directory: Path
    files: list[Path]


class MultiCameraFrameDataset:
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.root_dir = config.root_dir
        self.camera_folders = self._discover_camera_folders()
        self.length = self._resolve_length()

    def _discover_camera_folders(self) -> list[CameraFolder]:
        if self.config.camera_dirs:
            candidates = [self.root_dir / rel for rel in self.config.camera_dirs]
        else:
            candidates = sorted([p for p in self.root_dir.iterdir() if p.is_dir()])
        if not candidates:
            raise FileNotFoundError(
                f"No camera directories were found under dataset root: {self.root_dir}"
            )

        folders: list[CameraFolder] = []
        allowed_suffixes = {ext.lower() for ext in self.config.image_extensions}
        for directory in candidates:
            files = sorted(
                [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in allowed_suffixes]
            )
            if not files:
                continue
            folders.append(CameraFolder(camera_id=directory.name, directory=directory, files=files))
        if not folders:
            raise FileNotFoundError(
                f"No image files were found in the discovered camera directories under {self.root_dir}"
            )
        return folders

    def _resolve_length(self) -> int:
        lengths = [len(folder.files) for folder in self.camera_folders]
        if self.config.sync_strategy == "strict" and len(set(lengths)) != 1:
            raise ValueError(
                f"Camera folders do not have the same number of frames under strict sync: {lengths}"
            )
        return min(lengths)

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        for frame_index in range(self.length):
            frames: list[CameraFrame] = []
            stems = []
            for folder in self.camera_folders:
                image_path = folder.files[frame_index]
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"Failed to read image: {image_path}")
                frames.append(CameraFrame(camera_id=folder.camera_id, image_path=image_path, image=image))
                stems.append(image_path.stem)
            timestamp = stems[0] if stems and len(set(stems)) == 1 else f"{frame_index:06d}"
            yield MultiCameraFrameBatch(frame_index=frame_index, timestamp=timestamp, frames=frames)
