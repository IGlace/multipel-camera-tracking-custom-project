"""Output manager for sink fan-out."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import OutputSink, TrackObservation


class OutputManager:
    def __init__(self, output_root: Path, sinks: list[OutputSink]) -> None:
        self.output_root = output_root
        self.sinks = sinks
        self.output_root.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        timestamp: str,
        frame_index: int,
        images_by_camera: dict[str, np.ndarray],
        observations_by_camera: dict[str, list[TrackObservation]],
        payload: dict[str, Any] | None = None,
    ) -> None:
        for sink in self.sinks:
            sink.write(
                output_root=self.output_root,
                timestamp=timestamp,
                frame_index=frame_index,
                images_by_camera=images_by_camera,
                observations_by_camera=observations_by_camera,
                payload=payload,
            )

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()
