"""Ultralytics local tracker adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from mcmt_core.detections.ultralytics import UltralyticsDetector
from .local_tracker import LocalTracker


class UltralyticsLocalTracker(LocalTracker):
    def __init__(self, model: str, tracker: str = "botsort", tracker_config: str | None = None, **kwargs: Any) -> None:
        self.detector = UltralyticsDetector(model=model, **kwargs)
        self.tracker = tracker
        self.tracker_config = tracker_config

    def track(self, frame: np.ndarray) -> Any:
        model = self.detector._lazy_load()
        tracker_arg = self.tracker_config if self.tracker_config else f"{self.tracker}.yaml"
        return model.track(frame, tracker=tracker_arg, verbose=False)
