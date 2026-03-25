"""Ultralytics local tracker adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from .local_tracker import LocalTrackObservation, LocalTracker


class UltralyticsLocalTracker(LocalTracker):
    def __init__(self, model: str, tracker: str = "botsort", tracker_config: str | None = None, **kwargs: Any) -> None:
        self.model_name = model
        self.tracker = tracker
        self.tracker_config = tracker_config
        self.extra_kwargs = kwargs
        self._model = None

    def _lazy_load(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)
        return self._model

    def track(self, frame: np.ndarray, *, camera_id: str, frame_index: int, timestamp: str) -> list[LocalTrackObservation]:
        model = self._lazy_load()
        tracker_arg = self.tracker_config if self.tracker_config else f"{self.tracker}.yaml"
        results = model.track(frame, tracker=tracker_arg, persist=True, verbose=False, **self.extra_kwargs)
        if not results:
            return []
        result = results[0]
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        track_ids = getattr(boxes, "id", None)
        height, width = frame.shape[:2]
        output: list[LocalTrackObservation] = []
        for idx in range(len(boxes)):
            xyxy = tuple(float(x) for x in boxes.xyxy[idx].tolist())
            x1, y1, x2, y2 = xyxy
            center_x = ((x1 + x2) / 2.0) / max(1, width)
            center_y = ((y1 + y2) / 2.0) / max(1, height)
            area = max(1.0, x2 - x1) * max(1.0, y2 - y1)
            if track_ids is not None:
                local_track_id = int(track_ids[idx].item())
            else:
                local_track_id = idx
            class_id = int(boxes.cls[idx].item())
            output.append(
                LocalTrackObservation(
                    camera_id=camera_id,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    local_track_id=local_track_id,
                    bbox_xyxy=xyxy,
                    center_norm=(float(center_x), float(center_y)),
                    area=float(area),
                    confidence=float(boxes.conf[idx].item()),
                    class_id=class_id,
                    class_name=names.get(class_id),
                    metadata={"backend": "ultralytics", "model": self.model_name, "tracker": self.tracker},
                )
            )
        return output
