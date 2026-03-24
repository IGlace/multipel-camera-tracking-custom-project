"""Ultralytics-backed detector wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import DetectionRecord, Detector


class UltralyticsDetector(Detector):
    def __init__(self, model: str, confidence: float = 0.25, iou: float = 0.45, **kwargs: Any) -> None:
        self.model_name = model
        self.confidence = confidence
        self.iou = iou
        self.extra_kwargs = kwargs
        self._model = None

    def _lazy_load(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)
        return self._model

    def predict(self, image: np.ndarray) -> list[DetectionRecord]:
        model = self._lazy_load()
        results = model.predict(image, conf=self.confidence, iou=self.iou, verbose=False, **self.extra_kwargs)
        if not results:
            return []
        result = results[0]
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        output: list[DetectionRecord] = []
        for idx in range(len(boxes)):
            xyxy = tuple(float(x) for x in boxes.xyxy[idx].tolist())
            confidence = float(boxes.conf[idx].item())
            class_id = int(boxes.cls[idx].item())
            output.append(
                DetectionRecord(
                    xyxy=xyxy,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=names.get(class_id),
                    metadata={"backend": "ultralytics", "model": self.model_name},
                )
            )
        return output
