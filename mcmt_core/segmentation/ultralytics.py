"""Ultralytics-backed segmentor wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import SegmentationRecord, Segmentor


class UltralyticsSegmentor(Segmentor):
    def __init__(
        self,
        model: str,
        task: str = "sam",
        confidence: float = 0.25,
        iou: float = 0.45,
        **kwargs: Any,
    ) -> None:
        self.model_name = model
        self.task = task
        self.confidence = confidence
        self.iou = iou
        self.extra_kwargs = kwargs
        self._model = None

    def _lazy_load(self):
        if self._model is None:
            if self.task == "sam":
                from ultralytics import SAM

                self._model = SAM(self.model_name)
            else:
                from ultralytics import YOLO

                self._model = YOLO(self.model_name)
        return self._model

    def predict(self, image: np.ndarray) -> list[SegmentationRecord]:
        model = self._lazy_load()
        results = model.predict(image, conf=self.confidence, iou=self.iou, verbose=False, **self.extra_kwargs)
        if not results:
            return []
        result = results[0]
        masks = getattr(result, "masks", None)
        boxes = getattr(result, "boxes", None)
        if masks is None:
            return []
        output: list[SegmentationRecord] = []
        for idx in range(len(masks)):
            xyxy = None
            confidence = None
            class_id = None
            if boxes is not None and len(boxes) > idx:
                xyxy = tuple(float(x) for x in boxes.xyxy[idx].tolist())
                confidence = float(boxes.conf[idx].item())
                class_id = int(boxes.cls[idx].item())
            output.append(
                SegmentationRecord(
                    xyxy=xyxy,
                    confidence=confidence,
                    class_id=class_id,
                    mask=masks[idx],
                    metadata={"backend": "ultralytics", "model": self.model_name, "task": self.task},
                )
            )
        return output
