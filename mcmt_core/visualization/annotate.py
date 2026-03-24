"""Simple frame annotation helpers."""

from __future__ import annotations

import cv2
import numpy as np

from mcmt_core.outputs.base import TrackObservation


def annotate_frame(image: np.ndarray, observations: list[TrackObservation]) -> np.ndarray:
    canvas = image.copy()
    for obs in observations:
        x1, y1, x2, y2 = (int(v) for v in obs.bbox_xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID {obs.track_id} | {obs.confidence:.2f}"
        cv2.putText(canvas, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return canvas
