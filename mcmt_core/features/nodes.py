"""Node feature building primitives for graph construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mcmt_core.detections.base import DetectionRecord


@dataclass(slots=True)
class NodeRecord:
    node_id: str
    camera_id: str
    detection_id: int
    frame_index: int
    timestamp: str
    bbox_xyxy: tuple[float, float, float, float]
    center_norm: tuple[float, float]
    area: float
    confidence: float
    class_id: int
    class_name: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    global_id: int | None = None


class NodeFeatureBuilder:
    def build(
        self,
        detection: DetectionRecord,
        *,
        image_width: int,
        image_height: int,
        camera_id: str,
        frame_index: int,
        timestamp: str,
        detection_id: int,
    ) -> NodeRecord:
        x1, y1, x2, y2 = detection.xyxy
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        center_x = ((x1 + x2) / 2.0) / max(1, image_width)
        center_y = ((y1 + y2) / 2.0) / max(1, image_height)
        area = width * height
        return NodeRecord(
            node_id=f"{timestamp}:{camera_id}:{detection_id}",
            camera_id=camera_id,
            detection_id=detection_id,
            frame_index=frame_index,
            timestamp=timestamp,
            bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
            center_norm=(float(center_x), float(center_y)),
            area=float(area),
            confidence=float(detection.confidence),
            class_id=int(detection.class_id),
            class_name=detection.class_name,
            metadata=detection.metadata or {},
        )
