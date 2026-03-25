"""Tracklet dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mcmt_core.matching import LocalTrackObservation


@dataclass(slots=True)
class TrackletRecord:
    node_id: str
    camera_id: str
    local_track_id: int
    frame_index: int
    timestamp: str
    bbox_xyxy: tuple[float, float, float, float]
    center_norm: tuple[float, float]
    area: float
    confidence: float
    class_id: int
    class_name: str | None
    start_frame: int
    end_frame: int
    length: int
    observations: list[LocalTrackObservation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    global_id: int | None = None
