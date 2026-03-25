"""Rolling tracklet summary builder for realtime local tracks."""

from __future__ import annotations

from collections import defaultdict

from mcmt_core.matching import LocalTrackObservation
from .base import TrackletRecord


class SlidingWindowTrackletBuilder:
    def __init__(self, window_size: int = 15, min_length: int = 2, idle_tolerance: int = 1) -> None:
        self.window_size = window_size
        self.min_length = min_length
        self.idle_tolerance = idle_tolerance
        self._buffers: dict[tuple[str, int], list[LocalTrackObservation]] = defaultdict(list)
        self._last_seen: dict[tuple[str, int], int] = {}

    def update(self, frame_index: int, observations: list[LocalTrackObservation]) -> list[TrackletRecord]:
        seen_keys: set[tuple[str, int]] = set()
        for obs in observations:
            key = (obs.camera_id, obs.local_track_id)
            buffer = self._buffers[key]
            buffer.append(obs)
            if len(buffer) > self.window_size:
                del buffer[0]
            self._last_seen[key] = frame_index
            seen_keys.add(key)

        stale_keys = [
            key for key, last_seen in self._last_seen.items() if frame_index - last_seen > self.idle_tolerance
        ]
        for key in stale_keys:
            self._last_seen.pop(key, None)
            self._buffers.pop(key, None)

        summaries: list[TrackletRecord] = []
        for key, buffer in self._buffers.items():
            if len(buffer) < self.min_length:
                continue
            latest = buffer[-1]
            mean_conf = sum(obs.confidence for obs in buffer) / len(buffer)
            mean_area = sum(obs.area for obs in buffer) / len(buffer)
            summaries.append(
                TrackletRecord(
                    node_id=f"tracklet:{latest.camera_id}:{latest.local_track_id}:{latest.timestamp}",
                    camera_id=latest.camera_id,
                    local_track_id=latest.local_track_id,
                    frame_index=latest.frame_index,
                    timestamp=latest.timestamp,
                    bbox_xyxy=latest.bbox_xyxy,
                    center_norm=latest.center_norm,
                    area=float(mean_area),
                    confidence=float(mean_conf),
                    class_id=latest.class_id,
                    class_name=latest.class_name,
                    start_frame=buffer[0].frame_index,
                    end_frame=buffer[-1].frame_index,
                    length=len(buffer),
                    observations=list(buffer),
                    metadata={"window_size": self.window_size},
                )
            )
        return summaries
