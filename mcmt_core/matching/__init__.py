"""Matching interfaces and adapters."""
from .local_tracker import LocalTracker
from .temporal_ids import ClusterState, match_clusters, same_camera_iou
from .ultralytics_tracker import UltralyticsLocalTracker
__all__ = [
    "ClusterState",
    "LocalTracker",
    "UltralyticsLocalTracker",
    "match_clusters",
    "same_camera_iou",
]
