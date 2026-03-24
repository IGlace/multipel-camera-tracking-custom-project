"""Dataset adapters and loaders."""
from .base import CameraFrame, MultiCameraFrameBatch
from .multi_camera_frames import MultiCameraFrameDataset
__all__ = ["CameraFrame", "MultiCameraFrameBatch", "MultiCameraFrameDataset"]
