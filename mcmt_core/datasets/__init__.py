"""Dataset adapters and loaders."""
from .base import CameraFrame, MultiCameraFrameBatch
from .multi_camera_frames import MultiCameraFrameDataset
from .video_streams import MultiCameraVideoDataset
__all__ = ["CameraFrame", "MultiCameraFrameBatch", "MultiCameraFrameDataset", "MultiCameraVideoDataset"]
