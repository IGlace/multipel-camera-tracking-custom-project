"""Detection interfaces and wrappers."""
from .base import DetectionRecord, Detector
from .ultralytics import UltralyticsDetector
__all__ = ["DetectionRecord", "Detector", "UltralyticsDetector"]
