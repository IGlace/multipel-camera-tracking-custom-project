"""Segmentation interfaces and wrappers."""
from .base import SegmentationRecord, Segmentor
from .ultralytics import UltralyticsSegmentor
__all__ = ["SegmentationRecord", "Segmentor", "UltralyticsSegmentor"]
