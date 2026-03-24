"""Matching interfaces and adapters."""
from .local_tracker import LocalTracker
from .ultralytics_tracker import UltralyticsLocalTracker
__all__ = ["LocalTracker", "UltralyticsLocalTracker"]
