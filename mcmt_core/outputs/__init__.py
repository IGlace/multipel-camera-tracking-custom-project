"""Output sink utilities package."""
from .base import OutputSink, TrackObservation
from .frames import AnnotatedFrameSink
from .manager import OutputManager
from .mot import MOTSink
__all__ = ["AnnotatedFrameSink", "MOTSink", "OutputManager", "OutputSink", "TrackObservation"]
