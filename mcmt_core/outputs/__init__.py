"""Output sink utilities package."""
from .base import OutputSink, TrackObservation
from .frames import AnnotatedFrameSink
from .manager import OutputManager
from .mot import MOTSink
from .video import AnnotatedVideoSink
__all__ = [
    "AnnotatedFrameSink",
    "AnnotatedVideoSink",
    "MOTSink",
    "OutputManager",
    "OutputSink",
    "TrackObservation",
]
