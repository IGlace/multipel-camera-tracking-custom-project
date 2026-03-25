"""Output sink utilities package."""
from .base import OutputSink, TrackObservation
from .frames import AnnotatedFrameSink
from .graph_debug import GraphDebugSink
from .live_display import LiveGridDisplaySink
from .manager import OutputManager
from .mot import MOTSink
from .video import AnnotatedVideoSink
__all__ = [
    "AnnotatedFrameSink",
    "AnnotatedVideoSink",
    "GraphDebugSink",
    "LiveGridDisplaySink",
    "MOTSink",
    "OutputManager",
    "OutputSink",
    "TrackObservation",
]
