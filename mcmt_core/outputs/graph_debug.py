"""Graph debug output sink."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from mcmt_core.visualization import save_graph_debug_figure
from .base import OutputSink, TrackObservation


class GraphDebugSink(OutputSink):
    def __init__(self, subdir: str = "graph_debug") -> None:
        self.subdir = subdir

    def write(
        self,
        *,
        output_root: Path,
        timestamp: str,
        frame_index: int,
        images_by_camera: dict[str, np.ndarray],
        observations_by_camera: dict[str, list[TrackObservation]],
        payload: dict[str, Any] | None = None,
    ) -> None:
        del images_by_camera, observations_by_camera
        graph = None if payload is None else payload.get("graph")
        if graph is None or not isinstance(graph, nx.Graph):
            return
        path = output_root / self.subdir / f"{frame_index:06d}_{timestamp}.png"
        save_graph_debug_figure(graph, path, title=f"Graph debug | frame={frame_index} | timestamp={timestamp}")
