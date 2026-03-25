"""Shared helpers for direct-score frame baselines across apps."""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

from mcmt_core.config.schema import RuntimeConfig
from mcmt_core.features import EdgeFeatureBuilder, NodeRecord
from mcmt_core.outputs import (
    AnnotatedFrameSink,
    AnnotatedVideoSink,
    GraphDebugSink,
    LiveGridDisplaySink,
    MOTSink,
    OutputManager,
    TrackObservation,
)


class _UnionFind:
    def __init__(self, node_ids: list[str]) -> None:
        self.parent = {node_id: node_id for node_id in node_ids}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cluster_nodes_graph(graph: nx.Graph, threshold: float) -> list[list[NodeRecord]]:
    pruned = nx.Graph()
    pruned.add_nodes_from(graph.nodes(data=True))
    for source_id, target_id, data in graph.edges(data=True):
        if float(data.get("score", 0.0)) >= threshold:
            pruned.add_edge(source_id, target_id, **data)
    return [
        [pruned.nodes[node_id]["node"] for node_id in component]
        for component in nx.connected_components(pruned)
    ]


def cluster_nodes_pairwise(nodes: list[NodeRecord], edge_builder: EdgeFeatureBuilder, threshold: float) -> list[list[NodeRecord]]:
    uf = _UnionFind([node.node_id for node in nodes])
    for i, source in enumerate(nodes):
        for target in nodes[i + 1 :]:
            if source.camera_id == target.camera_id:
                continue
            edge = edge_builder.build(source, target)
            if edge.score >= threshold:
                uf.union(source.node_id, target.node_id)
    groups: dict[str, list[NodeRecord]] = {}
    for node in nodes:
        groups.setdefault(uf.find(node.node_id), []).append(node)
    return list(groups.values())


def build_output_manager(cfg: RuntimeConfig) -> OutputManager:
    sinks = []
    if cfg.outputs.enable_mot:
        sinks.append(MOTSink(subdir=cfg.outputs.mot_subdir))
    if cfg.outputs.enable_annotated_frames:
        sinks.append(AnnotatedFrameSink(subdir=cfg.outputs.annotated_frames_subdir))
    if cfg.outputs.enable_annotated_video:
        sinks.append(
            AnnotatedVideoSink(
                subdir=cfg.outputs.annotated_video_subdir,
                fps=cfg.outputs.video_fps,
            )
        )
    if cfg.outputs.enable_live_display:
        sinks.append(
            LiveGridDisplaySink(
                window_name=cfg.outputs.live_display_window_name,
                fullscreen=cfg.outputs.live_display_fullscreen,
            )
        )
    if cfg.outputs.enable_graph_debug:
        sinks.append(GraphDebugSink(subdir=cfg.outputs.graph_debug_subdir))
    return OutputManager(cfg.system.output_root, sinks)


def to_observations(cluster_states: Iterable, /) -> dict[str, list[TrackObservation]]:
    observations: dict[str, list[TrackObservation]] = {}
    for state in cluster_states:
        for node in state.nodes:
            observations.setdefault(node.camera_id, []).append(
                TrackObservation(
                    camera_id=node.camera_id,
                    frame_index=node.frame_index,
                    timestamp=node.timestamp,
                    track_id=state.global_id,
                    bbox_xyxy=node.bbox_xyxy,
                    confidence=node.confidence,
                    class_id=node.class_id,
                    class_name=node.class_name,
                )
            )
    return observations
