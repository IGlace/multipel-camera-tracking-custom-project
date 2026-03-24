"""First usable frame-graph baseline for rest_app.

This phase implements a direct-score graph pipeline over synchronized multi-camera
frame folders. It is intentionally simple and serves as the first runnable baseline:
- run Ultralytics detection per camera per timestamp
- build graph nodes from detections
- build cross-camera edges and direct scores
- cluster detections through score-thresholded connected components
- propagate global IDs across consecutive frames with a simple same-camera IoU matcher
- export MOT files and annotated frames
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx

from mcmt_core.config.schema import RuntimeConfig
from mcmt_core.datasets import MultiCameraFrameDataset
from mcmt_core.detections import UltralyticsDetector
from mcmt_core.features import EdgeFeatureBuilder, NodeFeatureBuilder, NodeRecord
from mcmt_core.graphs import build_spatial_frame_graph
from mcmt_core.outputs import AnnotatedFrameSink, MOTSink, OutputManager, TrackObservation


@dataclass(slots=True)
class ClusterState:
    global_id: int
    nodes: list[NodeRecord]


def _same_camera_iou(a: NodeRecord, b: NodeRecord) -> float:
    ax1, ay1, ax2, ay2 = a.bbox_xyxy
    bx1, by1, bx2, by2 = b.bbox_xyxy
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def _cluster_nodes(graph: nx.Graph, threshold: float) -> list[list[NodeRecord]]:
    pruned = nx.Graph()
    pruned.add_nodes_from(graph.nodes(data=True))
    for source_id, target_id, data in graph.edges(data=True):
        if float(data.get("score", 0.0)) >= threshold:
            pruned.add_edge(source_id, target_id, **data)
    return [
        [pruned.nodes[node_id]["node"] for node_id in component]
        for component in nx.connected_components(pruned)
    ]


def _match_clusters(previous: list[ClusterState], current_clusters: list[list[NodeRecord]], threshold: float, next_id: int):
    assigned: list[ClusterState] = []
    used_previous: set[int] = set()
    for cluster_nodes in current_clusters:
        best_idx = None
        best_score = -1.0
        for idx, prev in enumerate(previous):
            if idx in used_previous:
                continue
            shared_scores: list[float] = []
            for current in cluster_nodes:
                for old in prev.nodes:
                    if current.camera_id == old.camera_id:
                        shared_scores.append(_same_camera_iou(current, old))
            score = sum(shared_scores) / len(shared_scores) if shared_scores else 0.0
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None and best_score >= threshold:
            state = ClusterState(global_id=previous[best_idx].global_id, nodes=cluster_nodes)
            used_previous.add(best_idx)
        else:
            state = ClusterState(global_id=next_id, nodes=cluster_nodes)
            next_id += 1
        assigned.append(state)
    return assigned, next_id


def _build_output_manager(cfg: RuntimeConfig) -> OutputManager:
    sinks = []
    if cfg.outputs.enable_mot:
        sinks.append(MOTSink(subdir=cfg.outputs.mot_subdir))
    if cfg.outputs.enable_annotated_frames:
        sinks.append(AnnotatedFrameSink(subdir=cfg.outputs.annotated_frames_subdir))
    return OutputManager(cfg.system.output_root, sinks)


def _to_observations(cluster_states: Iterable[ClusterState]) -> dict[str, list[TrackObservation]]:
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


def run_frame_graph_baseline(cfg: RuntimeConfig, logger) -> None:
    dataset = MultiCameraFrameDataset(cfg.dataset)
    detector = UltralyticsDetector(
        model=cfg.detector.model,
        confidence=cfg.detector.confidence,
        iou=cfg.detector.iou,
    )
    node_builder = NodeFeatureBuilder()
    edge_builder = EdgeFeatureBuilder(
        selected_features=cfg.graph_model.spatial_edge_features,
        score_weights=cfg.graph_model.score_weights,
    )
    output_manager = _build_output_manager(cfg)

    previous_clusters: list[ClusterState] = []
    next_global_id = 1

    logger.info("Running frame_graph_baseline over %d synchronized timestamps", len(dataset))

    try:
        for batch in dataset:
            images_by_camera = {frame.camera_id: frame.image for frame in batch.frames}
            nodes: list[NodeRecord] = []
            for frame in batch.frames:
                height, width = frame.image.shape[:2]
                detections = detector.predict(frame.image)
                for detection_id, detection in enumerate(detections):
                    nodes.append(
                        node_builder.build(
                            detection,
                            image_width=width,
                            image_height=height,
                            camera_id=frame.camera_id,
                            frame_index=batch.frame_index,
                            timestamp=batch.timestamp,
                            detection_id=detection_id,
                        )
                    )

            graph = build_spatial_frame_graph(nodes, edge_builder)
            current_clusters = _cluster_nodes(graph, cfg.graph_model.edge_score_threshold)
            current_states, next_global_id = _match_clusters(
                previous_clusters,
                current_clusters,
                cfg.graph_model.temporal_match_threshold,
                next_global_id,
            )
            observations_by_camera = _to_observations(current_states)
            output_manager.write(
                timestamp=batch.timestamp,
                frame_index=batch.frame_index,
                images_by_camera=images_by_camera,
                observations_by_camera=observations_by_camera,
            )
            previous_clusters = current_states
            logger.info(
                "frame=%s nodes=%d edges=%d clusters=%d next_global_id=%d",
                batch.timestamp,
                len(nodes),
                graph.number_of_edges(),
                len(current_states),
                next_global_id,
            )
    finally:
        output_manager.close()
