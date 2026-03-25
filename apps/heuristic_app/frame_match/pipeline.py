"""First usable frame-match baseline for heuristic_app.

This phase implements a non-graph direct matching pipeline over synchronized multi-camera
frame folders. It intentionally avoids graph libraries and groups detections using
pairwise score thresholding with a union-find structure.
"""

from __future__ import annotations

from collections.abc import Iterable

from mcmt_core.config.schema import RuntimeConfig
from mcmt_core.datasets import MultiCameraFrameDataset
from mcmt_core.detections import UltralyticsDetector
from mcmt_core.features import EdgeFeatureBuilder, NodeFeatureBuilder, NodeRecord
from mcmt_core.matching import ClusterState, match_clusters
from mcmt_core.outputs import AnnotatedFrameSink, AnnotatedVideoSink, MOTSink, OutputManager, TrackObservation


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


def _cluster_nodes_without_graph(nodes: list[NodeRecord], edge_builder: EdgeFeatureBuilder, threshold: float) -> list[list[NodeRecord]]:
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


def _build_output_manager(cfg: RuntimeConfig) -> OutputManager:
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


def run_frame_match_baseline(cfg: RuntimeConfig, logger) -> None:
    if cfg.dataset.input_type != "frame_folders":
        raise NotImplementedError("The current frame_match_baseline only supports dataset.input_type='frame_folders'.")

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

    logger.info("Running frame_match_baseline over %d synchronized timestamps", len(dataset))

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

            current_clusters = _cluster_nodes_without_graph(nodes, edge_builder, cfg.graph_model.edge_score_threshold)
            current_states, next_global_id = match_clusters(
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
                "frame=%s nodes=%d clusters=%d next_global_id=%d",
                batch.timestamp,
                len(nodes),
                len(current_states),
                next_global_id,
            )
    finally:
        output_manager.close()
