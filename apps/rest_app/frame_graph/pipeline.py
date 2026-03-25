"""First usable frame-graph baseline for rest_app.

This phase implements a direct-score graph pipeline over synchronized multi-camera
frame folders. It is intentionally simple and serves as the first runnable baseline:
- run Ultralytics detection per camera per timestamp
- build graph nodes from detections
- build cross-camera edges and direct scores
- cluster detections through score-thresholded connected components
- propagate global IDs across consecutive frames with a simple same-camera IoU matcher
- export MOT files, annotated frames, and optional annotated videos
"""

from __future__ import annotations

from mcmt_core.config.schema import RuntimeConfig
from mcmt_core.datasets import MultiCameraFrameDataset
from mcmt_core.detections import UltralyticsDetector
from mcmt_core.features import EdgeFeatureBuilder, NodeFeatureBuilder, NodeRecord
from mcmt_core.graphs import build_spatial_frame_graph
from mcmt_core.matching import ClusterState, match_clusters
from mcmt_core.runtime import build_output_manager, cluster_nodes_graph, to_observations


def run_frame_graph_baseline(cfg: RuntimeConfig, logger) -> None:
    if cfg.dataset.input_type != "frame_folders":
        raise NotImplementedError("The current frame_graph_baseline only supports dataset.input_type='frame_folders'.")

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
    output_manager = build_output_manager(cfg)

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
            current_clusters = cluster_nodes_graph(graph, cfg.graph_model.edge_score_threshold)
            current_states, next_global_id = match_clusters(
                previous_clusters,
                current_clusters,
                cfg.graph_model.temporal_match_threshold,
                next_global_id,
            )
            observations_by_camera = to_observations(current_states)
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
