"""Experimental neural edge inference pipeline for rest_app.

This pipeline keeps the existing runtime graph construction path, but replaces the
hand-crafted edge score used for clustering with probabilities predicted by the
configured reasoning module (`direct_score`, `gnn`, or `hybrid`).
"""

from __future__ import annotations

from mcmt_core.config.schema import RuntimeConfig
from mcmt_core.datasets import MultiCameraFrameDataset
from mcmt_core.detections import UltralyticsDetector
from mcmt_core.features import EdgeFeatureBuilder, NodeFeatureBuilder, NodeRecord
from mcmt_core.graphs import build_spatial_frame_graph, infer_graph_edge_probabilities
from mcmt_core.matching import ClusterState, match_clusters
from mcmt_core.runtime import build_output_manager, cluster_nodes_graph, to_observations


def run_frame_graph_neural_inference(cfg: RuntimeConfig, logger) -> None:
    if cfg.dataset.input_type != "frame_folders":
        raise NotImplementedError("neural_edge_inference currently supports only frame_folders datasets.")

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

    logger.info("Running neural_edge_inference over %d synchronized timestamps", len(dataset))
    logger.info("Neural reasoning mode=%s", cfg.graph_model.reasoning_mode)

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
            scored_graph, model_outputs = infer_graph_edge_probabilities(graph, config=cfg.graph_model)
            current_clusters = cluster_nodes_graph(scored_graph, cfg.graph_model.edge_score_threshold)
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
                payload={"graph": scored_graph},
            )
            previous_clusters = current_states
            prob_preview = model_outputs["probabilities"][: min(5, model_outputs["probabilities"].shape[0])].detach().cpu().tolist()
            logger.info(
                "frame=%s nodes=%d edges=%d clusters=%d next_global_id=%d prob_preview=%s",
                batch.timestamp,
                len(nodes),
                scored_graph.number_of_edges(),
                len(current_states),
                next_global_id,
                prob_preview,
            )
    finally:
        output_manager.close()
