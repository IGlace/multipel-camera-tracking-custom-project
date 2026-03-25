"""Probe runners that execute reasoning modules on real detected graphs."""

from __future__ import annotations

from mcmt_core.datasets import MultiCameraFrameDataset
from mcmt_core.detections import UltralyticsDetector
from mcmt_core.features import EdgeFeatureBuilder, NodeFeatureBuilder
from mcmt_core.graphs import build_reasoning_module, build_spatial_frame_graph, graph_to_tensor_batch


def run_graph_tensor_probe(cfg, logger) -> None:
    if cfg.dataset.input_type != "frame_folders":
        raise NotImplementedError("graph_tensor_probe currently supports only frame_folders datasets.")

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

    node_input_dim = len(cfg.graph_model.node_feature_components)
    edge_input_dim = len(cfg.graph_model.spatial_edge_features)
    model = build_reasoning_module(
        cfg.graph_model.reasoning_mode,
        node_input_dim=node_input_dim,
        edge_input_dim=edge_input_dim,
        config=cfg.graph_model,
    )
    logger.info("Built graph reasoning module for probe mode=%s", cfg.graph_model.reasoning_mode)

    processed = 0
    for batch in dataset:
        nodes = []
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
        tensor_batch = graph_to_tensor_batch(
            graph,
            node_components=cfg.graph_model.node_feature_components,
            edge_feature_order=cfg.graph_model.spatial_edge_features,
        )
        logger.info(
            "probe frame=%s num_nodes=%d num_edges=%d node_dim=%d edge_dim=%d",
            batch.timestamp,
            tensor_batch.num_nodes,
            tensor_batch.num_edges,
            tensor_batch.node_features.shape[-1] if tensor_batch.num_nodes > 0 else 0,
            tensor_batch.edge_features.shape[-1] if tensor_batch.num_edges > 0 else 0,
        )

        if tensor_batch.num_edges > 0:
            outputs = model(tensor_batch) if cfg.graph_model.reasoning_mode != "direct_score" else {"logits": model(tensor_batch.edge_features)}
            for key, value in outputs.items():
                logger.info("probe output %s shape=%s", key, tuple(value.shape))
            logger.info("probe logits preview=%s", outputs["logits"][: min(5, outputs["logits"].shape[0])].detach().cpu().tolist())
        else:
            logger.warning("probe frame=%s has no cross-camera edges to score", batch.timestamp)

        processed += 1
        if processed >= 2:
            break
