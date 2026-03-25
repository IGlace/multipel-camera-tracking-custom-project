"""Sanity helpers for configurable reasoning model construction."""

from __future__ import annotations

import torch

from mcmt_core.graphs import GraphTensorBatch, build_reasoning_module


def run_gnn_sanity(cfg, logger) -> None:
    node_input_dim = 5
    edge_input_dim = len(cfg.graph_model.spatial_edge_features)
    model = build_reasoning_module(
        cfg.graph_model.reasoning_mode,
        node_input_dim=node_input_dim,
        edge_input_dim=edge_input_dim,
        config=cfg.graph_model,
    )

    logger.info("Built reasoning module for mode=%s", cfg.graph_model.reasoning_mode)
    logger.info("Model summary:\n%s", model)

    batch = GraphTensorBatch(
        node_features=torch.randn(4, node_input_dim),
        edge_index=torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=torch.long),
        edge_features=torch.randn(4, edge_input_dim),
        node_ids=["n0", "n1", "n2", "n3"],
        edge_ids=[("n0", "n1"), ("n0", "n2"), ("n1", "n3"), ("n2", "n3")],
    )

    outputs = model(batch) if cfg.graph_model.reasoning_mode != "direct_score" else {"logits": model(batch.edge_features)}
    for key, value in outputs.items():
        logger.info("Sanity output %s shape=%s", key, tuple(value.shape))
