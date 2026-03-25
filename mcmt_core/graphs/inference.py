"""Inference helpers for applying configurable reasoning modules to runtime graphs."""

from __future__ import annotations

import networkx as nx
import torch

from mcmt_core.config.schema import GraphModelConfig
from .conversion import graph_to_tensor_batch
from .model_factory import build_reasoning_module


def infer_graph_edge_probabilities(
    graph: nx.Graph,
    *,
    config: GraphModelConfig,
    device: torch.device | None = None,
) -> tuple[nx.Graph, dict[str, torch.Tensor]]:
    scored_graph = graph.copy()
    tensor_batch = graph_to_tensor_batch(
        scored_graph,
        node_components=config.node_feature_components,
        edge_feature_order=config.spatial_edge_features,
        device=device,
    )

    if tensor_batch.num_edges == 0:
        empty = torch.zeros((0,), dtype=torch.float32, device=device or torch.device("cpu"))
        return scored_graph, {"probabilities": empty, "logits": empty}

    node_input_dim = tensor_batch.node_features.shape[-1]
    edge_input_dim = tensor_batch.edge_features.shape[-1]
    model = build_reasoning_module(
        config.reasoning_mode,
        node_input_dim=node_input_dim,
        edge_input_dim=edge_input_dim,
        config=config,
    )
    if device is not None:
        model = model.to(device)
    model.eval()

    with torch.no_grad():
        if config.reasoning_mode == "direct_score":
            logits = model(tensor_batch.edge_features)
            outputs = {"logits": logits}
        else:
            outputs = model(tensor_batch)
            logits = outputs["fused_logits"] if config.reasoning_mode == "hybrid" else outputs["logits"]
        probabilities = torch.sigmoid(logits)

    for idx, (source_id, target_id) in enumerate(tensor_batch.edge_ids):
        edge_data = scored_graph[source_id][target_id]
        edge_data["model_logit"] = float(logits[idx].detach().cpu().item())
        edge_data["score"] = float(probabilities[idx].detach().cpu().item())
        edge_data["score_source"] = f"model:{config.reasoning_mode}"

    outputs["probabilities"] = probabilities
    return scored_graph, outputs
