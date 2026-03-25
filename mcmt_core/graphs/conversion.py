"""Conversion utilities from runtime graph objects to tensor graph batches."""

from __future__ import annotations

import networkx as nx
import torch

from mcmt_core.features import build_edge_feature_tensor, build_node_feature_tensor
from .tensors import GraphTensorBatch


def graph_to_tensor_batch(
    graph: nx.Graph,
    *,
    node_components: list[str],
    edge_feature_order: list[str],
    device: torch.device | None = None,
) -> GraphTensorBatch:
    node_ids = list(graph.nodes())
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    nodes = [graph.nodes[node_id]["node"] for node_id in node_ids]
    node_features = build_node_feature_tensor(nodes, node_components, device=device)

    edge_ids: list[tuple[str, str]] = []
    edge_feature_dicts: list[dict[str, float]] = []
    edge_pairs: list[list[int]] = []
    for source_id, target_id, data in graph.edges(data=True):
        edge_ids.append((source_id, target_id))
        edge_feature_dicts.append(dict(data.get("features", {})))
        edge_pairs.append([node_index[source_id], node_index[target_id]])

    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    if device is not None:
        edge_index = edge_index.to(device)

    edge_features = build_edge_feature_tensor(edge_feature_dicts, edge_feature_order, device=device)
    return GraphTensorBatch(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        node_ids=node_ids,
        edge_ids=edge_ids,
        metadata={"node_components": node_components, "edge_feature_order": edge_feature_order},
    )
