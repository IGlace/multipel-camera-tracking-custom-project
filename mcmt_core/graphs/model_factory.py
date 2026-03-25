"""Model factory for rest_app reasoning modes."""

from __future__ import annotations

import torch
from torch import nn

from mcmt_core.config.schema import GraphModelConfig
from .direct_scorer import DirectEdgeScorer
from .message_passing import GraphReasoningNetwork
from .tensors import GraphTensorBatch


class HybridReasoningModel(nn.Module):
    def __init__(self, node_input_dim: int, edge_input_dim: int, config: GraphModelConfig) -> None:
        super().__init__()
        self.alpha = float(config.gnn.hybrid_fusion_alpha)
        self.direct = DirectEdgeScorer(edge_input_dim, config.direct_scorer)
        self.gnn = GraphReasoningNetwork(node_input_dim, edge_input_dim, config.gnn)

    def forward(self, batch: GraphTensorBatch) -> dict[str, torch.Tensor]:
        direct_logits = self.direct(batch.edge_features)
        gnn_out = self.gnn(batch)
        fused_logits = (self.alpha * gnn_out["logits"]) + ((1.0 - self.alpha) * direct_logits)
        gnn_out["direct_logits"] = direct_logits
        gnn_out["fused_logits"] = fused_logits
        return gnn_out


def build_reasoning_module(mode: str, node_input_dim: int, edge_input_dim: int, config: GraphModelConfig):
    if mode == "direct_score":
        return DirectEdgeScorer(edge_input_dim, config.direct_scorer)
    if mode == "gnn":
        return GraphReasoningNetwork(node_input_dim, edge_input_dim, config.gnn)
    if mode == "hybrid":
        return HybridReasoningModel(node_input_dim, edge_input_dim, config)
    raise ValueError(f"Unsupported reasoning mode: {mode}")
