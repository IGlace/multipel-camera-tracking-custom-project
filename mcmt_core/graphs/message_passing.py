"""Configurable message-passing scaffolding for graph reasoning."""

from __future__ import annotations

import torch
from torch import nn

from mcmt_core.config.schema import GraphNeuralConfig
from .nn_blocks import ConfigurableMLP
from .tensors import GraphTensorBatch


def _aggregate_messages(dst_index: torch.Tensor, messages: torch.Tensor, num_nodes: int, mode: str) -> torch.Tensor:
    if messages.numel() == 0:
        return torch.zeros((num_nodes, messages.shape[-1]), device=messages.device, dtype=messages.dtype)

    if mode == "sum":
        aggregated = torch.zeros((num_nodes, messages.shape[-1]), device=messages.device, dtype=messages.dtype)
        aggregated.index_add_(0, dst_index, messages)
        return aggregated

    if mode == "mean":
        aggregated = torch.zeros((num_nodes, messages.shape[-1]), device=messages.device, dtype=messages.dtype)
        counts = torch.zeros((num_nodes, 1), device=messages.device, dtype=messages.dtype)
        aggregated.index_add_(0, dst_index, messages)
        counts.index_add_(0, dst_index, torch.ones((messages.shape[0], 1), device=messages.device, dtype=messages.dtype))
        return aggregated / counts.clamp_min(1.0)

    if mode == "max":
        aggregated = torch.full(
            (num_nodes, messages.shape[-1]),
            fill_value=-torch.inf,
            device=messages.device,
            dtype=messages.dtype,
        )
        for edge_idx in range(messages.shape[0]):
            node_idx = int(dst_index[edge_idx].item())
            aggregated[node_idx] = torch.maximum(aggregated[node_idx], messages[edge_idx])
        aggregated[torch.isinf(aggregated)] = 0.0
        return aggregated

    raise ValueError(f"Unsupported aggregation mode: {mode}")


class GraphReasoningNetwork(nn.Module):
    def __init__(self, node_input_dim: int, edge_input_dim: int, config: GraphNeuralConfig) -> None:
        super().__init__()
        self.config = config
        self.node_encoder = ConfigurableMLP(node_input_dim, config.node_encoder)
        self.edge_encoder = ConfigurableMLP(edge_input_dim, config.edge_encoder)

        message_input_dim = (2 * config.node_encoder.output_dim) + config.edge_encoder.output_dim
        self.message_encoder = ConfigurableMLP(message_input_dim, config.message_encoder)

        node_update_input_dim = config.node_encoder.output_dim + config.message_encoder.output_dim
        self.node_updater = ConfigurableMLP(node_update_input_dim, config.node_updater)

        edge_update_input_dim = (
            (2 * config.node_updater.output_dim)
            + config.edge_encoder.output_dim
            + config.message_encoder.output_dim
        )
        self.edge_updater = ConfigurableMLP(edge_update_input_dim, config.edge_updater)
        self.predictor = ConfigurableMLP(config.edge_updater.output_dim, config.predictor)

    def forward(self, batch: GraphTensorBatch) -> dict[str, torch.Tensor]:
        node_state = self.node_encoder(batch.node_features)
        edge_state = self.edge_encoder(batch.edge_features)

        src_index = batch.edge_index[0].long()
        dst_index = batch.edge_index[1].long()

        src_state = node_state[src_index]
        dst_state = node_state[dst_index]
        message_input = torch.cat([src_state, dst_state, edge_state], dim=-1)
        messages = self.message_encoder(message_input)
        aggregated = _aggregate_messages(dst_index, messages, batch.num_nodes, self.config.aggregation)

        updated_node_state = self.node_updater(torch.cat([node_state, aggregated], dim=-1))
        updated_src = updated_node_state[src_index]
        updated_dst = updated_node_state[dst_index]
        edge_update_input = torch.cat([updated_src, updated_dst, edge_state, messages], dim=-1)
        updated_edge_state = self.edge_updater(edge_update_input)
        logits = self.predictor(updated_edge_state).squeeze(-1)

        return {
            "logits": logits,
            "node_embeddings": updated_node_state,
            "edge_embeddings": updated_edge_state,
            "messages": messages,
        }
