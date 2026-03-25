"""Tensor graph containers for model-side graph reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class GraphTensorBatch:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    node_ids: list[str] = field(default_factory=list)
    edge_ids: list[tuple[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_features.shape[0])

    @property
    def device(self):
        return self.node_features.device
