"""Direct edge scorer modules."""

from __future__ import annotations

import torch
from torch import nn

from mcmt_core.config.schema import DirectScorerConfig
from .nn_blocks import ConfigurableMLP


class DirectEdgeScorer(nn.Module):
    def __init__(self, edge_input_dim: int, config: DirectScorerConfig) -> None:
        super().__init__()
        self.scorer = ConfigurableMLP(edge_input_dim, config)

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        logits = self.scorer(edge_features)
        return logits.squeeze(-1)
