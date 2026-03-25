"""Configurable neural building blocks for graph reasoning models."""

from __future__ import annotations

import torch
from torch import nn

from mcmt_core.config.schema import MLPBlockConfig


def build_activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "leaky_relu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "identity": nn.Identity,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]()


def build_norm(name: str, dim: int) -> nn.Module | None:
    if name == "none":
        return None
    if name == "layernorm":
        return nn.LayerNorm(dim)
    if name == "batchnorm":
        return nn.BatchNorm1d(dim)
    raise ValueError(f"Unsupported norm: {name}")


class ConfigurableMLP(nn.Module):
    def __init__(self, input_dim: int, config: MLPBlockConfig) -> None:
        super().__init__()
        dims = [input_dim, *config.hidden_dims, config.output_dim]
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            in_dim = dims[idx]
            out_dim = dims[idx + 1]
            is_last = idx == len(dims) - 2
            layers.append(nn.Linear(in_dim, out_dim))
            use_post = (not is_last) or config.activate_last
            if use_post:
                norm = build_norm(config.norm, out_dim)
                if norm is not None:
                    layers.append(norm)
                layers.append(build_activation(config.activation))
                if config.dropout > 0:
                    layers.append(nn.Dropout(config.dropout))
        self.net = nn.Sequential(*layers)
        self.use_residual = config.residual and input_dim == config.output_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.net(inputs)
        if self.use_residual:
            output = output + inputs
        return output
