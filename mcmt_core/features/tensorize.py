"""Helpers for turning node and edge records into model tensors."""

from __future__ import annotations

from typing import Iterable

import torch


def _component_value(node, name: str) -> float:
    if name == "center_x":
        return float(node.center_norm[0])
    if name == "center_y":
        return float(node.center_norm[1])
    if name == "area":
        return float(node.area)
    if name == "confidence":
        return float(node.confidence)
    if name == "class_id":
        return float(node.class_id)
    if name == "length":
        return float(getattr(node, "length", 1.0))
    if name == "duration":
        start_frame = float(getattr(node, "start_frame", getattr(node, "frame_index", 0)))
        end_frame = float(getattr(node, "end_frame", getattr(node, "frame_index", 0)))
        return float(max(0.0, end_frame - start_frame))
    raise ValueError(f"Unsupported node feature component: {name}")


def build_node_feature_tensor(nodes: Iterable, components: list[str], device: torch.device | None = None) -> torch.Tensor:
    rows = []
    for node in nodes:
        rows.append([_component_value(node, component) for component in components])
    tensor = torch.tensor(rows, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def build_edge_feature_tensor(feature_dicts: list[dict[str, float]], feature_order: list[str], device: torch.device | None = None) -> torch.Tensor:
    rows = []
    for feature_dict in feature_dicts:
        rows.append([float(feature_dict.get(name, 0.0)) for name in feature_order])
    tensor = torch.tensor(rows, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor
