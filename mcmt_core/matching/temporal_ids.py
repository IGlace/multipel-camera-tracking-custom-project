"""Shared helpers for carrying IDs across consecutive timestamps."""

from __future__ import annotations

from dataclasses import dataclass

from mcmt_core.features.nodes import NodeRecord


@dataclass(slots=True)
class ClusterState:
    global_id: int
    nodes: list[NodeRecord]


def same_camera_iou(a: NodeRecord, b: NodeRecord) -> float:
    ax1, ay1, ax2, ay2 = a.bbox_xyxy
    bx1, by1, bx2, by2 = b.bbox_xyxy
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def match_clusters(previous: list[ClusterState], current_clusters: list[list[NodeRecord]], threshold: float, next_id: int):
    assigned: list[ClusterState] = []
    used_previous: set[int] = set()
    for cluster_nodes in current_clusters:
        best_idx = None
        best_score = -1.0
        for idx, prev in enumerate(previous):
            if idx in used_previous:
                continue
            shared_scores: list[float] = []
            for current in cluster_nodes:
                for old in prev.nodes:
                    if current.camera_id == old.camera_id:
                        shared_scores.append(same_camera_iou(current, old))
            score = sum(shared_scores) / len(shared_scores) if shared_scores else 0.0
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None and best_score >= threshold:
            state = ClusterState(global_id=previous[best_idx].global_id, nodes=cluster_nodes)
            used_previous.add(best_idx)
        else:
            state = ClusterState(global_id=next_id, nodes=cluster_nodes)
            next_id += 1
        assigned.append(state)
    return assigned, next_id
