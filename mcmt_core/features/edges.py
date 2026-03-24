"""Edge feature and direct-score builders for the first frame-graph baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .nodes import NodeRecord


@dataclass(slots=True)
class EdgeRecord:
    source_id: str
    target_id: str
    features: dict[str, float]
    score: float


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
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


class EdgeFeatureBuilder:
    def __init__(self, selected_features: list[str], score_weights: dict[str, float]) -> None:
        self.selected_features = selected_features
        self.score_weights = score_weights

    def build(self, source: NodeRecord, target: NodeRecord) -> EdgeRecord:
        dx = source.center_norm[0] - target.center_norm[0]
        dy = source.center_norm[1] - target.center_norm[1]
        center_dist = math.sqrt(dx * dx + dy * dy)
        center_similarity = max(0.0, 1.0 - (center_dist / math.sqrt(2.0)))
        confidence_similarity = max(0.0, 1.0 - abs(source.confidence - target.confidence))
        class_match = 1.0 if source.class_id == target.class_id else 0.0
        area_ratio_similarity = min(source.area, target.area) / max(source.area, target.area)
        iou = _bbox_iou(source.bbox_xyxy, target.bbox_xyxy)

        feature_bank = {
            "center_similarity": center_similarity,
            "confidence_similarity": confidence_similarity,
            "class_match": class_match,
            "area_ratio_similarity": area_ratio_similarity,
            "iou": iou,
        }
        features = {name: float(feature_bank[name]) for name in self.selected_features if name in feature_bank}
        if not features:
            score = 0.0
        else:
            numerator = 0.0
            denominator = 0.0
            for name, value in features.items():
                weight = float(self.score_weights.get(name, 1.0))
                numerator += weight * value
                denominator += weight
            score = 0.0 if denominator <= 0 else numerator / denominator
        return EdgeRecord(source_id=source.node_id, target_id=target.node_id, features=features, score=score)
