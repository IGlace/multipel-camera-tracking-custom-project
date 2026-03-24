"""Typed configuration models for the Phase 1 scaffold."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class SystemConfig(BaseModel):
    project_name: str = "mcmt-platform"
    output_root: Path = Path("outputs")
    require_gpu: bool = True


class LoggingConfig(BaseModel):
    verbosity: Literal["quiet", "normal", "detailed", "trace"] = "normal"
    log_to_file: bool = True
    log_filename: str = "run.log"


class DatasetConfig(BaseModel):
    input_type: Literal["frame_folders", "video_file", "video_folder"] = "frame_folders"
    root_dir: Path = Path("data")
    camera_dirs: list[str] = Field(default_factory=list)
    image_extensions: list[str] = Field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".webp"])
    sync_strategy: Literal["min_length", "strict"] = "min_length"


class DetectorConfig(BaseModel):
    backend: Literal["ultralytics"] = "ultralytics"
    model: str = "yolo11n.pt"
    confidence: float = 0.25
    iou: float = 0.45


class SegmentorConfig(BaseModel):
    backend: Literal["ultralytics"] = "ultralytics"
    task: Literal["sam", "segment"] = "sam"
    model: str = "sam_b.pt"
    confidence: float = 0.25
    iou: float = 0.45


class LocalTrackerConfig(BaseModel):
    backend: Literal["ultralytics"] = "ultralytics"
    tracker: Literal["botsort", "bytetrack"] = "botsort"
    tracker_config: str | None = None


class ReIDConfig(BaseModel):
    backend: Literal["mgn", "callable"] = "mgn"
    weights_path: Path | None = None
    normalize_embeddings: bool = True


class AttributesConfig(BaseModel):
    backend: Literal["person_attribute_recognition", "callable"] = "person_attribute_recognition"
    weights_path: Path | None = None
    attribute_names_path: Path | None = None


class OutputsConfig(BaseModel):
    enable_mot: bool = True
    enable_annotated_frames: bool = True
    mot_subdir: str = "mot"
    annotated_frames_subdir: str = "annotated_frames"


class GraphModelConfig(BaseModel):
    reasoning_mode: Literal["direct_score", "gnn", "hybrid"] = "direct_score"
    spatial_edge_features: list[str] = Field(
        default_factory=lambda: [
            "center_similarity",
            "iou",
            "confidence_similarity",
            "class_match",
            "area_ratio_similarity",
        ]
    )
    score_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "center_similarity": 0.35,
            "iou": 0.35,
            "confidence_similarity": 0.10,
            "class_match": 0.10,
            "area_ratio_similarity": 0.10,
        }
    )
    edge_score_threshold: float = 0.55
    temporal_match_threshold: float = 0.30


class RuntimeConfig(BaseModel):
    app_name: str = Field(default="unknown")
    system: SystemConfig = Field(default_factory=SystemConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    segmentor: SegmentorConfig = Field(default_factory=SegmentorConfig)
    local_tracker: LocalTrackerConfig = Field(default_factory=LocalTrackerConfig)
    reid: ReIDConfig = Field(default_factory=ReIDConfig)
    attributes: AttributesConfig = Field(default_factory=AttributesConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)
    graph_model: GraphModelConfig = Field(default_factory=GraphModelConfig)
    config_path: Path | None = None
