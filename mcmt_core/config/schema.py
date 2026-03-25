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
    video_extensions: list[str] = Field(default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv", ".webm"])
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


class TrackletConfig(BaseModel):
    window_size: int = 15
    min_length: int = 2
    idle_tolerance: int = 1


class OutputsConfig(BaseModel):
    enable_mot: bool = True
    enable_annotated_frames: bool = True
    enable_annotated_video: bool = False
    enable_live_display: bool = False
    enable_graph_debug: bool = False
    mot_subdir: str = "mot"
    annotated_frames_subdir: str = "annotated_frames"
    annotated_video_subdir: str = "annotated_videos"
    graph_debug_subdir: str = "graph_debug"
    live_display_window_name: str = "MCMT Live Grid"
    live_display_fullscreen: bool = True
    video_fps: int = 20


class MLPBlockConfig(BaseModel):
    hidden_dims: list[int] = Field(default_factory=list)
    output_dim: int = 64
    activation: Literal["relu", "gelu", "silu", "leaky_relu", "tanh", "identity"] = "relu"
    norm: Literal["none", "layernorm", "batchnorm"] = "none"
    dropout: float = 0.0
    residual: bool = False
    activate_last: bool = False


class DirectScorerConfig(MLPBlockConfig):
    output_dim: int = 1
    hidden_dims: list[int] = Field(default_factory=lambda: [64, 32])


class GraphNeuralConfig(BaseModel):
    aggregation: Literal["sum", "mean", "max"] = "sum"
    hybrid_fusion_alpha: float = 0.5
    node_encoder: MLPBlockConfig = Field(default_factory=lambda: MLPBlockConfig(output_dim=64, hidden_dims=[128]))
    edge_encoder: MLPBlockConfig = Field(default_factory=lambda: MLPBlockConfig(output_dim=32, hidden_dims=[64]))
    message_encoder: MLPBlockConfig = Field(default_factory=lambda: MLPBlockConfig(output_dim=64, hidden_dims=[128]))
    node_updater: MLPBlockConfig = Field(default_factory=lambda: MLPBlockConfig(output_dim=64, hidden_dims=[128]))
    edge_updater: MLPBlockConfig = Field(default_factory=lambda: MLPBlockConfig(output_dim=32, hidden_dims=[64]))
    predictor: MLPBlockConfig = Field(default_factory=lambda: MLPBlockConfig(output_dim=1, hidden_dims=[32]))


class GraphModelConfig(BaseModel):
    reasoning_mode: Literal["direct_score", "gnn", "hybrid"] = "direct_score"
    node_feature_components: list[str] = Field(
        default_factory=lambda: ["center_x", "center_y", "area", "confidence", "class_id"]
    )
    spatial_edge_features: list[str] = Field(
        default_factory=lambda: [
            "center_similarity",
            "iou",
            "confidence_similarity",
            "class_match",
            "area_ratio_similarity",
        ]
    )
    temporal_edge_features: list[str] = Field(
        default_factory=lambda: [
            "center_similarity",
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
    direct_scorer: DirectScorerConfig = Field(default_factory=DirectScorerConfig)
    gnn: GraphNeuralConfig = Field(default_factory=GraphNeuralConfig)


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
    tracklet: TrackletConfig = Field(default_factory=TrackletConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)
    graph_model: GraphModelConfig = Field(default_factory=GraphModelConfig)
    config_path: Path | None = None
