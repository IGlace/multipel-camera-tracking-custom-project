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


class RuntimeConfig(BaseModel):
    app_name: str = Field(default="unknown")
    system: SystemConfig = Field(default_factory=SystemConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    segmentor: SegmentorConfig = Field(default_factory=SegmentorConfig)
    local_tracker: LocalTrackerConfig = Field(default_factory=LocalTrackerConfig)
    reid: ReIDConfig = Field(default_factory=ReIDConfig)
    attributes: AttributesConfig = Field(default_factory=AttributesConfig)
    config_path: Path | None = None
