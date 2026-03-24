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


class RuntimeConfig(BaseModel):
    app_name: str = Field(default="unknown")
    system: SystemConfig = Field(default_factory=SystemConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    config_path: Path | None = None
