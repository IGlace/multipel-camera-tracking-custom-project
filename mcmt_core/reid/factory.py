"""Factory helpers for building ReID extractors from runtime config."""

from __future__ import annotations

from mcmt_core.config.schema import ReIDConfig
from .mgn import MGNReIDExtractor


def build_reid_extractor(config: ReIDConfig, *, device: str = "cuda"):
    if config.backend == "mgn" and config.weights_path:
        return MGNReIDExtractor.from_checkpoint(
            config.weights_path,
            device=device,
            normalize_embeddings=config.normalize_embeddings,
        )
    if config.backend == "mgn":
        return MGNReIDExtractor(device=device, normalize_embeddings=config.normalize_embeddings)
    raise NotImplementedError(f"Unsupported reid backend: {config.backend}")
