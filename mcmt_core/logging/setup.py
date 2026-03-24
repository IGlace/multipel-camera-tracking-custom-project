"""Rich-backed logging setup for Phase 1."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler

from mcmt_core.config.schema import LoggingConfig


def build_logger(cfg: LoggingConfig, output_root: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("mcmt")
    level_map = {
        "quiet": logging.ERROR,
        "normal": logging.INFO,
        "detailed": logging.DEBUG,
        "trace": logging.DEBUG,
    }
    logger.setLevel(level_map[cfg.verbosity])
    logger.handlers.clear()

    console_handler = RichHandler(rich_tracebacks=True, markup=True)
    console_handler.setLevel(level_map[cfg.verbosity])
    logger.addHandler(console_handler)

    if cfg.log_to_file:
        base_dir = output_root if output_root is not None else Path("outputs")
        log_path = base_dir / cfg.log_filename
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level_map[cfg.verbosity])
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
