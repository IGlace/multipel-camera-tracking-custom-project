"""CLI entrypoint for the MGN trainer backend."""

from __future__ import annotations

import argparse
from pathlib import Path

from mcmt_core.config.loader import load_runtime_config
from mcmt_core.logging.setup import build_logger
from mcmt_core.runtime.manifest import write_run_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ReID MGN trainer backend")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--mode", default="train", choices=["train", "eval", "export"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_runtime_config(args.config, app_name="trainer_reid_mgn")
    output_root = Path(cfg.system.output_root)
    logger = build_logger(cfg.logging, output_root=output_root)
    logger.info("Initialized trainer_reid_mgn in %s mode", args.mode)
    write_run_manifest(output_root, cfg, app_name="trainer_reid_mgn", mode=args.mode)
    logger.info("Phase 1 scaffold only: MGN integration begins in later commits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
