"""CLI entrypoint for the graph-native ReST application."""

from __future__ import annotations

import argparse
from pathlib import Path

from mcmt_core.config.loader import load_runtime_config
from mcmt_core.logging.setup import build_logger
from mcmt_core.runtime.manifest import write_run_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ReST graph application")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--mode", default="infer", choices=["train", "eval", "test", "infer"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_runtime_config(args.config, app_name="rest_app")
    logger = build_logger(cfg.logging)
    logger.info("Initialized rest_app in %s mode", args.mode)
    write_run_manifest(Path(cfg.system.output_root), cfg, app_name="rest_app", mode=args.mode)
    logger.info("Phase 1 scaffold only: pipeline implementation begins in later commits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
