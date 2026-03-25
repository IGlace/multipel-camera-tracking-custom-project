"""CLI entrypoint for the heuristic non-graph application."""

from __future__ import annotations

import argparse
from pathlib import Path

from apps.heuristic_app.frame_match.pipeline import run_frame_match_baseline
from mcmt_core.config.loader import load_runtime_config
from mcmt_core.logging.setup import build_logger
from mcmt_core.runtime.manifest import write_run_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the heuristic tracking application")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--mode", default="infer", choices=["test", "infer"])
    parser.add_argument(
        "--pipeline",
        default="frame_match_baseline",
        choices=["frame_match_baseline"],
        help="Pipeline implementation to run for the current phase.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_runtime_config(args.config, app_name="heuristic_app")
    output_root = Path(cfg.system.output_root)
    logger = build_logger(cfg.logging, output_root=output_root)
    logger.info("Initialized heuristic_app in %s mode with %s", args.mode, args.pipeline)
    write_run_manifest(output_root, cfg, app_name="heuristic_app", mode=args.mode)

    if args.mode == "infer" and args.pipeline == "frame_match_baseline":
        run_frame_match_baseline(cfg, logger)
        return 0

    logger.info("This mode is scaffolded but not yet implemented in the current phase.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
