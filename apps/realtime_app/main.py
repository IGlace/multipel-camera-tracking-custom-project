"""CLI entrypoint for the realtime tracking application."""

from __future__ import annotations

import argparse
from pathlib import Path

from apps.realtime_app.frame_graph_runtime.pipeline import run_realtime_frame_runtime_baseline
from mcmt_core.config.loader import load_runtime_config
from mcmt_core.logging.setup import build_logger
from mcmt_core.runtime.manifest import write_run_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the realtime tracking application")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--mode", default="infer", choices=["infer", "live", "batch_video"])
    parser.add_argument(
        "--pipeline",
        default="frame_runtime_baseline",
        choices=["frame_runtime_baseline"],
        help="Pipeline implementation to run for the current phase.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_runtime_config(args.config, app_name="realtime_app")
    output_root = Path(cfg.system.output_root)
    logger = build_logger(cfg.logging, output_root=output_root)
    logger.info("Initialized realtime_app in %s mode with %s", args.mode, args.pipeline)
    write_run_manifest(output_root, cfg, app_name="realtime_app", mode=args.mode)

    if args.pipeline == "frame_runtime_baseline":
        run_realtime_frame_runtime_baseline(cfg, logger, mode=args.mode)
        return 0

    logger.info("This mode is scaffolded but not yet implemented in the current phase.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
