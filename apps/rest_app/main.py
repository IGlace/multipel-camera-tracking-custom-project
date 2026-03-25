"""CLI entrypoint for the graph-native ReST application."""

from __future__ import annotations

import argparse
from pathlib import Path

from apps.rest_app.frame_graph.neural_pipeline import run_frame_graph_neural_inference
from apps.rest_app.frame_graph.pipeline import run_frame_graph_baseline
from apps.rest_app.runners.model_probe import run_graph_tensor_probe
from apps.rest_app.runners.model_sanity import run_gnn_sanity
from mcmt_core.config.loader import load_runtime_config
from mcmt_core.logging.setup import build_logger
from mcmt_core.runtime.manifest import write_run_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ReST graph application")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--mode", default="infer", choices=["train", "eval", "test", "infer"])
    parser.add_argument(
        "--pipeline",
        default="frame_graph_baseline",
        choices=["frame_graph_baseline", "gnn_sanity", "graph_tensor_probe", "neural_edge_inference"],
        help="Pipeline implementation to run for the current phase.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_runtime_config(args.config, app_name="rest_app")
    output_root = Path(cfg.system.output_root)
    logger = build_logger(cfg.logging, output_root=output_root)
    logger.info("Initialized rest_app in %s mode with %s", args.mode, args.pipeline)
    write_run_manifest(output_root, cfg, app_name="rest_app", mode=args.mode)

    if args.mode == "infer" and args.pipeline == "frame_graph_baseline":
        run_frame_graph_baseline(cfg, logger)
        return 0
    if args.pipeline == "gnn_sanity":
        run_gnn_sanity(cfg, logger)
        return 0
    if args.pipeline == "graph_tensor_probe":
        run_graph_tensor_probe(cfg, logger)
        return 0
    if args.mode == "infer" and args.pipeline == "neural_edge_inference":
        run_frame_graph_neural_inference(cfg, logger)
        return 0

    logger.info("This mode is scaffolded but not yet implemented in the current phase.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
