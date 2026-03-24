"""Shared command entrypoint for the MCMT platform.

This script provides a single container/runtime entrypoint that validates the GPU
requirement, resolves the requested app or trainer, and dispatches execution to the
corresponding Python module.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bootstrap_gpu import require_gpu

ROUTES = {
    "rest": "apps.rest_app.main",
    "heuristic": "apps.heuristic_app.main",
    "realtime": "apps.realtime_app.main",
    "trainer-reid-mgn": "trainers.reid_mgn.main",
    "trainer-attributes": "trainers.attributes.main",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MCMT platform entrypoint")
    parser.add_argument(
        "target",
        choices=sorted(ROUTES.keys()),
        help="Target application or trainer to run.",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to the target module.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    parsed = parser.parse_args()

    require_gpu()

    module_name = ROUTES[parsed.target]
    module = importlib.import_module(module_name)
    if not hasattr(module, "main"):
        raise AttributeError(f"Target module '{module_name}' does not expose a main() function.")

    forwarded_args = parsed.args
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

    return int(module.main(forwarded_args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
