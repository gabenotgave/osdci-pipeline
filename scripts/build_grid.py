#!/usr/bin/env python3
"""CLI entry point for building the CONUS H3 grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from osdci.grid import DEFAULT_RESOLUTION, build_h3_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an H3 hexagonal grid covering the USA boundary."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help=f"H3 resolution (default: {DEFAULT_RESOLUTION})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Directory for parquet and CSV outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_h3_grid(resolution=args.resolution, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
