#!/usr/bin/env python3
"""Compute distance from each H3 cell to the nearest ≥69 kV transmission line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_MIN_KV = 69
DEFAULT_RESOLUTION = 6
VOLTAGE_NULL_SENTINEL = -999999

OUTPUT_COLUMNS = ["h3_index", "dist_tx_line_m", "dist_tx_line_km", "source"]


def log(message: str) -> None:
    print(message, flush=True)


def _format_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.1f} GB"


def find_shapefile(raw_dir: Path) -> Path:
    patterns = [
        "**/Electric_Power_Transmission_Lines_A.shp",
        "**/*Transmission*Lines*.shp",
    ]
    for pattern in patterns:
        matches = sorted(raw_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"No HIFLD transmission lines shapefile found under {raw_dir}. "
        "Expected Electric_Power_Transmission_Lines_A.shp"
    )


def load_transmission_lines(shp_path: Path, min_kv: float) -> gpd.GeoDataFrame:
    log(f"Loading transmission lines from {shp_path}...")
    lines = gpd.read_file(shp_path)
    total = len(lines)

    if "VOLTAGE" not in lines.columns:
        raise ValueError(f"Shapefile missing VOLTAGE column: {lines.columns.tolist()}")

    lines = lines.copy()
    lines["VOLTAGE"] = pd.to_numeric(lines["VOLTAGE"], errors="coerce")
    lines = lines[
        lines["VOLTAGE"].notna()
        & (lines["VOLTAGE"] > 0)
        & (lines["VOLTAGE"] != VOLTAGE_NULL_SENTINEL)
        & (lines["VOLTAGE"] >= min_kv)
    ]
    log(f"  {len(lines):,} segments ≥ {min_kv:.0f} kV (of {total:,} total)")

    if lines.empty:
        raise ValueError(f"No transmission lines found with VOLTAGE >= {min_kv}")

    return lines[["geometry"]].reset_index(drop=True)


def load_grid(grid_path: Path) -> gpd.GeoDataFrame:
    log(f"Loading H3 grid from {grid_path}...")
    grid = gpd.read_parquet(grid_path)
    if grid.index.name == "h3_index":
        grid = grid.reset_index()
    if "h3_index" not in grid.columns:
        raise ValueError(f"Grid missing h3_index: {grid.columns.tolist()}")

    points = gpd.GeoDataFrame(
        grid[["h3_index", "lat", "lon"]],
        geometry=gpd.points_from_xy(grid["lon"], grid["lat"]),
        crs="EPSG:4326",
    )
    log(f"  {len(points):,} hex cells")
    return points


def compute_nearest_distance(
    grid_pts: gpd.GeoDataFrame, lines: gpd.GeoDataFrame
) -> pd.DataFrame:
    log("Computing nearest transmission line distance...")
    target_crs = lines.crs
    if target_crs is None:
        raise ValueError("Transmission lines shapefile has no CRS")

    points = grid_pts.to_crs(target_crs)
    nearest = gpd.sjoin_nearest(
        points[["h3_index", "geometry"]],
        lines,
        how="left",
        distance_col="dist_tx_line_m",
    )
    nearest = nearest[["h3_index", "dist_tx_line_m"]].drop_duplicates(
        subset="h3_index", keep="first"
    )
    nearest["dist_tx_line_km"] = nearest["dist_tx_line_m"] / 1000.0
    nearest["source"] = "HIFLD_Electric_Power_Transmission_Lines"
    return nearest[OUTPUT_COLUMNS]


def print_summary(df: pd.DataFrame, min_kv: float, output_path: Path) -> None:
    log("\n" + "=" * 60)
    log("Transmission distance feature complete")
    log("=" * 60)
    log(f"H3 cells:                    {len(df):,}")
    log(f"Min voltage filter:          ≥ {min_kv:.0f} kV")
    log("\nDistance to nearest line (km):")
    log(f"  min:    {df['dist_tx_line_km'].min():,.2f}")
    log(f"  median: {df['dist_tx_line_km'].median():,.2f}")
    log(f"  mean:   {df['dist_tx_line_km'].mean():,.2f}")
    log(f"  max:    {df['dist_tx_line_km'].max():,.2f}")
    log(f"  ≤ 1 km: {(df['dist_tx_line_km'] <= 1).sum():,} cells")
    log(f"  ≤ 5 km: {(df['dist_tx_line_km'] <= 5).sum():,} cells")
    if output_path.exists():
        log(f"\nOutput: {output_path} ({_format_bytes(output_path.stat().st_size)})")


def compute_tx_distance(
    shp_path: Path | None = None,
    grid_path: Path | None = None,
    output_path: Path | None = None,
    min_kv: float = DEFAULT_MIN_KV,
) -> pd.DataFrame:
    shp_path = shp_path or find_shapefile(RAW_DIR)
    resolution = DEFAULT_RESOLUTION
    if grid_path is None:
        grid_path = PROCESSED_DIR / f"h3_grid_res{resolution}.parquet"
    if output_path is None:
        output_path = PROCESSED_DIR / f"features_tx_distance_res{resolution}.parquet"

    if not grid_path.exists():
        raise FileNotFoundError(
            f"H3 grid not found: {grid_path}. Run: python scripts/build_grid.py --resolution {resolution}"
        )

    lines = load_transmission_lines(shp_path, min_kv)
    grid_pts = load_grid(grid_path)
    result = compute_nearest_distance(grid_pts, lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result.to_parquet(output_path, index=False)
    except OSError as exc:
        raise OSError(f"Failed to write {output_path}: {exc}") from exc

    print_summary(result, min_kv, output_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute H3 distance to nearest ≥69 kV transmission line (HIFLD)"
    )
    parser.add_argument(
        "--shp",
        type=Path,
        default=None,
        help="Path to HIFLD transmission lines shapefile (auto-discovered if omitted)",
    )
    parser.add_argument(
        "--grid",
        type=Path,
        default=PROCESSED_DIR / f"h3_grid_res{DEFAULT_RESOLUTION}.parquet",
        help="H3 grid parquet to compute distances for",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / f"features_tx_distance_res{DEFAULT_RESOLUTION}.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--min-kv",
        type=float,
        default=DEFAULT_MIN_KV,
        help=f"Minimum line voltage in kV (default: {DEFAULT_MIN_KV})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        compute_tx_distance(
            shp_path=args.shp,
            grid_path=args.grid,
            output_path=args.output,
            min_kv=args.min_kv,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        log(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
