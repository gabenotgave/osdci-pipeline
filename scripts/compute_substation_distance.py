#!/usr/bin/env python3
"""Compute distance from each H3 cell to the nearest electrical substation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_RESOLUTION = 6

OUTPUT_COLUMNS = ["h3_index", "dist_substation_m", "dist_substation_km", "source"]


def log(message: str) -> None:
    print(message, flush=True)


def _format_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.1f} GB"


def find_substations_csv(raw_dir: Path) -> Path:
    for name in ("Substations.csv", "substations.csv"):
        path = raw_dir / name
        if path.exists():
            return path
    matches = sorted(raw_dir.glob("**/*[Ss]ubstation*.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No substations CSV found under {raw_dir}. Expected Substations.csv"
    )


def load_substations(
    csv_path: Path,
    in_service_only: bool = True,
    usa_only: bool = True,
) -> gpd.GeoDataFrame:
    log(f"Loading substations from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    total = len(df)

    required = {"LATITUDE", "LONGITUDE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns {missing}: {df.columns.tolist()}")

    df = df.copy()
    df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    df = df.dropna(subset=["LATITUDE", "LONGITUDE"])

    if usa_only and "COUNTRY" in df.columns:
        df = df[df["COUNTRY"].astype(str).str.strip().str.upper() == "USA"]

    if in_service_only and "STATUS" in df.columns:
        df = df[df["STATUS"].astype(str).str.strip().str.upper() == "IN SERVICE"]

    log(f"  {len(df):,} substations kept (of {total:,} total)")

    if df.empty:
        raise ValueError("No substations remaining after filters")

    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["LONGITUDE"], df["LATITUDE"]),
        crs="EPSG:4326",
    )[["geometry"]].reset_index(drop=True)


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
    grid_pts: gpd.GeoDataFrame, substations: gpd.GeoDataFrame
) -> pd.DataFrame:
    log("Computing nearest substation distance...")
    # Project to Web Mercator for meter distances (same approach as transmission lines)
    substations_m = substations.to_crs(epsg=3857)
    points_m = grid_pts.to_crs(epsg=3857)

    nearest = gpd.sjoin_nearest(
        points_m[["h3_index", "geometry"]],
        substations_m,
        how="left",
        distance_col="dist_substation_m",
    )
    nearest = nearest[["h3_index", "dist_substation_m"]].drop_duplicates(
        subset="h3_index", keep="first"
    )
    nearest["dist_substation_km"] = nearest["dist_substation_m"] / 1000.0
    nearest["source"] = "HIFLD_Substations"
    return nearest[OUTPUT_COLUMNS]


def print_summary(df: pd.DataFrame, output_path: Path) -> None:
    log("\n" + "=" * 60)
    log("Substation distance feature complete")
    log("=" * 60)
    log(f"H3 cells:                    {len(df):,}")
    log("\nDistance to nearest substation (km):")
    log(f"  min:    {df['dist_substation_km'].min():,.2f}")
    log(f"  median: {df['dist_substation_km'].median():,.2f}")
    log(f"  mean:   {df['dist_substation_km'].mean():,.2f}")
    log(f"  max:    {df['dist_substation_km'].max():,.2f}")
    log(f"  ≤ 1 km: {(df['dist_substation_km'] <= 1).sum():,} cells")
    log(f"  ≤ 5 km: {(df['dist_substation_km'] <= 5).sum():,} cells")
    if output_path.exists():
        log(f"\nOutput: {output_path} ({_format_bytes(output_path.stat().st_size)})")


def compute_substation_distance(
    csv_path: Path | None = None,
    grid_path: Path | None = None,
    output_path: Path | None = None,
    in_service_only: bool = True,
    usa_only: bool = True,
) -> pd.DataFrame:
    csv_path = csv_path or find_substations_csv(RAW_DIR)
    resolution = DEFAULT_RESOLUTION
    if grid_path is None:
        grid_path = PROCESSED_DIR / f"h3_grid_res{resolution}.parquet"
    if output_path is None:
        output_path = PROCESSED_DIR / f"features_substation_distance_res{resolution}.parquet"

    if not grid_path.exists():
        raise FileNotFoundError(
            f"H3 grid not found: {grid_path}. "
            f"Run: python scripts/build_grid.py --resolution {resolution}"
        )

    substations = load_substations(csv_path, in_service_only=in_service_only, usa_only=usa_only)
    grid_pts = load_grid(grid_path)
    result = compute_nearest_distance(grid_pts, substations)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result.to_parquet(output_path, index=False)
    except OSError as exc:
        raise OSError(f"Failed to write {output_path}: {exc}") from exc

    print_summary(result, output_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute H3 distance to nearest substation (HIFLD)"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to Substations.csv (auto-discovered under data/raw/ if omitted)",
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
        default=PROCESSED_DIR / f"features_substation_distance_res{DEFAULT_RESOLUTION}.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--include-all-status",
        action="store_true",
        help="Include substations not marked IN SERVICE",
    )
    parser.add_argument(
        "--include-non-usa",
        action="store_true",
        help="Include substations outside USA",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        compute_substation_distance(
            csv_path=args.csv,
            grid_path=args.grid,
            output_path=args.output,
            in_service_only=not args.include_all_status,
            usa_only=not args.include_non_usa,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        log(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
