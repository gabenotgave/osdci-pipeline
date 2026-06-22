#!/usr/bin/env python3
"""Compute terrain slope features per H3 cell from USGS 3DEP via py3dep."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_RESOLUTION = 6
DEFAULT_TILE_DEG = 2.0
DEFAULT_PY3DEP_RESOLUTION_M = 90
SLOPE_NODATA = 255

CONUS_MIN_LAT = 24.0
CONUS_MAX_LAT = 50.0
CONUS_MIN_LON = -125.0
CONUS_MAX_LON = -66.0

FLAT_THRESHOLD_DEG = 5.0
VERY_FLAT_THRESHOLD_DEG = 2.0

CHECKPOINT_PATH = RAW_DIR / "slope_checkpoint.jsonl"
SLOPE_CRS = "EPSG:5070"

OUTPUT_COLUMNS = [
    "h3_index",
    "mean_slope_deg",
    "max_slope_deg",
    "std_slope_deg",
    "is_flat",
    "is_very_flat",
    "slope_pixel_count",
]


def log(message: str) -> None:
    print(message, flush=True)


def _format_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.1f} GB"


def write_parquet_safe(df: pd.DataFrame, path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    with path.open("wb") as handle:
        pq.write_table(table, handle)


def load_grid(grid_path: Path) -> gpd.GeoDataFrame:
    log(f"Loading H3 grid from {grid_path}...")
    grid = gpd.read_parquet(grid_path)
    if grid.index.name == "h3_index":
        grid = grid.reset_index()
    if "h3_index" not in grid.columns:
        raise ValueError(f"Grid missing h3_index: {grid.columns.tolist()}")
    if grid.crs is None:
        grid = grid.set_crs("EPSG:4326")
    log(f"  {len(grid):,} hex cells")
    return grid


def filter_conus_grid(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "is_conus" in grid.columns:
        conus = grid[grid["is_conus"].astype(bool)].copy()
        log(f"  CONUS cells (is_conus): {len(conus):,}")
        return conus

    conus = grid[
        (grid["lat"] >= CONUS_MIN_LAT)
        & (grid["lat"] <= CONUS_MAX_LAT)
        & (grid["lon"] >= CONUS_MIN_LON)
        & (grid["lon"] <= CONUS_MAX_LON)
    ].copy()
    log(f"  CONUS cells (bbox filter): {len(conus):,}")
    return conus


def build_tile_bounds(tile_deg: float) -> list[tuple[float, float, float, float]]:
    tiles: list[tuple[float, float, float, float]] = []
    lat = CONUS_MIN_LAT
    while lat < CONUS_MAX_LAT:
        next_lat = min(lat + tile_deg, CONUS_MAX_LAT)
        lon = CONUS_MIN_LON
        while lon < CONUS_MAX_LON:
            next_lon = min(lon + tile_deg, CONUS_MAX_LON)
            tiles.append((lon, lat, next_lon, next_lat))
            lon += tile_deg
        lat += tile_deg
    return tiles


def tile_contains_cells(
    grid: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
) -> bool:
    minx, miny, maxx, maxy = bounds
    lat_ok = (grid["lat"] >= miny) & (grid["lat"] <= maxy)
    lon_ok = (grid["lon"] >= minx) & (grid["lon"] <= maxx)
    return bool((lat_ok & lon_ok).any())


def cells_in_tile(
    grid: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = bounds
    mask = (
        (grid["lat"] >= miny)
        & (grid["lat"] <= maxy)
        & (grid["lon"] >= minx)
        & (grid["lon"] <= maxx)
    )
    return grid.loc[mask, ["h3_index", "geometry"]].copy()


def load_checkpoint(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    results: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            results[record["h3_index"]] = record
    return results


def append_checkpoint_records(path: Path, records: list[dict]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def tile_fully_processed(
    tile_cells: gpd.GeoDataFrame,
    checkpoint: dict[str, dict],
) -> bool:
    if tile_cells.empty:
        return True
    indices = tile_cells["h3_index"].tolist()
    return all(h3 in checkpoint for h3 in indices)


def fetch_slope_raster(
    tile_bounds: tuple[float, float, float, float],
    resolution_m: int,
):
    import py3dep

    tile_box = box(*tile_bounds)
    return py3dep.get_map(
        "Slope Degrees",
        geometry=tile_box,
        resolution=resolution_m,
        geo_crs="EPSG:4326",
        crs=SLOPE_CRS,
    )


def zonal_stats_for_tile(
    tile_cells: gpd.GeoDataFrame,
    slope_path: Path,
    nodata: int | float,
) -> list[dict | None]:
    from rasterstats import zonal_stats

    cells_5070 = tile_cells.to_crs(SLOPE_CRS)
    return zonal_stats(
        vectors=cells_5070.geometry,
        raster=str(slope_path),
        stats=["mean", "max", "std", "count"],
        nodata=nodata,
        all_touched=False,
    )


def stats_record(h3_index: str, stats: dict | None) -> dict:
    record: dict = {"h3_index": h3_index}
    if not stats:
        record.update({"mean": None, "max": None, "std": None, "count": 0})
        return record

    count = int(stats.get("count") or 0)
    record["count"] = count
    for key in ("mean", "max", "std"):
        val = stats.get(key)
        record[key] = float(val) if val is not None and pd.notna(val) else None
    return record


def records_to_features(
    conus_grid: gpd.GeoDataFrame,
    checkpoint: dict[str, dict],
) -> pd.DataFrame:
    rows: list[dict] = []
    for h3_index in conus_grid["h3_index"]:
        stats = checkpoint.get(h3_index)
        empty = {
            "h3_index": h3_index,
            "mean_slope_deg": pd.NA,
            "max_slope_deg": pd.NA,
            "std_slope_deg": pd.NA,
            "is_flat": 0,
            "is_very_flat": 0,
            "slope_pixel_count": 0,
        }
        if not stats or int(stats.get("count") or 0) == 0:
            rows.append(empty)
            continue

        mean_val = stats.get("mean")
        max_val = stats.get("max")
        std_val = stats.get("std")
        count = int(stats["count"])

        mean_slope = float(mean_val) if mean_val is not None else pd.NA
        max_slope = float(max_val) if max_val is not None else pd.NA
        std_slope = float(std_val) if std_val is not None else pd.NA

        is_flat = (
            int(mean_slope <= FLAT_THRESHOLD_DEG) if pd.notna(mean_slope) else 0
        )
        is_very_flat = (
            int(mean_slope <= VERY_FLAT_THRESHOLD_DEG) if pd.notna(mean_slope) else 0
        )

        rows.append(
            {
                "h3_index": h3_index,
                "mean_slope_deg": mean_slope,
                "max_slope_deg": max_slope,
                "std_slope_deg": std_slope,
                "is_flat": is_flat,
                "is_very_flat": is_very_flat,
                "slope_pixel_count": count,
            }
        )

    out = pd.DataFrame(rows)
    out["is_flat"] = out["is_flat"].astype(int)
    out["is_very_flat"] = out["is_very_flat"].astype(int)
    out["slope_pixel_count"] = out["slope_pixel_count"].astype(int)
    return out[OUTPUT_COLUMNS]


def process_tiles(
    conus_grid: gpd.GeoDataFrame,
    tile_deg: float,
    resolution_m: int,
    checkpoint_path: Path,
    from_checkpoint: bool,
) -> tuple[dict[str, dict], int, int]:
    checkpoint = load_checkpoint(checkpoint_path) if from_checkpoint else {}
    if from_checkpoint and checkpoint:
        log(f"Loaded {len(checkpoint):,} cells from checkpoint")
    elif not from_checkpoint:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            log("Starting fresh (removed existing checkpoint)")
        else:
            log("Starting fresh")

    all_bounds = build_tile_bounds(tile_deg)
    tiles = [b for b in all_bounds if tile_contains_cells(conus_grid, b)]
    log(f"Processing {len(tiles):,} non-empty tiles at {resolution_m}m resolution")

    processed_tiles = 0
    failed_tiles = 0
    failed_indices: list[int] = []

    for i, tile_bounds in enumerate(tiles, start=1):
        tile_cells = cells_in_tile(conus_grid, tile_bounds)
        pending = tile_cells[
            ~tile_cells["h3_index"].isin(checkpoint.keys())
        ]
        if tile_fully_processed(tile_cells, checkpoint):
            if i % 10 == 0 or i == len(tiles):
                log(
                    f"Tile {i}/{len(tiles)} skipped (checkpoint) | "
                    f"{len(checkpoint):,} cells | {failed_tiles} failed"
                )
            continue

        if pending.empty:
            continue

        try:
            slope_da = fetch_slope_raster(tile_bounds, resolution_m)
            nodata = slope_da.rio.nodata if slope_da.rio.nodata is not None else SLOPE_NODATA

            with tempfile.TemporaryDirectory(prefix="osdci_slope_") as tmpdir:
                slope_path = Path(tmpdir) / "slope.tif"
                slope_da.rio.to_raster(slope_path)
                tile_stats = zonal_stats_for_tile(pending, slope_path, nodata)

            records = [
                stats_record(h3, stat)
                for h3, stat in zip(pending["h3_index"], tile_stats, strict=True)
            ]
            append_checkpoint_records(checkpoint_path, records)
            for record in records:
                checkpoint[record["h3_index"]] = record
            processed_tiles += 1
        except Exception as exc:
            failed_tiles += 1
            failed_indices.append(i)
            log(f"  Warning: tile {i}/{len(tiles)} failed ({tile_bounds}): {exc}")
            time.sleep(5)

        if i % 10 == 0 or i == len(tiles):
            log(
                f"Tile {i}/{len(tiles)} done | {len(checkpoint):,} cells | "
                f"{failed_tiles} failed"
            )

        time.sleep(1)

    if failed_indices:
        log(f"Failed tile indices: {failed_indices[:20]}{'...' if len(failed_indices) > 20 else ''}")

    return checkpoint, processed_tiles, failed_tiles


def print_summary(
    df: pd.DataFrame,
    processed_tiles: int,
    failed_tiles: int,
    output_path: Path,
) -> None:
    has_data = df["slope_pixel_count"] > 0
    log("\n" + "=" * 60)
    log("Slope features complete")
    log("=" * 60)
    log(f"Tiles processed:              {processed_tiles:,}")
    log(f"Tiles failed:                 {failed_tiles:,}")
    log(f"Cells with slope data:        {has_data.sum():,}")
    log(f"Cells with null slope:        {(~has_data).sum():,}")

    if has_data.any():
        mean_col = df.loc[has_data, "mean_slope_deg"]
        log("\nmean_slope_deg:")
        log(f"  min:    {mean_col.min():.2f}")
        log(f"  median: {mean_col.median():.2f}")
        log(f"  mean:   {mean_col.mean():.2f}")
        log(f"  max:    {mean_col.max():.2f}")

        flat = df["is_flat"].sum()
        very_flat = df["is_very_flat"].sum()
        log(f"\nis_flat (<= {FLAT_THRESHOLD_DEG:g}°): {flat:,} ({100.0 * flat / len(df):.1f}%)")
        log(
            f"is_very_flat (<= {VERY_FLAT_THRESHOLD_DEG:g}°): "
            f"{very_flat:,} ({100.0 * very_flat / len(df):.1f}%)"
        )

    if output_path.exists():
        log(f"\nOutput: {output_path} ({_format_bytes(output_path.stat().st_size)})")


def compute_slope_features(
    grid_path: Path | None = None,
    output_path: Path | None = None,
    checkpoint_path: Path | None = None,
    tile_deg: float = DEFAULT_TILE_DEG,
    resolution_m: int = DEFAULT_PY3DEP_RESOLUTION_M,
    from_checkpoint: bool = False,
    assemble_only: bool = False,
) -> pd.DataFrame:
    resolution = DEFAULT_RESOLUTION
    if grid_path is None:
        grid_path = PROCESSED_DIR / f"h3_grid_res{resolution}.parquet"
    if output_path is None:
        output_path = PROCESSED_DIR / f"features_slope_res{resolution}.parquet"
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATH

    if not grid_path.exists():
        raise FileNotFoundError(
            f"H3 grid not found: {grid_path}. "
            f"Run: python scripts/build_grid.py --resolution {resolution}"
        )

    grid = load_grid(grid_path)
    conus_grid = filter_conus_grid(grid)

    if assemble_only:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path)
        log(f"Assembling from checkpoint only ({len(checkpoint):,} cells)...")
        processed_tiles = 0
        failed_tiles = 0
    else:
        if from_checkpoint:
            log("Mode: resume from checkpoint")
        else:
            log("Mode: fresh tile fetch")
        checkpoint, processed_tiles, failed_tiles = process_tiles(
            conus_grid,
            tile_deg,
            resolution_m,
            checkpoint_path,
            from_checkpoint=from_checkpoint,
        )

    result = records_to_features(conus_grid, checkpoint)
    log(f"Writing {output_path.name}...")
    write_parquet_safe(result, output_path)
    log("  Parquet write complete")
    print_summary(result, processed_tiles, failed_tiles, output_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute USGS 3DEP slope features per H3 cell (py3dep, tiled)"
    )
    parser.add_argument(
        "--grid",
        type=Path,
        default=PROCESSED_DIR / f"h3_grid_res{DEFAULT_RESOLUTION}.parquet",
        help="H3 grid parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / f"features_slope_res{DEFAULT_RESOLUTION}.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help="JSONL checkpoint path",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_PY3DEP_RESOLUTION_M,
        help="py3dep fetch resolution in meters (default: 90)",
    )
    parser.add_argument(
        "--tile-deg",
        type=float,
        default=DEFAULT_TILE_DEG,
        help="Tile size in degrees (default: 2.0)",
    )
    parser.add_argument(
        "--from-checkpoint",
        action="store_true",
        help="Resume: load checkpoint and skip fully processed tiles",
    )
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help="Build parquet from checkpoint without fetching new tiles",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        compute_slope_features(
            grid_path=args.grid,
            output_path=args.output,
            checkpoint_path=args.checkpoint,
            tile_deg=args.tile_deg,
            resolution_m=args.resolution,
            from_checkpoint=args.from_checkpoint,
            assemble_only=args.assemble_only,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        log(f"Error: {exc}")
        return 1
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
