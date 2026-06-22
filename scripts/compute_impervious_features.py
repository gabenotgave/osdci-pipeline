#!/usr/bin/env python3
"""Compute NLCD fractional impervious surface features per H3 cell."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_RESOLUTION = 6
CHUNK_SIZE = 25_000
BUILT_UP_THRESHOLD = 30.0
HIGH_IMPERVIOUS_THRESHOLD = 50.0

OUTPUT_COLUMNS = [
    "h3_index",
    "mean_impervious_pct",
    "max_impervious_pct",
    "pct_high_impervious",
    "is_built_up",
    "valid_pixel_count",
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
    """Write parquet via file handle (avoids pyarrow/rasterio filesystem conflict)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    with path.open("wb") as handle:
        pq.write_table(table, handle)


def stats_cache_path(resolution: int) -> Path:
    return RAW_DIR / f"nlcd_impervious_stats_res{resolution}.pkl"


def save_zonal_stats_cache(
    cache_path: Path,
    h3_index: pd.Series,
    stat_dicts: list[dict | None],
    nodata: int | float,
    *,
    partial: bool = False,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(
            {
                "h3_index": h3_index.tolist(),
                "stats": stat_dicts,
                "nodata": nodata,
                "complete": not partial,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    label = "checkpoint" if partial else "zonal stats"
    log(f"  Cached {label} ({len(stat_dicts):,} cells) to {cache_path.name}")


def load_zonal_stats_cache(
    cache_path: Path,
) -> tuple[list[str], list[dict | None], int | float, bool]:
    log(f"Loading cached zonal stats from {cache_path.name}...")
    with cache_path.open("rb") as handle:
        payload = pickle.load(handle)
    h3_index = payload["h3_index"]
    stat_dicts = payload["stats"]
    nodata = payload["nodata"]
    complete = payload.get("complete", len(stat_dicts) == len(h3_index))
    log(f"  {len(stat_dicts):,} cached cell statistics ({'complete' if complete else 'partial'})")
    return h3_index, stat_dicts, nodata, complete


def find_impervious_raster(raw_dir: Path) -> Path:
    default = raw_dir / "Annual_NLCD_FctImp_2024_CU_C1V1.tif"
    if default.exists():
        return default
    matches = sorted(
        path
        for path in raw_dir.glob("**/*.tif")
        if "nlcd" in path.name.lower() and "fctimp" in path.name.lower()
    )
    if not matches:
        raise FileNotFoundError(
            f"No NLCD impervious GeoTIFF found under {raw_dir}. "
            "Expected Annual_NLCD_FctImp_2024_CU_C1V1.tif"
        )
    return matches[0]


def open_raster_metadata(raster_path: Path) -> tuple[str, int | float]:
    import rasterio  # defer: avoids pyarrow 'file' scheme conflict with gpd.read_parquet

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Impervious raster has no CRS: {raster_path}")
        if src.nodata is None:
            raise ValueError(
                f"Impervious raster has no nodata value in metadata: {raster_path}"
            )
        return str(src.crs), src.nodata


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


def _high_impervious_fraction(pixels: np.ndarray) -> float | None:
    """Fraction of valid (non-nodata) pixels with impervious >= 50%."""
    valid = np.ma.asarray(pixels).compressed()
    if valid.size == 0:
        return None
    return float(np.count_nonzero(valid >= HIGH_IMPERVIOUS_THRESHOLD) / valid.size)


def run_zonal_stats_chunked(
    grid: gpd.GeoDataFrame,
    h3_index: pd.Series,
    raster_path: Path,
    nodata: int | float,
    chunk_size: int = CHUNK_SIZE,
    stats_cache: Path | None = None,
    start_offset: int = 0,
    existing_results: list[dict | None] | None = None,
) -> list[dict | None]:
    from rasterstats import zonal_stats  # defer: import after grid parquet load

    total = len(grid)
    results: list[dict | None] = list(existing_results or [])
    if start_offset > 0:
        log(f"Resuming zonal statistics from cell {start_offset:,}...")
    log(f"Running zonal statistics in chunks of {chunk_size:,}...")

    for start in range(start_offset, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = grid.iloc[start:end]
        chunk_stats = zonal_stats(
            vectors=chunk.geometry,
            raster=str(raster_path),
            stats=["mean", "max", "count"],
            add_stats={"high_impervious": _high_impervious_fraction},
            nodata=nodata,
            all_touched=False,
        )
        results.extend(chunk_stats)
        log(f"  Processed {end:,} / {total:,} cells...")
        if stats_cache is not None:
            save_zonal_stats_cache(
                stats_cache,
                h3_index,
                results,
                nodata,
                partial=(end < total),
            )

    return results


def stats_to_row(stats: dict | None) -> dict:
    empty = {
        "mean_impervious_pct": pd.NA,
        "max_impervious_pct": pd.NA,
        "pct_high_impervious": pd.NA,
        "is_built_up": 0,
        "valid_pixel_count": 0,
    }
    if not stats:
        return empty

    count = int(stats.get("count") or 0)
    if count == 0:
        return empty

    mean_val = stats.get("mean")
    max_val = stats.get("max")
    high_frac = stats.get("high_impervious")

    mean_impervious = float(mean_val) if mean_val is not None and pd.notna(mean_val) else pd.NA
    max_impervious = float(max_val) if max_val is not None and pd.notna(max_val) else pd.NA
    pct_high = float(high_frac) if high_frac is not None and pd.notna(high_frac) else pd.NA

    is_built_up = (
        int(mean_impervious >= BUILT_UP_THRESHOLD)
        if pd.notna(mean_impervious)
        else 0
    )

    return {
        "mean_impervious_pct": mean_impervious,
        "max_impervious_pct": max_impervious,
        "pct_high_impervious": pct_high,
        "is_built_up": is_built_up,
        "valid_pixel_count": count,
    }


def build_features(
    h3_index: pd.Series,
    raster_path: Path,
    nodata: int | float,
    grid_for_zonal: gpd.GeoDataFrame | None = None,
    stat_dicts: list[dict | None] | None = None,
    chunk_size: int = CHUNK_SIZE,
    stats_cache: Path | None = None,
) -> pd.DataFrame:
    if stat_dicts is None:
        if grid_for_zonal is None:
            raise ValueError("grid_for_zonal is required when stat_dicts is not provided")
        stat_dicts = run_zonal_stats_chunked(
            grid_for_zonal,
            h3_index,
            raster_path,
            nodata,
            chunk_size,
            stats_cache=stats_cache,
        )

    log("Aggregating impervious features...")
    rows = [stats_to_row(item) for item in stat_dicts]
    features = pd.DataFrame(rows)
    out = pd.concat([h3_index.reset_index(drop=True).to_frame(), features], axis=1)
    out["is_built_up"] = out["is_built_up"].astype(int)
    out["valid_pixel_count"] = out["valid_pixel_count"].astype(int)
    log(f"  Built feature table for {len(out):,} cells")
    return out[OUTPUT_COLUMNS]


def print_summary(df: pd.DataFrame, output_path: Path) -> None:
    no_data = df["valid_pixel_count"] == 0
    has_data = ~no_data
    log("\n" + "=" * 60)
    log("NLCD impervious surface features complete")
    log("=" * 60)
    log(f"Total cells processed:          {len(df):,}")
    log(f"Cells with no valid pixels:     {no_data.sum():,}")

    if has_data.any():
        mean_col = df.loc[has_data, "mean_impervious_pct"]
        log("\nmean_impervious_pct:")
        log(f"  min:    {mean_col.min():.1f}")
        log(f"  median: {mean_col.median():.1f}")
        log(f"  mean:   {mean_col.mean():.1f}")
        log(f"  max:    {mean_col.max():.1f}")

        built_up = df["is_built_up"].sum()
        log(f"\nis_built_up (mean >= {BUILT_UP_THRESHOLD:g}%):")
        log(f"  count: {built_up:,} ({100.0 * built_up / len(df):.1f}% of all cells)")

        greenfield = (df.loc[has_data, "mean_impervious_pct"] == 0).sum()
        log(f"\nCells with mean_impervious_pct == 0 (greenfield): {greenfield:,}")

    if output_path.exists():
        log(f"\nOutput: {output_path} ({_format_bytes(output_path.stat().st_size)})")


def compute_impervious_features(
    raster_path: Path | None = None,
    grid_path: Path | None = None,
    output_path: Path | None = None,
    chunk_size: int = CHUNK_SIZE,
    from_checkpoint: bool = False,
    resume: bool = False,
    stats_cache: Path | None = None,
) -> pd.DataFrame:
    raster_path = raster_path or find_impervious_raster(RAW_DIR)
    resolution = DEFAULT_RESOLUTION
    if grid_path is None:
        grid_path = PROCESSED_DIR / f"h3_grid_res{resolution}.parquet"
    if output_path is None:
        output_path = PROCESSED_DIR / f"features_impervious_res{resolution}.parquet"
    if stats_cache is None:
        stats_cache = stats_cache_path(resolution)

    if not raster_path.exists():
        raise FileNotFoundError(f"Impervious raster not found: {raster_path}")

    log(f"Using impervious raster: {raster_path.name}")
    if from_checkpoint:
        log("Mode: from checkpoint (skip zonal stats)")
    elif resume:
        log("Mode: resume (continue partial zonal stats if cache exists)")
    else:
        log("Mode: fresh computation")

    stat_dicts: list[dict | None] | None = None
    h3_index: pd.Series
    nodata: int | float

    if from_checkpoint:
        if not stats_cache.exists():
            raise FileNotFoundError(
                f"Zonal stats cache not found: {stats_cache}. "
                "Re-run without --from-checkpoint."
            )
        cached_h3, stat_dicts, nodata, complete = load_zonal_stats_cache(stats_cache)
        if not complete:
            raise FileNotFoundError(
                f"Cache is incomplete ({len(stat_dicts):,} / {len(cached_h3):,} cells). "
                "Re-run with --resume to continue, or delete the cache and start fresh."
            )
        h3_index = pd.Series(cached_h3, name="h3_index")
    elif resume and stats_cache.exists():
        cached_h3, stat_dicts, nodata, complete = load_zonal_stats_cache(stats_cache)
        h3_index = pd.Series(cached_h3, name="h3_index")
        if complete:
            log("  Cache already complete; aggregating and writing output...")
        else:
            if not grid_path.exists():
                raise FileNotFoundError(f"H3 grid not found: {grid_path}")
            grid = load_grid(grid_path)
            grid_reprojected = grid.to_crs(open_raster_metadata(raster_path)[0])
            if len(stat_dicts) > len(grid_reprojected):
                raise ValueError("Cache has more cells than grid; delete cache and re-run.")
            start_offset = len(stat_dicts)
            stat_dicts = run_zonal_stats_chunked(
                grid_reprojected,
                h3_index,
                raster_path,
                nodata,
                chunk_size,
                stats_cache=stats_cache,
                start_offset=start_offset,
                existing_results=stat_dicts,
            )
    else:
        if not grid_path.exists():
            raise FileNotFoundError(
                f"H3 grid not found: {grid_path}. "
                f"Run: python scripts/build_grid.py --resolution {resolution}"
            )
        grid = load_grid(grid_path)
        raster_crs, nodata = open_raster_metadata(raster_path)
        log(f"Raster nodata value: {nodata}")
        grid_reprojected = grid.to_crs(raster_crs)
        log(f"Raster CRS: {raster_crs} | H3 grid reprojected to match")
        h3_index = grid_reprojected["h3_index"]
        result = build_features(
            h3_index,
            raster_path,
            nodata,
            grid_for_zonal=grid_reprojected,
            chunk_size=chunk_size,
            stats_cache=stats_cache,
        )
        log(f"Writing {output_path.name}...")
        try:
            write_parquet_safe(result, output_path)
        except OSError as exc:
            raise OSError(f"Failed to write {output_path}: {exc}") from exc
        log("  Parquet write complete")
        print_summary(result, output_path)
        return result

    result = build_features(
        h3_index,
        raster_path,
        nodata,
        stat_dicts=stat_dicts,
    )
    log(f"Writing {output_path.name}...")
    try:
        write_parquet_safe(result, output_path)
    except OSError as exc:
        raise OSError(f"Failed to write {output_path}: {exc}") from exc
    log("  Parquet write complete")
    print_summary(result, output_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute NLCD fractional impervious surface features per H3 cell"
    )
    parser.add_argument(
        "--raster",
        type=Path,
        default=None,
        help="NLCD impervious GeoTIFF path (auto-discovered under data/raw/ if omitted)",
    )
    parser.add_argument(
        "--grid",
        type=Path,
        default=PROCESSED_DIR / f"h3_grid_res{DEFAULT_RESOLUTION}.parquet",
        help="H3 grid parquet with polygon geometry",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / f"features_impervious_res{DEFAULT_RESOLUTION}.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Cells per zonal_stats batch (default: {CHUNK_SIZE})",
    )
    parser.add_argument(
        "--from-checkpoint",
        action="store_true",
        help="Skip zonal stats; rebuild parquet from a complete .pkl cache",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from partial .pkl cache after an interrupted run",
    )
    parser.add_argument(
        "--stats-cache",
        type=Path,
        default=RAW_DIR / f"nlcd_impervious_stats_res{DEFAULT_RESOLUTION}.pkl",
        help="Pickle cache for raw zonal stat dicts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        compute_impervious_features(
            raster_path=args.raster,
            grid_path=args.grid,
            output_path=args.output,
            chunk_size=args.chunk_size,
            from_checkpoint=args.from_checkpoint,
            resume=args.resume,
            stats_cache=args.stats_cache,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        log(f"Error: {exc}")
        return 1
    # GDAL/rasterio teardown can hang for minutes on exit after long zonal_stats runs.
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
