#!/usr/bin/env python3
"""Compute NLCD land cover zonal statistics per H3 cell."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_RESOLUTION = 6
CHUNK_SIZE = 10_000
DEFAULT_NODATA = -9999

NLCD_CLASS_NAMES: dict[int, str] = {
    11: "Open Water",
    12: "Perennial Ice/Snow",
    21: "Developed, Open Space",
    22: "Developed, Low Intensity",
    23: "Developed, Medium Intensity",
    24: "Developed, High Intensity",
    31: "Barren Land (Rock/Sand/Clay)",
    41: "Deciduous Forest",
    42: "Evergreen Forest",
    43: "Mixed Forest",
    52: "Shrub/Scrub",
    71: "Grassland/Herbaceous",
    81: "Pasture/Hay",
    82: "Cultivated Crops",
    90: "Woody Wetlands",
    95: "Emergent Herbaceous Wetlands",
}

DEVELOPED = {21, 22, 23, 24}
DEVELOPED_HIGH = {23, 24}
FOREST = {41, 42, 43}
AGRICULTURE = {81, 82}
SHRUB_GRASSLAND = {52, 71}
WETLAND = {90, 95}
BARREN = {31}
WATER = {11}

OUTPUT_COLUMNS = [
    "h3_index",
    "dominant_class",
    "dominant_class_name",
    "pct_developed",
    "pct_developed_high",
    "pct_forest",
    "pct_agriculture",
    "pct_shrub_grassland",
    "pct_wetland",
    "pct_barren",
    "pct_water",
    "total_pixels",
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
    return PROCESSED_DIR / f"nlcd_zonal_stats_res{resolution}.pkl"


def save_zonal_stats_cache(
    cache_path: Path,
    h3_index: pd.Series,
    stat_dicts: list[dict | None],
    nodata: int | float,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(
            {"h3_index": h3_index.tolist(), "stats": stat_dicts, "nodata": nodata},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    log(f"  Cached zonal stats to {cache_path.name}")


def load_zonal_stats_cache(cache_path: Path) -> tuple[list[str], list[dict | None], int | float]:
    log(f"Loading cached zonal stats from {cache_path.name}...")
    with cache_path.open("rb") as handle:
        payload = pickle.load(handle)
    h3_index = payload["h3_index"]
    stat_dicts = payload["stats"]
    nodata = payload["nodata"]
    log(f"  {len(stat_dicts):,} cached cell statistics")
    return h3_index, stat_dicts, nodata


def find_nlcd_raster(raw_dir: Path) -> Path:
    matches = sorted(
        path
        for path in raw_dir.glob("**/*.tif")
        if "nlcd" in path.name.lower() and "lndcov" in path.name.lower()
    )
    if not matches:
        raise FileNotFoundError(
            f"No NLCD land cover GeoTIFF found under {raw_dir}. "
            "Expected a .tif with 'NLCD' and 'LndCov' in the filename."
        )
    return matches[0]


def open_nlcd_metadata(raster_path: Path) -> tuple[str, int | float]:
    import rasterio  # defer: avoids pyarrow 'file' scheme conflict with gpd.read_parquet

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"NLCD raster has no CRS: {raster_path}")
        nodata = src.nodata if src.nodata is not None else DEFAULT_NODATA
        return str(src.crs), nodata


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


def run_zonal_stats_chunked(
    grid: gpd.GeoDataFrame,
    raster_path: Path,
    nodata: int | float,
    chunk_size: int = CHUNK_SIZE,
) -> list[dict | None]:
    from rasterstats import zonal_stats  # defer: import after grid parquet load

    total = len(grid)
    results: list[dict | None] = []
    log(f"Running zonal statistics in chunks of {chunk_size:,}...")

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = grid.iloc[start:end]
        chunk_stats = zonal_stats(
            vectors=chunk.geometry,
            raster=str(raster_path),
            categorical=True,
            nodata=nodata,
            all_touched=False,
        )
        results.extend(chunk_stats)
        log(f"  Processed {end:,} / {total:,} cells...")

    return results


def _class_sum(counts: dict[int, int], classes: set[int]) -> int:
    return sum(counts.get(code, 0) for code in classes)


def counts_to_row(counts: dict | None, nodata: int | float) -> dict:
    empty = {
        "dominant_class": pd.NA,
        "dominant_class_name": pd.NA,
        "pct_developed": pd.NA,
        "pct_developed_high": pd.NA,
        "pct_forest": pd.NA,
        "pct_agriculture": pd.NA,
        "pct_shrub_grassland": pd.NA,
        "pct_wetland": pd.NA,
        "pct_barren": pd.NA,
        "pct_water": pd.NA,
        "total_pixels": 0,
    }
    if not counts:
        return empty

    parsed: dict[int, int] = {}
    for key, value in counts.items():
        if key is None:
            continue
        code = int(key)
        if nodata is not None and code == int(nodata):
            continue
        if value and int(value) > 0:
            parsed[code] = int(value)

    total_pixels = sum(parsed.values())
    if total_pixels == 0:
        return empty

    dominant_class = max(parsed, key=parsed.get)
    dominant_name = NLCD_CLASS_NAMES.get(dominant_class, f"Class {dominant_class}")

    def pct(class_set: set[int]) -> float:
        return _class_sum(parsed, class_set) / total_pixels

    return {
        "dominant_class": dominant_class,
        "dominant_class_name": dominant_name,
        "pct_developed": pct(DEVELOPED),
        "pct_developed_high": pct(DEVELOPED_HIGH),
        "pct_forest": pct(FOREST),
        "pct_agriculture": pct(AGRICULTURE),
        "pct_shrub_grassland": pct(SHRUB_GRASSLAND),
        "pct_wetland": pct(WETLAND),
        "pct_barren": pct(BARREN),
        "pct_water": pct(WATER),
        "total_pixels": total_pixels,
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
        stat_dicts = run_zonal_stats_chunked(grid_for_zonal, raster_path, nodata, chunk_size)
        if stats_cache is not None:
            save_zonal_stats_cache(stats_cache, h3_index, stat_dicts, nodata)

    log("Aggregating class percentages...")
    rows = [counts_to_row(item, nodata) for item in stat_dicts]
    features = pd.DataFrame(rows)
    out = pd.concat([h3_index.reset_index(drop=True).to_frame(), features], axis=1)
    out["dominant_class"] = out["dominant_class"].astype("Int64")
    out["total_pixels"] = out["total_pixels"].astype(int)
    log(f"  Built feature table for {len(out):,} cells")
    return out[OUTPUT_COLUMNS]


def print_summary(df: pd.DataFrame, output_path: Path) -> None:
    no_data = df["total_pixels"] == 0
    log("\n" + "=" * 60)
    log("NLCD land cover features complete")
    log("=" * 60)
    log(f"Total cells processed:          {len(df):,}")
    log(f"Cells with no data:             {no_data.sum():,}")

    valid = df["dominant_class"].notna()
    if valid.any():
        log("\nTop 5 dominant land cover classes (by cell count):")
        top = (
            df.loc[valid, "dominant_class_name"]
            .value_counts()
            .head(5)
        )
        for name, count in top.items():
            log(f"  {name}: {count:,}")

        has_pixels = df["total_pixels"] > 0
        log(f"\nMean pct_developed:     {df.loc[has_pixels, 'pct_developed'].mean():.3f}")
        log(f"Mean pct_forest:        {df.loc[has_pixels, 'pct_forest'].mean():.3f}")
        log(f"Mean pct_agriculture:   {df.loc[has_pixels, 'pct_agriculture'].mean():.3f}")
        wetland_exclusion = (df["pct_wetland"] > 0.5).sum()
        log(f"Cells with pct_wetland > 0.5: {wetland_exclusion:,}")

    if output_path.exists():
        log(f"\nOutput: {output_path} ({_format_bytes(output_path.stat().st_size)})")


def compute_nlcd_features(
    raster_path: Path | None = None,
    grid_path: Path | None = None,
    output_path: Path | None = None,
    chunk_size: int = CHUNK_SIZE,
    from_checkpoint: bool = False,
    stats_cache: Path | None = None,
) -> pd.DataFrame:
    raster_path = raster_path or find_nlcd_raster(RAW_DIR)
    resolution = DEFAULT_RESOLUTION
    if grid_path is None:
        grid_path = PROCESSED_DIR / f"h3_grid_res{resolution}.parquet"
    if output_path is None:
        output_path = PROCESSED_DIR / f"features_nlcd_res{resolution}.parquet"
    if stats_cache is None:
        stats_cache = stats_cache_path(resolution)

    if not raster_path.exists():
        raise FileNotFoundError(f"NLCD raster not found: {raster_path}")

    log(f"Using NLCD raster: {raster_path.name}")

    stat_dicts: list[dict | None] | None = None
    h3_index: pd.Series
    nodata: int | float

    if from_checkpoint:
        if not stats_cache.exists():
            raise FileNotFoundError(
                f"Zonal stats cache not found: {stats_cache}. "
                "Re-run without --from-checkpoint."
            )
        cached_h3, stat_dicts, nodata = load_zonal_stats_cache(stats_cache)
        h3_index = pd.Series(cached_h3, name="h3_index")
    else:
        if not grid_path.exists():
            raise FileNotFoundError(
                f"H3 grid not found: {grid_path}. "
                f"Run: python scripts/build_grid.py --resolution {resolution}"
            )
        grid = load_grid(grid_path)
        nlcd_crs, nodata = open_nlcd_metadata(raster_path)
        log(f"NLCD nodata value: {nodata}")
        grid_reprojected = grid.to_crs(nlcd_crs)
        log(f"NLCD CRS: {nlcd_crs} | H3 grid reprojected to match")
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
        description="Compute NLCD land cover zonal statistics per H3 cell"
    )
    parser.add_argument(
        "--raster",
        type=Path,
        default=None,
        help="NLCD GeoTIFF path (auto-discovered under data/raw/ if omitted)",
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
        default=PROCESSED_DIR / f"features_nlcd_res{DEFAULT_RESOLUTION}.parquet",
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
        help="Skip zonal stats; rebuild features from nlcd_zonal_stats_res6.pkl cache",
    )
    parser.add_argument(
        "--stats-cache",
        type=Path,
        default=PROCESSED_DIR / f"nlcd_zonal_stats_res{DEFAULT_RESOLUTION}.pkl",
        help="Pickle cache for raw zonal stat dicts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        compute_nlcd_features(
            raster_path=args.raster,
            grid_path=args.grid,
            output_path=args.output,
            chunk_size=args.chunk_size,
            from_checkpoint=args.from_checkpoint,
            stats_cache=args.stats_cache,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        log(f"Error: {exc}")
        return 1
    # GDAL/rasterio teardown can hang for minutes on exit after long zonal_stats runs.
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
