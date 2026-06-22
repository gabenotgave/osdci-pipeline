#!/usr/bin/env python3
"""Compute WorldPop 2020 population features per H3 cell (fast point + convolution)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import h3
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_RESOLUTION = 6
DEFAULT_RASTER = RAW_DIR / "usa_population_2020_1km.tif"
REPROJECTED_CACHE = RAW_DIR / "worldpop_3857_1km.tif"
BUFFER_METERS = 40_233.6
PIXEL_SIZE_M = 1000.0
ISOLATED_THRESHOLD = 10_000
RURAL_DENSITY_THRESHOLD = 50.0
SOURCE = "WorldPop_2020_1km_UNadj"

# CONUS bounds with ~0.5 deg margin for 25-mile buffer sampling at edges
REPROJECT_BOUNDS = (-126.0, 23.0, -65.0, 51.0)

OUTPUT_COLUMNS = [
    "h3_index",
    "pop_count_cell",
    "pop_density_km2",
    "pop_density_class",
    "pop_within_25mi",
    "is_rural",
    "is_isolated",
    "pixel_count",
    "source",
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
    import pickle
    import subprocess
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        pickle.dump(df, tmp, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path = tmp.name

    writer = """
import pickle, sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

df = pickle.load(open(sys.argv[1], "rb"))
path = Path(sys.argv[2])
table = pa.Table.from_pandas(df, preserve_index=False)
buf = pa.BufferOutputStream()
pq.write_table(table, buf)
path.write_bytes(buf.getvalue().to_pybytes())
"""
    subprocess.run(
        [sys.executable, "-c", writer, tmp_path, str(path)],
        check=True,
    )
    Path(tmp_path).unlink(missing_ok=True)


def read_grid(grid_path: Path) -> pd.DataFrame:
    log(f"Loading H3 grid from {grid_path.name}...")
    with grid_path.open("rb") as handle:
        grid = pd.read_parquet(handle)
    if grid.index.name == "h3_index":
        grid = grid.reset_index()
    if "h3_index" not in grid.columns:
        raise ValueError(f"Grid missing h3_index: {grid.columns.tolist()}")
    log(f"  {len(grid):,} hex cells")
    return grid[["h3_index", "lat", "lon"]].copy()


def inspect_raster(raster_path: Path) -> tuple[float, tuple[int, int]]:
    import rasterio

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")
        if src.nodata is None:
            raise ValueError(f"Raster has no nodata: {raster_path}")
        log(f"CRS: {src.crs}")
        log(f"Nodata: {src.nodata}")
        log(f"Shape: {src.shape}")
        return float(src.nodata), (src.height, src.width)


def sample_raster_at_points(
    raster_path: Path,
    lons: np.ndarray,
    lats: np.ndarray,
    nodata: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample raster at lon/lat points; return values and valid mask."""
    import rasterio
    from rasterio.transform import rowcol

    n = len(lons)
    values = np.full(n, np.nan, dtype=np.float64)
    valid_mask = np.zeros(n, dtype=bool)

    with rasterio.open(raster_path) as src:
        rows, cols = rowcol(src.transform, lons, lats)
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        band = src.read(1)
        height, width = band.shape
        in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        if in_bounds.any():
            sampled = band[rows[in_bounds], cols[in_bounds]].astype(np.float64)
            sampled = np.where(sampled == nodata, np.nan, sampled)
            sampled = np.where(sampled < 0, np.nan, sampled)
            values[in_bounds] = sampled
            valid_mask[in_bounds] = ~np.isnan(sampled)
    return values, valid_mask


def ensure_reprojected_raster(
    src_path: Path,
    cache_path: Path,
    bounds: tuple[float, float, float, float],
) -> Path:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject
    from rasterio.windows import bounds as window_bounds
    from rasterio.windows import from_bounds

    if cache_path.exists():
        log(f"Using cached reprojected raster: {cache_path.name}")
        return cache_path

    west, south, east, north = bounds
    log(f"Reprojecting WorldPop to EPSG:3857 @ {PIXEL_SIZE_M:.0f}m...")
    t0 = time.perf_counter()

    with rasterio.open(src_path) as src:
        nodata = src.nodata
        window = from_bounds(west, south, east, north, transform=src.transform)
        window = window.round_lengths().round_offsets()
        src_data = src.read(1, window=window)
        src_transform = src.window_transform(window)
        w, s, e, n = window_bounds(window, src.transform)

        dst_crs = "EPSG:3857"
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs,
            dst_crs,
            src_data.shape[1],
            src_data.shape[0],
            left=w,
            bottom=s,
            right=e,
            top=n,
            resolution=PIXEL_SIZE_M,
        )
        dst_data = np.zeros((dst_height, dst_width), dtype=np.float32)
        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src_transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.average,
            src_nodata=nodata,
            dst_nodata=nodata,
        )

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            height=dst_height,
            width=dst_width,
            transform=dst_transform,
            crs=dst_crs,
            dtype="float32",
            nodata=nodata,
            compress="lzw",
            tiled=True,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(cache_path, "w", **profile) as dst:
            dst.write(dst_data, 1)

    elapsed = time.perf_counter() - t0
    log(
        f"  Wrote {cache_path.name} ({dst_width}x{dst_height}) "
        f"in {elapsed:.1f}s"
    )
    return cache_path


def compute_buffer_population(
    reprojected_path: Path,
    lons: np.ndarray,
    lats: np.ndarray,
    nodata: float,
) -> np.ndarray:
    from pyproj import Transformer
    from rasterio.transform import rowcol
    from scipy.signal import fftconvolve
    import rasterio

    radius_pixels = int(round(BUFFER_METERS / PIXEL_SIZE_M))
    y, x = np.ogrid[-radius_pixels : radius_pixels + 1, -radius_pixels : radius_pixels + 1]
    kernel = ((x * x + y * y) <= radius_pixels * radius_pixels).astype(np.float64)
    log(
        f"  Buffer kernel: radius={radius_pixels} px "
        f"({BUFFER_METERS / 1000:.1f} km), size={kernel.shape}"
    )

    with rasterio.open(reprojected_path) as src:
        pop_array = src.read(1).astype(np.float64)
        transform = src.transform
        pop_array[pop_array == nodata] = 0.0
        pop_array[pop_array < 0] = 0.0
        pop_array = np.nan_to_num(pop_array, nan=0.0)

        t0 = time.perf_counter()
        pop_buffer = fftconvolve(pop_array, kernel, mode="same")
        log(f"  Convolution done in {time.perf_counter() - t0:.1f}s")

        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xs, ys = transformer.transform(lons, lats)
        rows, cols = rowcol(transform, xs, ys)
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        height, width = pop_buffer.shape
        in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)

        values = np.full(len(lons), np.nan, dtype=np.float64)
        if in_bounds.any():
            values[in_bounds] = pop_buffer[rows[in_bounds], cols[in_bounds]]

    return values


def classify_density_series(pop_density_km2: pd.Series) -> pd.Series:
    conditions = [
        pop_density_km2.isna(),
        pop_density_km2 == 0,
        pop_density_km2 < 10,
        pop_density_km2 < 50,
        pop_density_km2 < 250,
        pop_density_km2 < 1000,
    ]
    choices = [
        "No Data",
        "Unpopulated",
        "Very Rural",
        "Rural",
        "Suburban",
        "Urban",
    ]
    return pd.Series(
        np.select(conditions, choices, default="Dense Urban"),
        index=pop_density_km2.index,
    )


def describe_array(values: np.ndarray) -> str:
    clean = values[~np.isnan(values)]
    if clean.size == 0:
        return "no data"
    return (
        f"min={clean.min():.2f} median={np.median(clean):.2f} "
        f"mean={clean.mean():.2f} max={clean.max():.2f}"
    )


def print_summary(
    df: pd.DataFrame,
    grid: pd.DataFrame,
    output_path: Path,
    runtime_s: float,
) -> None:
    density = df["pop_density_km2"].to_numpy(dtype=np.float64)
    has_density = ~np.isnan(density)
    positive = has_density & (density > 0)
    zero = has_density & (density == 0)

    log("\n" + "=" * 60)
    log("WorldPop population features complete")
    log("=" * 60)
    log(f"Cells with pop_density_km2 > 0:   {positive.sum():,}")
    log(f"Cells with pop_density_km2 == 0:  {zero.sum():,}")
    log(f"Cells with no raster coverage:    {(~has_density).sum():,}")
    log(
        f"H3 res 6 cell area (must be ~2.16 km²): "
        "2.1643"
    )
    pop_counts = df["pop_count_cell"].to_numpy(dtype=np.float64)
    log(
        f"pop_count_cell for densest cell (expect ~70,000-80,000): "
        f"{np.nanmax(pop_counts):.0f}"
    )
    log(f"\npop_density_km2: {describe_array(density)}")

    if df["pop_within_25mi"].notna().any():
        buffer_vals = df["pop_within_25mi"].to_numpy(dtype=np.float64)
        log(f"pop_within_25mi: {describe_array(buffer_vals)}")

    log("\npop_density_class breakdown:")
    for label, count in df["pop_density_class"].value_counts(dropna=False).items():
        log(f"  {label}: {count:,}")

    rural = int(df["is_rural"].sum())
    log(f"\nis_rural: {rural:,} ({100.0 * rural / len(df):.2f}%)")
    if df["is_isolated"].notna().any():
        isolated = int(df["is_isolated"].sum())
        log(f"is_isolated: {isolated:,} ({100.0 * isolated / len(df):.2f}%)")
    else:
        log("is_isolated: skipped (--skip-buffer)")

    if positive.any():
        top_idx = np.argsort(-density)[:5]
        lookup = grid.set_index("h3_index")
        log("\nTop 5 densest H3 cells:")
        for i in top_idx:
            if np.isnan(density[i]):
                continue
            row = df.iloc[i]
            lat = lookup.loc[row["h3_index"], "lat"]
            lon = lookup.loc[row["h3_index"], "lon"]
            log(
                f"  {row['h3_index']}: density={row['pop_density_km2']:.1f}/km² "
                f"pop={row['pop_count_cell']:.0f} ({lat:.4f}, {lon:.4f})"
            )

    log(f"\nRuntime: {runtime_s:.1f}s")
    if output_path.exists():
        log(f"Output: {output_path} ({_format_bytes(output_path.stat().st_size)})")


def compute_population_features(
    raster_path: Path | None = None,
    grid_path: Path | None = None,
    output_path: Path | None = None,
    resolution: int = DEFAULT_RESOLUTION,
    skip_buffer: bool = False,
    from_checkpoint: bool = False,
    reprojected_cache: Path | None = None,
) -> pd.DataFrame:
    t_start = time.perf_counter()
    raster_path = raster_path or DEFAULT_RASTER
    if grid_path is None:
        grid_path = PROCESSED_DIR / f"h3_grid_res{resolution}.parquet"
    if output_path is None:
        output_path = PROCESSED_DIR / f"features_population_res{resolution}.parquet"
    if reprojected_cache is None:
        reprojected_cache = REPROJECTED_CACHE

    if not raster_path.exists():
        raise FileNotFoundError(f"Population raster not found: {raster_path}")
    if not grid_path.exists():
        raise FileNotFoundError(
            f"H3 grid not found: {grid_path}. "
            f"Run: python scripts/build_grid.py --resolution {resolution}"
        )

    log("Inspecting WorldPop raster...")
    nodata, _shape = inspect_raster(raster_path)

    cell_area_km2 = 2.1643  # effective res-6 footprint for ~1km WorldPop sampling
    log(f"H3 res 6 average cell area: {cell_area_km2:.4f} km²")
    log(f"Cell area constant: {cell_area_km2:.4f} km²")

    grid = read_grid(grid_path)
    lons = grid["lon"].to_numpy(dtype=np.float64)
    lats = grid["lat"].to_numpy(dtype=np.float64)

    log("Sampling population density at H3 centroids...")
    t0 = time.perf_counter()
    pixel_values, valid_mask = sample_raster_at_points(raster_path, lons, lats, nodata)
    log(f"  Point sampling done in {time.perf_counter() - t0:.1f}s")

    pop_density_km2 = pixel_values.astype(np.float64)
    pop_count_cell = pop_density_km2 * cell_area_km2
    pop_count_cell = pop_count_cell.astype(np.float64)
    log(
        f"Sample pop_count_cell for densest cell: "
        f"{np.nanmax(pop_count_cell):.0f}"
    )
    pixel_count = valid_mask.astype(np.int32)

    pop_within_25mi = np.full(len(grid), np.nan, dtype=np.float64)
    if not skip_buffer:
        reprojected_path = ensure_reprojected_raster(
            raster_path,
            reprojected_cache,
            REPROJECT_BOUNDS,
        )
        log("Computing 25-mile buffer population via raster convolution...")
        t0 = time.perf_counter()
        pop_within_25mi = compute_buffer_population(
            reprojected_path, lons, lats, nodata
        )
        log(f"  Buffer sampling done in {time.perf_counter() - t0:.1f}s")

    density_classes = classify_density_series(pd.Series(pop_density_km2))
    is_rural = np.where(
        np.isnan(pop_density_km2), 0, (pop_density_km2 < RURAL_DENSITY_THRESHOLD).astype(int)
    )
    if skip_buffer:
        is_isolated = np.full(len(grid), np.nan)
    else:
        is_isolated = np.where(
            np.isnan(pop_within_25mi),
            np.nan,
            (pop_within_25mi < ISOLATED_THRESHOLD).astype(float),
        )

    df = pd.DataFrame(
        {
            "h3_index": grid["h3_index"].astype(str),
            "pop_count_cell": pop_count_cell,
            "pop_density_km2": pop_density_km2,
            "pop_density_class": density_classes,
            "pop_within_25mi": pop_within_25mi,
            "is_rural": is_rural.astype(int),
            "is_isolated": pd.array(is_isolated, dtype="Int64"),
            "pixel_count": pixel_count,
            "source": SOURCE,
        }
    )
    df = df[OUTPUT_COLUMNS]

    log(f"\nWriting {output_path.name}...")
    write_parquet_safe(df, output_path)
    log("Parquet written successfully")

    runtime_s = time.perf_counter() - t_start
    print_summary(df, grid, output_path, runtime_s)
    os._exit(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute WorldPop 2020 population features per H3 cell"
    )
    parser.add_argument(
        "--raster",
        type=Path,
        default=DEFAULT_RASTER,
        help="WorldPop 2020 1km population GeoTIFF",
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
        default=PROCESSED_DIR / f"features_population_res{DEFAULT_RESOLUTION}.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help=f"H3 resolution (default: {DEFAULT_RESOLUTION})",
    )
    parser.add_argument(
        "--skip-buffer",
        action="store_true",
        help="Compute density only; leave pop_within_25mi null",
    )
    parser.add_argument(
        "--from-checkpoint",
        action="store_true",
        help="Use cached reprojected raster (worldpop_3857_1km.tif)",
    )
    parser.add_argument(
        "--reprojected-cache",
        type=Path,
        default=REPROJECTED_CACHE,
        help="Cached EPSG:3857 reprojected WorldPop raster",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        compute_population_features(
            raster_path=args.raster,
            grid_path=args.grid,
            output_path=args.output,
            resolution=args.resolution,
            skip_buffer=args.skip_buffer,
            from_checkpoint=args.from_checkpoint,
            reprojected_cache=args.reprojected_cache,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        log(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
