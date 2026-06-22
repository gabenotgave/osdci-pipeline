#!/usr/bin/env python3
"""Compute PRISM 30-year normal climate features per H3 cell."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_RESOLUTION = 6
PRISM_DIR = RAW_DIR / "prism_normals"
TRACT_MAPPING = PROCESSED_DIR / f"h3_tract_mapping_res{DEFAULT_RESOLUTION}.parquet"

CDD_BASE_C = 18.3
SOURCE = "PRISM_30yr_normals_1991_2020_800m"

CONUS_MIN_LAT = 24.0
CONUS_MAX_LAT = 50.0
CONUS_MIN_LON = -125.0
CONUS_MAX_LON = -66.0

DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

STATE_FIPS_TO_ABBR = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT",
    "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL",
    "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD",
    "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT", "31": "NE",
    "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV",
    "55": "WI", "56": "WY",
}

OUTPUT_COLUMNS = [
    "h3_index",
    "mean_temp_c",
    "annual_cdd",
    "climate_zone",
    "cooling_favorable",
    "source",
]

MONTHLY_TIF_RE = re.compile(r"2020(0[1-9]|1[0-2])_avg_30y")


def log(message: str) -> None:
    print(message, flush=True)


def _format_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.1f} GB"


def write_parquet_safe(df: pd.DataFrame, path: Path) -> None:
    """Write parquet in a subprocess to avoid rasterio/pyarrow shutdown hangs."""
    import pickle
    import subprocess
    import sys
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


def read_grid_parquet_safe(grid_path: Path) -> pd.DataFrame:
    with grid_path.open("rb") as handle:
        grid = pd.read_parquet(handle)
    if grid.index.name == "h3_index":
        grid = grid.reset_index()
    return grid[["h3_index", "lat", "lon"]].copy()


def discover_prism_rasters(prism_dir: Path) -> tuple[Path, dict[int, Path]]:
    paths = sorted(prism_dir.glob("**/prism_tmean*.tif"))
    if not paths:
        raise FileNotFoundError(
            f"No PRISM rasters found under {prism_dir}. "
            "Expected pattern: **/prism_tmean*.tif"
        )

    annual_path: Path | None = None
    monthly_paths: dict[int, Path] = {}

    for path in paths:
        name = path.name
        monthly_match = MONTHLY_TIF_RE.search(name)
        if monthly_match:
            month = int(monthly_match.group(1))
            monthly_paths[month] = path
            continue
        if "2020_avg_30y" in name and not MONTHLY_TIF_RE.search(name):
            if annual_path is not None and annual_path != path:
                log(f"  Warning: multiple annual PRISM files; using {path.name}")
            annual_path = path

    if annual_path is None:
        raise FileNotFoundError("No annual PRISM raster found (expected *2020_avg_30y*.tif)")

    missing_months = [m for m in range(1, 13) if m not in monthly_paths]
    if missing_months:
        raise FileNotFoundError(f"Missing monthly PRISM rasters for months: {missing_months}")

    log(f"Found annual raster: {annual_path.name}")
    log(f"Found {len(monthly_paths)} monthly rasters")
    return annual_path, monthly_paths


def sample_raster_at_points(
    raster_path: Path,
    lons: np.ndarray,
    lats: np.ndarray,
    nodata: float | None = None,
) -> np.ndarray:
    import rasterio
    from rasterio.transform import rowcol

    n = len(lons)
    values = np.full(n, np.nan, dtype=np.float64)

    with rasterio.open(raster_path) as src:
        if nodata is None:
            nodata = src.nodata
        rows, cols = rowcol(src.transform, lons, lats)
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        band = src.read(1)
        height, width = band.shape
        valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        if valid.any():
            sampled = band[rows[valid], cols[valid]].astype(np.float64)
            if nodata is not None:
                sampled = np.where(sampled == nodata, np.nan, sampled)
            values[valid] = sampled
    return values


def compute_annual_cdd(monthly_temps: np.ndarray) -> np.ndarray:
    """Annual CDD from monthly mean temps (approximation; not daily data).

    Uses base 18.3 C (65 F). If any month is null, annual CDD is null.
    """
    n_cells = monthly_temps.shape[0]
    annual = np.full(n_cells, np.nan, dtype=np.float64)
    all_valid = ~np.isnan(monthly_temps).any(axis=1)
    if not all_valid.any():
        return annual

    days = np.array(DAYS_PER_MONTH, dtype=np.float64)
    excess = np.maximum(0.0, monthly_temps[all_valid] - CDD_BASE_C)
    annual[all_valid] = (excess * days).sum(axis=1)
    return annual


def classify_climate_zones(temps: np.ndarray) -> np.ndarray:
    zones = np.full(len(temps), None, dtype=object)
    valid = ~np.isnan(temps)
    t = temps[valid]
    idx = np.where(valid)[0]
    zones[idx[t < 5.0]] = "Cold"
    zones[idx[(t >= 5.0) & (t < 12.0)]] = "Cool"
    zones[idx[(t >= 12.0) & (t < 18.0)]] = "Moderate"
    zones[idx[(t >= 18.0) & (t < 24.0)]] = "Warm"
    zones[idx[t >= 24.0]] = "Hot"
    return zones


def load_state_lookup() -> pd.DataFrame | None:
    if not TRACT_MAPPING.exists():
        return None
    mapping = pd.read_parquet(TRACT_MAPPING, columns=["h3_index", "tract_fips"])
    mapping["state"] = mapping["tract_fips"].astype(str).str[:2].map(STATE_FIPS_TO_ABBR)
    return mapping[["h3_index", "state"]]


def compute_climate_features(
    prism_dir: Path | None = None,
    grid_path: Path | None = None,
    output_path: Path | None = None,
    sample: int | None = None,
) -> pd.DataFrame:
    prism_dir = prism_dir or PRISM_DIR
    resolution = DEFAULT_RESOLUTION
    if grid_path is None:
        grid_path = PROCESSED_DIR / f"h3_grid_res{resolution}.parquet"
    if output_path is None:
        output_path = PROCESSED_DIR / f"features_climate_res{resolution}.parquet"

    if not grid_path.exists():
        raise FileNotFoundError(
            f"H3 grid not found: {grid_path}. "
            f"Run: python scripts/build_grid.py --resolution {resolution}"
        )

    log(f"Loading H3 grid from {grid_path.name}...")
    grid = read_grid_parquet_safe(grid_path)
    if sample is not None:
        grid = grid.head(sample).copy()
        log(f"  Sample mode: using first {len(grid):,} cells")
    else:
        log(f"  {len(grid):,} hex cells")

    lons = grid["lon"].to_numpy(dtype=np.float64)
    lats = grid["lat"].to_numpy(dtype=np.float64)
    n_cells = len(grid)

    annual_path, monthly_paths = discover_prism_rasters(prism_dir)

    import rasterio

    with rasterio.open(annual_path) as src:
        log(f"Annual PRISM CRS: {src.crs}")
        log(f"Annual PRISM nodata: {src.nodata}")
        nodata = src.nodata

    log("Sampling annual mean temperature...")
    mean_temp_c = sample_raster_at_points(annual_path, lons, lats, nodata=nodata)

    monthly_temps = np.full((n_cells, 12), np.nan, dtype=np.float64)
    for month in range(1, 13):
        monthly_temps[:, month - 1] = sample_raster_at_points(
            monthly_paths[month], lons, lats, nodata=nodata
        )

    log(f"Sampled {n_cells:,} cells across 12 monthly rasters")

    annual_cdd = compute_annual_cdd(monthly_temps)
    climate_zones = classify_climate_zones(mean_temp_c)
    cooling_favorable = np.where(np.isnan(mean_temp_c), 0, (mean_temp_c < 12.0).astype(int))

    import gc
    gc.collect()

    result = pd.DataFrame(
        {
            "h3_index": grid["h3_index"],
            "mean_temp_c": mean_temp_c,
            "annual_cdd": annual_cdd,
            "climate_zone": climate_zones,
            "cooling_favorable": cooling_favorable,
            "source": SOURCE,
        }
    )

    state_lookup = load_state_lookup()
    print_summary(result, grid, state_lookup, output_path=None)

    log(f"Writing {output_path.name}...")
    write_parquet_safe(result[OUTPUT_COLUMNS], output_path)
    if output_path.exists():
        log(f"Output: {output_path} ({_format_bytes(output_path.stat().st_size)})")
    os._exit(0)



def describe_series(series: pd.Series) -> str:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return "no data"
    return (
        f"min={clean.min():.2f} median={clean.median():.2f} "
        f"mean={clean.mean():.2f} max={clean.max():.2f}"
    )


def print_summary(
    result: pd.DataFrame,
    grid: pd.DataFrame,
    state_lookup: pd.DataFrame | None,
    output_path: Path,
) -> None:
    has_temp = result["mean_temp_c"].notna()
    coverage_pct = 100.0 * has_temp.sum() / len(result)

    conus = (
        (grid["lat"] >= CONUS_MIN_LAT)
        & (grid["lat"] <= CONUS_MAX_LAT)
        & (grid["lon"] >= CONUS_MIN_LON)
        & (grid["lon"] <= CONUS_MAX_LON)
    )
    conus_df = result.loc[conus]

    log("Printing summary...")
    log("\n" + "=" * 60)
    log("PRISM climate features complete")
    log("=" * 60)
    log(
        f"Cells with mean_temp_c data:        {has_temp.sum():,} "
        f"({coverage_pct:.2f}%)"
    )
    log(f"mean_temp_c: {describe_series(result['mean_temp_c'])}")
    log(f"annual_cdd:  {describe_series(result['annual_cdd'])}")

    log("\nClimate zone breakdown:")
    for zone, count in result.loc[has_temp, "climate_zone"].value_counts().items():
        log(f"  {zone}: {count:,}")

    cool_count = int(conus_df["cooling_favorable"].sum())
    cool_pct = 100.0 * cool_count / len(conus_df) if len(conus_df) else 0.0
    log(f"\ncooling_favorable (CONUS):          {cool_count:,} ({cool_pct:.2f}%)")

    if state_lookup is not None and has_temp.any():
        with_state = result.loc[has_temp].merge(state_lookup, on="h3_index", how="left")
        with_state = with_state.dropna(subset=["state"])
        if not with_state.empty:
            log("\nTop 5 coldest states by mean mean_temp_c:")
            for state, temp in (
                with_state.groupby("state")["mean_temp_c"]
                .mean()
                .sort_values()
                .head(5)
                .items()
            ):
                log(f"  {state}: {temp:.2f} C")

            log("\nTop 5 hottest states by mean mean_temp_c:")
            for state, temp in (
                with_state.groupby("state")["mean_temp_c"]
                .mean()
                .sort_values(ascending=False)
                .head(5)
                .items()
            ):
                log(f"  {state}: {temp:.2f} C")
    else:
        log("\nState breakdown skipped (h3_tract_mapping_res6.parquet not found)")

    if output_path is not None and output_path.exists():
        log(f"\nOutput: {output_path} ({_format_bytes(output_path.stat().st_size)})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PRISM 30-year normal climate features per H3 cell"
    )
    parser.add_argument(
        "--prism-dir",
        type=Path,
        default=PRISM_DIR,
        help="Directory containing PRISM normal GeoTIFFs",
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
        default=PROCESSED_DIR / f"features_climate_res{DEFAULT_RESOLUTION}.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N grid cells (for testing)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        compute_climate_features(
            prism_dir=args.prism_dir,
            grid_path=args.grid,
            output_path=args.output,
            sample=args.sample,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        log(f"Error: {exc}")
        return 1
    os._exit(0)  # avoid GDAL/rasterio shutdown hang


if __name__ == "__main__":
    sys.exit(main())
