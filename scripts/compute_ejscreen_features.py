#!/usr/bin/env python3
"""Compute EPA EJScreen 2024 environmental justice features per H3 cell."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pygris

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_RESOLUTION = 6
DEFAULT_EJSCREEN_CSV = RAW_DIR / "EJSCREEN_2024_BG_with_AS_CNMI_GU_VI.csv"
BLOCK_GROUPS_CACHE = RAW_DIR / "block_groups_2020_cb.parquet"
MAPPING_CACHE = RAW_DIR / "h3_bg_mapping_res6.parquet"
CENSUS_BG_YEAR = 2020
WORK_CRS = "EPSG:5070"
SOURCE = "EJScreen_2024_v2.31"
DEMOG_INDEX_MAX = 4.0
EJ_BURDEN_THRESHOLD = 2.0
DEMOG_INDEX_MAX = 4.0
EJ_BURDEN_THRESHOLD = 2.0

TERRITORIES_EXCLUDE = {
    "American Samoa",
    "Commonwealth of the Northern Mariana Islands",
    "Guam",
    "United States Virgin Islands",
}

STATE_FIPS = [
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13", "15", "16",
    "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
    "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42",
    "44", "45", "46", "47", "48", "49", "50", "51", "53", "54", "55", "56", "72",
]

EJSCREEN_USECOLS = [
    "ID",
    "STATE_NAME",
    "ST_ABBREV",
    "ACSTOTPOP",
    "PM25",
    "OZONE",
    "DSLPM",
    "RSEI_AIR",
    "PTRAF",
    "PNPL",
    "PRMP",
    "P_PM25",
    "P_OZONE",
    "P_DSLPM",
    "P_RSEI_AIR",
    "P_PTRAF",
    "P_PNPL",
    "P_PRMP",
    "DEMOGIDX_2",
    "LOWINCPCT",
    "PEOPCOLORPCT",
    "T_PM25",
    "T_PTRAF",
]

KEY_NUMERIC_COLS = [
    "PM25",
    "P_PM25",
    "P_PTRAF",
    "DEMOGIDX_2",
    "LOWINCPCT",
    "PEOPCOLORPCT",
    "P_RSEI_AIR",
    "RSEI_AIR",
]

OUTPUT_COLUMNS = [
    "h3_index",
    "pm25_raw",
    "pm25_pctile",
    "ozone_pctile",
    "diesel_pm_pctile",
    "cancer_risk_pctile",
    "resp_risk_pctile",
    "traffic_pctile",
    "superfund_pctile",
    "rmp_pctile",
    "demographic_index",
    "demographic_index_norm",
    "pct_low_income",
    "pct_minority",
    "ej_pm25_pctile",
    "ej_traffic_pctile",
    "high_pm25",
    "high_traffic",
    "high_cancer_risk",
    "high_ej_burden",
    "block_group_fips",
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
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    with path.open("wb") as handle:
        pq.write_table(table, handle)


def write_gdf_parquet_safe(gdf: gpd.GeoDataFrame, path: Path) -> None:
    from geopandas.io.arrow import _to_parquet

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        _to_parquet(gdf, handle)


def read_grid_parquet_safe(grid_path: Path) -> gpd.GeoDataFrame:
    with grid_path.open("rb") as handle:
        return gpd.read_parquet(handle)


def read_gdf_parquet_safe(path: Path) -> gpd.GeoDataFrame:
    with path.open("rb") as handle:
        return gpd.read_parquet(handle)


def parse_pctile_value(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else np.nan


def load_ejscreen_csv(csv_path: Path) -> pd.DataFrame:
    log(f"Loading EJScreen CSV from {csv_path.name}...")
    available = pd.read_csv(csv_path, nrows=0).columns.tolist()
    usecols = [c for c in EJSCREEN_USECOLS if c in available]
    missing = sorted(set(EJSCREEN_USECOLS) - set(usecols))
    if missing:
        log(f"  Warning: missing expected columns: {', '.join(missing)}")

    ej_cols = [c for c in available if "EJ" in c.upper() or "PCTILE" in c.upper()]
    if ej_cols:
        log(f"  Columns containing EJ/PCTILE: {', '.join(ej_cols)}")
    else:
        log(
            "  Note: no EJ_PCTILE_* columns; using T_PM25/T_PTRAF for combined EJ percentiles"
        )

    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    log(f"  Shape: {len(df):,} rows x {len(df.columns)} columns")
    log(f"  First 3 columns: {', '.join(df.columns[:3].tolist())}")

    df["ID"] = df["ID"].astype(str).str.zfill(12)

    present_keys = [c for c in KEY_NUMERIC_COLS if c in df.columns]
    if present_keys:
        log("\n  Key column describe():")
        log(df[present_keys].describe().to_string())

    log("\n  STATE_NAME value counts (top 10):")
    for state, count in df["STATE_NAME"].value_counts().head(10).items():
        log(f"    {state}: {count:,}")

    before = len(df)
    df = df[~df["STATE_NAME"].isin(TERRITORIES_EXCLUDE)].copy()
    log(f"\n  After territory filter: {len(df):,} rows (removed {before - len(df):,})")
    return df


def download_block_groups(
    cache_path: Path,
    year: int = CENSUS_BG_YEAR,
    force: bool = False,
) -> gpd.GeoDataFrame:
    if cache_path.exists() and not force:
        log(f"Loading block groups from {cache_path.name}...")
        bg = read_gdf_parquet_safe(cache_path)
        if bg.crs is None:
            bg = bg.set_crs("EPSG:4326")
        log(f"  {len(bg):,} block groups (cached)")
        return bg

    log(f"Downloading Census block groups for year {year} ({len(STATE_FIPS)} states)...")
    frames: list[gpd.GeoDataFrame] = []
    failed: list[str] = []

    for i, fips in enumerate(STATE_FIPS, start=1):
        try:
            state_bg = pygris.block_groups(state=fips, year=year, cb=True, cache=True)
            if state_bg is not None and len(state_bg):
                frames.append(state_bg[["GEOID", "STATEFP", "geometry"]])
        except Exception as exc:
            failed.append(fips)
            log(f"  Warning: skipping state FIPS {fips}: {exc}")
        if i % 10 == 0:
            log(f"  Downloaded block groups for {i}/{len(STATE_FIPS)} states...")

    if not frames:
        raise ValueError("No Census block groups downloaded")

    bg = pd.concat(frames, ignore_index=True)
    bg = gpd.GeoDataFrame(bg, geometry="geometry", crs=frames[0].crs)
    bg["GEOID"] = bg["GEOID"].astype(str).str.zfill(12)
    write_gdf_parquet_safe(bg, cache_path)
    log(f"  Saved {len(bg):,} block groups to {cache_path.name}")
    if failed:
        log(f"  Warning: {len(failed)} states skipped: {', '.join(failed)}")
    return bg


def load_grid_centroids(grid_path: Path, sample: int | None) -> gpd.GeoDataFrame:
    log(f"Loading H3 grid from {grid_path.name}...")
    grid = read_grid_parquet_safe(grid_path)
    if grid.index.name == "h3_index":
        grid = grid.reset_index()
    if sample is not None:
        grid = grid.head(sample).copy()
        log(f"  Sample mode: using first {len(grid):,} cells")
    else:
        log(f"  {len(grid):,} hex cells")

    points = gpd.GeoDataFrame(
        grid[["h3_index", "lat", "lon"]],
        geometry=gpd.points_from_xy(grid["lon"], grid["lat"]),
        crs="EPSG:4326",
    )
    return points.to_crs(WORK_CRS)


def build_h3_bg_mapping(
    grid_points: gpd.GeoDataFrame,
    block_groups: gpd.GeoDataFrame,
) -> pd.DataFrame:
    bg_gdf = block_groups.copy()
    if bg_gdf.crs != WORK_CRS:
        bg_gdf = bg_gdf.to_crs(WORK_CRS)

    log("Spatial join: H3 centroids within block group polygons...")
    joined = gpd.sjoin(
        grid_points[["h3_index", "lat", "lon", "geometry"]],
        bg_gdf[["GEOID", "geometry"]],
        how="left",
        predicate="within",
    )
    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])

    dupes = joined["h3_index"].duplicated(keep=False).sum()
    if dupes:
        joined = joined.sort_values("h3_index").drop_duplicates("h3_index", keep="first")
        log(f"  Warning: resolved {dupes:,} duplicate centroid matches")

    mapping = joined[["h3_index", "lat", "lon", "GEOID"]].rename(columns={"GEOID": "block_group_fips"})
    matched = mapping["block_group_fips"].notna().sum()
    log(f"  {matched:,} / {len(mapping):,} cells matched to a block group")
    return mapping


def load_or_build_h3_bg_mapping(
    grid_points: gpd.GeoDataFrame,
    block_groups: gpd.GeoDataFrame,
    mapping_path: Path,
    force_join: bool,
    from_checkpoint: bool,
) -> pd.DataFrame:
    expected_cells = len(grid_points)
    if mapping_path.exists() and not force_join:
        log(f"Loading H3→block group mapping from {mapping_path.name}...")
        mapping = pd.read_parquet(mapping_path)
        if len(mapping) == expected_cells:
            log(f"  {len(mapping):,} cells (cached mapping)")
            return mapping
        log(
            f"  Warning: mapping has {len(mapping):,} rows but grid has "
            f"{expected_cells:,}; rebuilding spatial join"
        )

    mapping = build_h3_bg_mapping(grid_points, block_groups)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_safe(mapping, mapping_path)
    log(f"  Saved mapping to {mapping_path.name}")
    return mapping


def build_features(mapping: pd.DataFrame, ejscreen_df: pd.DataFrame) -> pd.DataFrame:
    attrs = ejscreen_df.copy()
    attrs["ej_pm25_pctile"] = attrs["T_PM25"].map(parse_pctile_value) if "T_PM25" in attrs else np.nan
    attrs["ej_traffic_pctile"] = (
        attrs["T_PTRAF"].map(parse_pctile_value) if "T_PTRAF" in attrs else np.nan
    )

    merged = mapping.merge(
        attrs,
        left_on="block_group_fips",
        right_on="ID",
        how="left",
    )

    pm25_pctile = pd.to_numeric(merged.get("P_PM25"), errors="coerce")
    traffic_pctile = pd.to_numeric(merged.get("P_PTRAF"), errors="coerce")
    cancer_pctile = pd.to_numeric(merged.get("P_RSEI_AIR"), errors="coerce")
    demog = pd.to_numeric(merged.get("DEMOGIDX_2"), errors="coerce")

    result = pd.DataFrame(
        {
            "h3_index": merged["h3_index"],
            "pm25_raw": pd.to_numeric(merged.get("PM25"), errors="coerce"),
            "pm25_pctile": pm25_pctile,
            "ozone_pctile": pd.to_numeric(merged.get("P_OZONE"), errors="coerce"),
            "diesel_pm_pctile": pd.to_numeric(merged.get("P_DSLPM"), errors="coerce"),
            "cancer_risk_pctile": cancer_pctile,
            "resp_risk_pctile": np.nan,
            "traffic_pctile": traffic_pctile,
            "superfund_pctile": pd.to_numeric(merged.get("P_PNPL"), errors="coerce"),
            "rmp_pctile": pd.to_numeric(merged.get("P_PRMP"), errors="coerce"),
            "demographic_index": demog,
            "pct_low_income": pd.to_numeric(merged.get("LOWINCPCT"), errors="coerce"),
            "pct_minority": pd.to_numeric(merged.get("PEOPCOLORPCT"), errors="coerce"),
            "ej_pm25_pctile": pd.to_numeric(merged.get("ej_pm25_pctile"), errors="coerce"),
            "ej_traffic_pctile": pd.to_numeric(merged.get("ej_traffic_pctile"), errors="coerce"),
            "block_group_fips": merged["block_group_fips"],
            "source": SOURCE,
            "_state": merged.get("ST_ABBREV"),
        }
    )

    result["demographic_index_norm"] = (demog / DEMOG_INDEX_MAX).astype(np.float64)
    result["high_pm25"] = (pm25_pctile >= 80).fillna(False).astype(int)
    result["high_traffic"] = (traffic_pctile >= 80).fillna(False).astype(int)
    result["high_cancer_risk"] = (cancer_pctile >= 80).fillna(False).astype(int)
    result["high_ej_burden"] = (demog >= EJ_BURDEN_THRESHOLD).fillna(False).astype(int)
    return result


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
    ejscreen_df: pd.DataFrame,
    bg_matched: int,
    output_path: Path,
) -> None:
    has_data = result["pm25_pctile"].notna()
    coverage_pct = 100.0 * has_data.sum() / len(result) if len(result) else 0.0

    log("\n" + "=" * 60)
    log("EJScreen features complete")
    log("=" * 60)
    log(f"Block groups loaded from CSV:       {len(ejscreen_df):,}")
    log(f"Block groups matched to geometry:   {bg_matched:,}")
    log(
        f"H3 cells with EJScreen data:        {has_data.sum():,} "
        f"({coverage_pct:.2f}%)"
    )
    log(f"pm25_pctile: {describe_series(result['pm25_pctile'])}")
    log(f"traffic_pctile: {describe_series(result['traffic_pctile'])}")
    log(f"demographic_index: {describe_series(result['demographic_index'])}")
    log(f"demographic_index_norm: {describe_series(result['demographic_index_norm'])}")

    data = result.loc[has_data]
    for col in ("high_pm25", "high_traffic", "high_ej_burden"):
        count = int(data[col].sum())
        pct = 100.0 * count / len(data) if len(data) else 0.0
        log(f"{col}: {count:,} ({pct:.2f}% of cells with data)")

    if "_state" in result.columns and has_data.any():
        with_state = result.loc[has_data].dropna(subset=["_state"])
        if not with_state.empty:
            log("\nTop 5 states by mean pm25_pctile:")
            for state, score in (
                with_state.groupby("_state")["pm25_pctile"]
                .mean()
                .sort_values(ascending=False)
                .head(5)
                .items()
            ):
                log(f"  {state}: {score:.2f}")
            log("\nTop 5 states by mean demographic_index:")
            for state, score in (
                with_state.groupby("_state")["demographic_index"]
                .mean()
                .sort_values(ascending=False)
                .head(5)
                .items()
            ):
                log(f"  {state}: {score:.4f}")

    if output_path.exists():
        log(f"\nOutput: {output_path} ({_format_bytes(output_path.stat().st_size)})")


def compute_ejscreen_features(
    csv_path: Path | None = None,
    grid_path: Path | None = None,
    output_path: Path | None = None,
    resolution: int = DEFAULT_RESOLUTION,
    bg_cache: Path | None = None,
    mapping_path: Path | None = None,
    sample: int | None = None,
    force_join: bool = False,
    from_checkpoint: bool = False,
    force_bg: bool = False,
) -> pd.DataFrame:
    csv_path = csv_path or DEFAULT_EJSCREEN_CSV
    if grid_path is None:
        grid_path = PROCESSED_DIR / f"h3_grid_res{resolution}.parquet"
    if output_path is None:
        output_path = PROCESSED_DIR / f"features_ejscreen_res{resolution}.parquet"
    if bg_cache is None:
        bg_cache = BLOCK_GROUPS_CACHE
    if mapping_path is None:
        mapping_path = MAPPING_CACHE

    if not csv_path.exists():
        raise FileNotFoundError(f"EJScreen CSV not found: {csv_path}")
    if not grid_path.exists():
        raise FileNotFoundError(
            f"H3 grid not found: {grid_path}. "
            f"Run: python scripts/build_grid.py --resolution {resolution}"
        )

    csv_rows_loaded = len(
        pd.read_csv(csv_path, usecols=["ID"], low_memory=False)
    )

    ejscreen_df = load_ejscreen_csv(csv_path)
    block_groups = download_block_groups(bg_cache, force=force_bg)

    ej_ids = set(ejscreen_df["ID"])
    bg_ids = set(block_groups["GEOID"].astype(str))
    bg_matched = len(ej_ids & bg_ids)
    log(f"  EJScreen IDs with block group geometry: {bg_matched:,}")

    grid_points = load_grid_centroids(grid_path, sample)
    mapping = load_or_build_h3_bg_mapping(
        grid_points,
        block_groups,
        mapping_path,
        force_join=force_join,
        from_checkpoint=from_checkpoint,
    )

    features = build_features(mapping, ejscreen_df)
    features_out = features[OUTPUT_COLUMNS]

    log(f"\nWriting {output_path.name}...")
    write_parquet_safe(features_out, output_path)

    print_summary(
        features,
        ejscreen_df,
        bg_matched,
        output_path,
    )
    log(f"  (CSV raw row count before filter: {csv_rows_loaded:,})")
    os._exit(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute EPA EJScreen 2024 features per H3 cell"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_EJSCREEN_CSV,
        help="EJScreen block group CSV path",
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
        default=PROCESSED_DIR / f"features_ejscreen_res{DEFAULT_RESOLUTION}.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help=f"H3 resolution (default: {DEFAULT_RESOLUTION})",
    )
    parser.add_argument(
        "--bg-cache",
        type=Path,
        default=BLOCK_GROUPS_CACHE,
        help="Cached Census block groups parquet",
    )
    parser.add_argument(
        "--mapping-cache",
        type=Path,
        default=MAPPING_CACHE,
        help="Cached H3→block group mapping parquet",
    )
    parser.add_argument(
        "--from-checkpoint",
        action="store_true",
        help="Use cached H3→block group mapping when available",
    )
    parser.add_argument(
        "--force-join",
        action="store_true",
        help="Recompute spatial join and refresh mapping cache",
    )
    parser.add_argument(
        "--force-bg",
        action="store_true",
        help="Re-download Census block groups",
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
        compute_ejscreen_features(
            csv_path=args.csv,
            grid_path=args.grid,
            output_path=args.output,
            resolution=args.resolution,
            bg_cache=args.bg_cache,
            mapping_path=args.mapping_cache,
            sample=args.sample,
            force_join=args.force_join,
            from_checkpoint=args.from_checkpoint,
            force_bg=args.force_bg,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        log(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
