#!/usr/bin/env python3
"""Compute WRI Aqueduct 4.0 baseline water stress features per H3 cell."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_RESOLUTION = 6
DEFAULT_AQUEDUCT_CSV = RAW_DIR / "Aqueduct40_baseline_annual_y2023m07d05.csv"
HYBAS_URL = (
    "zip+https://data.hydrosheds.org/file/HydroBASINS/standard/"
    "hybas_na_lev06_v1c.zip!hybas_na_lev06_v1c.shp"
)
HYBAS_CACHE = RAW_DIR / "hydrobasins_na_lev06.parquet"
WORK_CRS = "EPSG:5070"
SENTINEL = -9999.0
SENTINEL_POS = 9999.0
SOURCE = "WRI_Aqueduct_4.0_2023"

CONUS_MIN_LAT = 24.0
CONUS_MAX_LAT = 50.0
CONUS_MIN_LON = -125.0
CONUS_MAX_LON = -66.0

TRACT_MAPPING = PROCESSED_DIR / f"h3_tract_mapping_res{DEFAULT_RESOLUTION}.parquet"

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

AQUEDUCT_COLS = [
    "string_id",
    "pfaf_id",
    "gid_0",
    "gid_1",
    "name_1",
    "bws_raw",
    "bws_cat",
    "bws_label",
    "bwd_raw",
    "iav_raw",
    "gtd_raw",
    "drr_raw",
    "drr_cat",
    "w_awr_def_tot_raw",
]

RAW_SCORE_COLS = [
    "bws_raw",
    "bwd_raw",
    "iav_raw",
    "gtd_raw",
    "drr_raw",
    "w_awr_def_tot_raw",
]

INSPECT_COLS = [
    "bws_raw",
    "bws_cat",
    "bws_label",
    "bwd_raw",
    "iav_raw",
    "gtd_raw",
    "drr_raw",
    "drr_cat",
    "w_awr_def_tot_raw",
]

OUTPUT_COLUMNS = [
    "h3_index",
    "water_stress_score",
    "water_stress_cat",
    "water_stress_label",
    "high_water_stress",
    "extreme_water_stress",
    "arid_low_use",
    "water_depletion_score",
    "interannual_var_score",
    "groundwater_decline",
    "drought_risk_score",
    "high_drought_risk",
    "high_drought_risk_raw",
    "overall_water_risk",
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


def write_gdf_parquet_safe(gdf: gpd.GeoDataFrame, path: Path) -> None:
    from geopandas.io.arrow import _to_parquet

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        _to_parquet(gdf, handle)


def read_gdf_parquet_safe(path: Path) -> gpd.GeoDataFrame:
    with path.open("rb") as handle:
        return gpd.read_parquet(handle)


def mask_sentinel_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include="number").columns
    out[numeric_cols] = out[numeric_cols].replace(SENTINEL, pd.NA)
    # Aqueduct also uses +9999 for some score columns (e.g. bws_raw, bwd_raw).
    out[numeric_cols] = out[numeric_cols].replace(SENTINEL_POS, pd.NA)
    return out


def inspect_aqueduct_csv(csv_path: Path) -> None:
    log(f"Loading {csv_path.name} for inspection...")
    df = pd.read_csv(csv_path, low_memory=False)
    log(f"Shape: {len(df):,} rows x {len(df.columns)} columns")
    log("\nColumns:")
    for col in df.columns:
        log(f"  {col}")

    if "geometry" in df.columns:
        log("\ngeometry column present (WKT)")
    else:
        log(
            "\nNote: no geometry column in this CSV. "
            "Catchment polygons come from HydroBASINS lev 6 via pfaf_id join."
        )

    log("\nHead (2 rows, key columns):")
    key_cols = [c for c in INSPECT_COLS if c in df.columns]
    log(df[key_cols].head(2).to_string())

    df = mask_sentinel_values(df)
    if "bws_raw" in df.columns:
        log(f"\nAfter sentinel mask — bws_raw nulls: {df['bws_raw'].isna().sum()}")
        log(f"bws_raw describe after mask:\n{df['bws_raw'].describe()}")

    log("\nKey column describe() (after sentinel mask):")
    for col in INSPECT_COLS:
        if col in df.columns:
            log(f"\n{col}:")
            log(df[col].describe().to_string())

    if "drr_raw" in df.columns and "drr_cat" in df.columns:
        log(
            "\nDrought flags: high_drought_risk = drr_cat >= 2; "
            "high_drought_risk_raw = drr_raw >= 0.5."
        )


def load_aqueduct_us(csv_path: Path) -> pd.DataFrame:
    log(f"Loading Aqueduct CSV from {csv_path.name}...")
    available = pd.read_csv(csv_path, nrows=0).columns.tolist()
    usecols = [c for c in AQUEDUCT_COLS if c in available]
    if "pfaf_id" not in usecols:
        raise ValueError("Aqueduct CSV missing pfaf_id column")

    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    log(f"  {len(df):,} global catchment rows")
    log(f"  Columns: {', '.join(usecols)}")

    df = mask_sentinel_values(df)
    log(f"After sentinel mask — bws_raw nulls: {df['bws_raw'].isna().sum()}")
    log(f"bws_raw describe after mask:\n{df['bws_raw'].describe()}")

    if "geometry" in available:
        log("  Parsing WKT geometry column...")
        full = pd.read_csv(
            csv_path, usecols=usecols + ["geometry"], low_memory=False
        )
        geom = full["geometry"]
        attrs = mask_sentinel_values(full.drop(columns=["geometry"]))
        gdf = gpd.GeoDataFrame(
            attrs,
            geometry=gpd.GeoSeries.from_wkt(geom),
            crs="EPSG:4326",
        )
        return gdf

    us = df[df["gid_0"] == "USA"].copy()
    log(f"  US rows (gid_0=USA): {len(us):,}")
    us = us[us["pfaf_id"] > 0].copy()

    # Multiple Aqueduct rows share a pfaf_id (province/aquifer splits).
    # For basin polygons, keep the highest baseline water stress per pfaf_id.
    us["_rank"] = us["bws_raw"].fillna(-1)
    us = (
        us.sort_values("_rank", ascending=False)
        .drop_duplicates("pfaf_id", keep="first")
        .drop(columns=["_rank"])
    )
    log(f"  {len(us):,} unique US pfaf_id records after dedupe")
    return us


def load_hydrobasins(cache_path: Path, force: bool = False) -> gpd.GeoDataFrame:
    if cache_path.exists() and not force:
        log(f"Loading HydroBASINS from {cache_path.name}...")
        basins = read_gdf_parquet_safe(cache_path)
        if basins.crs is None:
            basins = basins.set_crs("EPSG:4326")
        log(f"  {len(basins):,} basins (cached)")
        return basins

    log("Downloading HydroBASINS North America lev 6...")
    basins = gpd.read_file(HYBAS_URL)[["PFAF_ID", "geometry"]]
    basins = basins.rename(columns={"PFAF_ID": "pfaf_id"})
    basins = basins[basins["pfaf_id"] > 0]
    write_gdf_parquet_safe(basins, cache_path)
    log(f"  Saved {len(basins):,} basins to {cache_path.name}")
    return basins


def build_catchment_gdf(
    aqueduct_us: pd.DataFrame | gpd.GeoDataFrame,
    basins: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if isinstance(aqueduct_us, gpd.GeoDataFrame):
        gdf = aqueduct_us.copy()
    else:
        merged = basins.merge(aqueduct_us, on="pfaf_id", how="inner")
        gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs=basins.crs)

    gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty]
    gdf = gdf[gdf.geometry.is_valid]
    gdf_proj = gdf.to_crs(WORK_CRS)
    cent = gdf_proj.geometry.centroid.to_crs("EPSG:4326")
    gdf = gdf[
        (cent.x >= CONUS_MIN_LON)
        & (cent.x <= CONUS_MAX_LON)
        & (cent.y >= CONUS_MIN_LAT)
        & (cent.y <= CONUS_MAX_LAT)
    ].copy()
    log(f"  {len(gdf):,} US catchments after CONUS bbox filter")
    return gdf


def load_grid_centroids(grid_path: Path, sample: int | None) -> gpd.GeoDataFrame:
    log(f"Loading H3 grid from {grid_path.name}...")
    with grid_path.open("rb") as handle:
        grid = gpd.read_parquet(handle)
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
    return points


def spatial_join_aqueduct(
    grid_points: gpd.GeoDataFrame,
    catchments: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    feature_cols = [
        c
        for c in (
            "bws_raw",
            "bws_cat",
            "bws_label",
            "bwd_raw",
            "iav_raw",
            "gtd_raw",
            "drr_raw",
            "drr_cat",
            "w_awr_def_tot_raw",
        )
        if c in catchments.columns
    ]
    catch_proj = catchments[feature_cols + ["geometry"]].to_crs(WORK_CRS)
    grid_proj = grid_points.to_crs(WORK_CRS)

    log("Spatial join: H3 centroids within Aqueduct catchments...")
    joined = gpd.sjoin(
        grid_proj[["h3_index", "geometry"]],
        catch_proj,
        how="left",
        predicate="within",
    )
    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])

    missing_ids = joined.loc[joined["bws_raw"].isna(), "h3_index"]
    if len(missing_ids):
        null_points = grid_proj[grid_proj["h3_index"].isin(missing_ids)]
        log(
            f"  Retrying {len(null_points):,} unmatched cells with predicate='intersects'..."
        )
        fallback = gpd.sjoin(
            null_points[["h3_index", "geometry"]],
            catch_proj,
            how="left",
            predicate="intersects",
        )
        if "index_right" in fallback.columns:
            fallback = fallback.drop(columns=["index_right"])
        fallback = fallback.sort_values("bws_raw", ascending=False, na_position="last")
        fallback = fallback.drop_duplicates("h3_index", keep="first")

        matched_within = joined[joined["bws_raw"].notna()].copy()
        fallback = fallback[~fallback["h3_index"].isin(matched_within["h3_index"])]
        still_null = joined[
            joined["h3_index"].isin(missing_ids)
            & ~joined["h3_index"].isin(fallback["h3_index"])
        ]
        joined = pd.concat([matched_within, fallback, still_null], ignore_index=True)

    dupes = joined["h3_index"].duplicated().sum()
    if dupes:
        joined = joined.sort_values("bws_raw", ascending=False, na_position="last")
        joined = joined.drop_duplicates("h3_index", keep="first")
        log(f"  Warning: resolved {dupes:,} duplicate join matches")

    matched = joined["bws_raw"].notna().sum()
    log(f"  {matched:,} / {len(joined):,} cells matched catchment attributes")
    return joined


def derive_binary_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bws_cat = pd.to_numeric(out["water_stress_cat"], errors="coerce")
    drr_cat = pd.to_numeric(out.get("drr_cat"), errors="coerce")

    out["high_water_stress"] = (bws_cat >= 3).fillna(False).astype(int)
    out["extreme_water_stress"] = (bws_cat == 4).fillna(False).astype(int)
    out["arid_low_use"] = (bws_cat == -1).fillna(False).astype(int)
    out["high_drought_risk"] = (drr_cat >= 2).fillna(False).astype(int)
    drr_raw = pd.to_numeric(out["drought_risk_score"], errors="coerce")
    out["high_drought_risk_raw"] = (drr_raw >= 0.5).fillna(False).astype(int)
    return out


def build_features(joined: gpd.GeoDataFrame) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "h3_index": joined["h3_index"],
            "water_stress_score": pd.to_numeric(joined["bws_raw"], errors="coerce"),
            "water_stress_cat": pd.to_numeric(joined["bws_cat"], errors="coerce"),
            "water_stress_label": joined["bws_label"],
            "water_depletion_score": pd.to_numeric(joined["bwd_raw"], errors="coerce"),
            "interannual_var_score": pd.to_numeric(joined["iav_raw"], errors="coerce"),
            "groundwater_decline": pd.to_numeric(joined["gtd_raw"], errors="coerce"),
            "drought_risk_score": pd.to_numeric(joined["drr_raw"], errors="coerce"),
            "drr_cat": pd.to_numeric(joined["drr_cat"], errors="coerce"),
            "overall_water_risk": pd.to_numeric(
                joined["w_awr_def_tot_raw"], errors="coerce"
            ),
            "source": SOURCE,
        }
    )
    result = derive_binary_flags(result)
    for col in ("water_stress_cat",):
        result[col] = result[col].astype("Int64")
    return result.drop(columns=["drr_cat"], errors="ignore")


def load_state_lookup() -> pd.DataFrame | None:
    if not TRACT_MAPPING.exists():
        return None
    mapping = pd.read_parquet(TRACT_MAPPING, columns=["h3_index", "tract_fips"])
    mapping["state"] = mapping["tract_fips"].astype(str).str[:2].map(STATE_FIPS_TO_ABBR)
    return mapping[["h3_index", "state"]]


def describe_series(series: pd.Series) -> str:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return "no data"
    return (
        f"min={clean.min():.4f} median={clean.median():.4f} "
        f"mean={clean.mean():.4f} max={clean.max():.4f}"
    )


def print_summary(
    result: pd.DataFrame,
    global_count: int,
    us_catchment_count: int,
    output_path: Path | None,
) -> None:
    has_data = result["water_stress_score"].notna()
    coverage_pct = 100.0 * has_data.sum() / len(result) if len(result) else 0.0

    log("\n" + "=" * 60)
    log("WRI Aqueduct water stress features complete")
    log("=" * 60)
    log(f"Global catchments loaded:           {global_count:,}")
    log(f"US catchments after bbox filter:    {us_catchment_count:,}")
    log(
        f"H3 cells with water stress data:    {has_data.sum():,} "
        f"({coverage_pct:.2f}%)"
    )
    wss = pd.to_numeric(result["water_stress_score"], errors="coerce")
    neg_count = int((wss < 0).sum())
    log("\nSentinel check:")
    log(f"  water_stress_score < 0: {neg_count}")
    log(f"  water_stress_score nulls: {int(wss.isna().sum())}")
    log(f"  water_stress_score describe(): {describe_series(wss)}")

    log("\nwater_stress_label breakdown:")
    for label, count in (
        result.loc[has_data, "water_stress_label"].value_counts(dropna=False).items()
    ):
        log(f"  {label}: {count:,}")

    data = result.loc[has_data]
    for col in (
        "high_water_stress",
        "extreme_water_stress",
        "arid_low_use",
    ):
        count = int(data[col].sum())
        pct = 100.0 * count / len(data) if len(data) else 0.0
        log(f"{col}: {count:,} ({pct:.2f}% of cells with data)")

    drr = pd.to_numeric(data["drought_risk_score"], errors="coerce")
    log("\nDrought check:")
    log(f"  drr_raw describe() for US cells: {describe_series(drr)}")
    hd_count = int(data["high_drought_risk"].sum())
    hd_pct = 100.0 * hd_count / len(data) if len(data) else 0.0
    log(f"  high_drought_risk (drr_cat >= 2): {hd_count:,} ({hd_pct:.2f}%)")
    hdr_count = int(data["high_drought_risk_raw"].sum())
    hdr_pct = 100.0 * hdr_count / len(data) if len(data) else 0.0
    log(f"  high_drought_risk_raw (drr_raw >= 0.5): {hdr_count:,} ({hdr_pct:.2f}%)")

    state_lookup = load_state_lookup()
    if state_lookup is not None and has_data.any():
        with_state = result.loc[has_data].merge(state_lookup, on="h3_index", how="left")
        with_state = with_state.dropna(subset=["state"])
        if not with_state.empty:
            log("\nTop 5 states by mean water_stress_score:")
            for state, score in (
                with_state.groupby("state")["water_stress_score"]
                .mean()
                .sort_values(ascending=False)
                .head(5)
                .items()
            ):
                log(f"  {state}: {score:.4f}")
            log("\nTop 5 states by mean drr_raw:")
            for state, score in (
                with_state.groupby("state")["drought_risk_score"]
                .mean()
                .sort_values(ascending=False)
                .head(5)
                .items()
            ):
                log(f"  {state}: {score:.4f}")

    if output_path is not None and output_path.exists():
        log(f"\nOutput: {output_path} ({_format_bytes(output_path.stat().st_size)})")


def compute_water_stress_features(
    csv_path: Path | None = None,
    grid_path: Path | None = None,
    output_path: Path | None = None,
    hybas_cache: Path | None = None,
    sample: int | None = None,
    force_hybas: bool = False,
) -> pd.DataFrame:
    csv_path = csv_path or DEFAULT_AQUEDUCT_CSV
    resolution = DEFAULT_RESOLUTION
    if grid_path is None:
        grid_path = PROCESSED_DIR / f"h3_grid_res{resolution}.parquet"
    if output_path is None:
        output_path = PROCESSED_DIR / f"features_water_stress_res{resolution}.parquet"
    if hybas_cache is None:
        hybas_cache = HYBAS_CACHE

    if not csv_path.exists():
        raise FileNotFoundError(f"Aqueduct CSV not found: {csv_path}")
    if not grid_path.exists():
        raise FileNotFoundError(
            f"H3 grid not found: {grid_path}. "
            f"Run: python scripts/build_grid.py --resolution {resolution}"
        )

    global_count = len(pd.read_csv(csv_path, usecols=["gid_0"], low_memory=False))

    aqueduct_us = load_aqueduct_us(csv_path)
    if isinstance(aqueduct_us, gpd.GeoDataFrame):
        catchments = build_catchment_gdf(aqueduct_us, basins=gpd.GeoDataFrame())
    else:
        basins = load_hydrobasins(hybas_cache, force=force_hybas)
        catchments = build_catchment_gdf(aqueduct_us, basins)

    us_catchment_count = len(catchments)
    grid_points = load_grid_centroids(grid_path, sample)
    joined = spatial_join_aqueduct(grid_points, catchments)
    features = build_features(joined)

    print_summary(features, global_count, us_catchment_count, output_path=None)

    log(f"\nWriting {output_path.name}...")
    write_parquet_safe(features[OUTPUT_COLUMNS], output_path)
    if output_path.exists():
        log(f"Output: {output_path} ({_format_bytes(output_path.stat().st_size)})")
    os._exit(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute WRI Aqueduct 4.0 water stress features per H3 cell"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_AQUEDUCT_CSV,
        help="Aqueduct baseline annual CSV path",
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
        default=PROCESSED_DIR / f"features_water_stress_res{DEFAULT_RESOLUTION}.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--hybas-cache",
        type=Path,
        default=HYBAS_CACHE,
        help="Cached HydroBASINS NA lev 6 parquet",
    )
    parser.add_argument(
        "--force-hybas",
        action="store_true",
        help="Re-download HydroBASINS polygons",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Load CSV, print columns and describe(), then exit",
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
        if args.inspect:
            inspect_aqueduct_csv(args.csv)
            return 0
        compute_water_stress_features(
            csv_path=args.csv,
            grid_path=args.grid,
            output_path=args.output,
            hybas_cache=args.hybas_cache,
            sample=args.sample,
            force_hybas=args.force_hybas,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        log(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
