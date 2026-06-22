#!/usr/bin/env python3
"""Compute flood, seismic, and wildfire risk features per H3 cell from FEMA NRI.

Uses the FEMA National Risk Index (NRI) Census Tract dataset joined to tract
geometries via pygris. Output: features_nri_res6.parquet (flood, seismic,
wildfire, and composite NRI scores).
"""

from __future__ import annotations

import argparse
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
DEFAULT_NRI_CSV = RAW_DIR / "NRI_Table_CensusTracts.csv"
TRACTS_CACHE = RAW_DIR / "census_tracts_2021_cb.parquet"
CENSUS_TRACT_YEAR = 2021
WORK_CRS = "EPSG:5070"
SOURCE_TAG = "FEMA_NRI_v1.20_2025"

CONUS_MIN_LAT = 24.0
CONUS_MAX_LAT = 50.0
CONUS_MIN_LON = -125.0
CONUS_MAX_LON = -66.0

STATE_FIPS = [
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13", "15", "16",
    "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
    "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42",
    "44", "45", "46", "47", "48", "49", "50", "51", "53", "54", "55", "56",
]

# NRI v1.20 splits flood into coastal (CFLD) and inland (IFLD); older releases
# used FLOODP_* columns. We load both schemas and derive a composite flood field.
NRI_USECOLS = [
    "TRACTFIPS",
    "STATEFIPS",
    "STATEABBRV",
    "COUNTY",
    "BUILDVALUE",
    "POPULATION",
    "FLOODP_RISKS",
    "FLOODP_RISKR",
    "FLOODP_EALS",
    "FLOODP_EXPS",
    "CFLD_RISKS",
    "CFLD_RISKR",
    "CFLD_EALS",
    "IFLD_RISKS",
    "IFLD_RISKR",
    "IFLD_EALS",
    "ERQK_RISKS",
    "ERQK_RISKR",
    "ERQK_EALS",
    "WFIR_RISKS",
    "WFIR_RISKR",
    "WFIR_EALS",
    "RISK_SCORE",
    "RISK_RATNG",
]

OUTPUT_COLUMNS = [
    "h3_index",
    "flood_risk_score",
    "flood_risk_rating",
    "flood_eal_dollars",
    "in_floodplain_100yr",
    "in_floodplain_500yr",
    "seismic_risk_score",
    "seismic_risk_rating",
    "seismic_eal_dollars",
    "in_high_seismic",
    "wildfire_risk_score",
    "wildfire_risk_rating",
    "wildfire_eal_dollars",
    "in_high_wildfire",
    "composite_risk_score",
    "composite_risk_rating",
    "tract_fips",
    "source",
]

# NRI v1.20 exact rating strings. "Insufficient Data" and "No Rating" → flags = 0.
RATING_HIGH = frozenset({"Relatively High", "Very High"})
RATING_MODERATE_PLUS = frozenset({"Relatively Moderate", "Relatively High", "Very High"})
UNKNOWN_RATINGS = frozenset({"Insufficient Data", "No Rating"})


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


def available_nri_columns(csv_path: Path) -> list[str]:
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    if header and header[0].startswith("\ufeff"):
        header[0] = header[0].lstrip("\ufeff")
    return [col for col in NRI_USECOLS if col in header]


def rating_flag(rating: object, allowed: frozenset[str]) -> int:
    if pd.isna(rating):
        return 0
    text = str(rating).strip()
    if text in UNKNOWN_RATINGS:
        return 0
    return int(text in allowed)


def apply_binary_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Derive binary hazard flags from exact NRI v1.20 rating strings.

    in_floodplain_100yr / in_high_* use Relatively High or Very High.
    in_floodplain_500yr adds Relatively Moderate.
    These are risk-rating proxies, not direct FEMA NFHL zone intersections.
    """
    out = df.copy()
    out["in_floodplain_100yr"] = out["flood_risk_rating"].map(
        lambda r: rating_flag(r, RATING_HIGH)
    )
    out["in_floodplain_500yr"] = out["flood_risk_rating"].map(
        lambda r: rating_flag(r, RATING_MODERATE_PLUS)
    )
    out["in_high_seismic"] = out["seismic_risk_rating"].map(
        lambda r: rating_flag(r, RATING_HIGH)
    )
    out["in_high_wildfire"] = out["wildfire_risk_rating"].map(
        lambda r: rating_flag(r, RATING_HIGH)
    )
    return out


def describe_series(series: pd.Series) -> str:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return "no data"
    return (
        f"min={clean.min():.2f} median={clean.median():.2f} "
        f"mean={clean.mean():.2f} max={clean.max():.2f}"
    )


def print_raw_nri_distributions(nri_df: pd.DataFrame) -> None:
    log("\nRaw NRI distributions (tract level, before spatial join):")
    log(f"  flood_risk_score:    {describe_series(nri_df['flood_risk_score'])}")
    log(f"  seismic_risk_score:  {describe_series(nri_df['ERQK_RISKS'])}")
    log(f"  wildfire_risk_score: {describe_series(nri_df['WFIR_RISKS'])}")

    wfir = pd.to_numeric(nri_df["WFIR_RISKS"], errors="coerce").dropna()
    if not wfir.empty and wfir.min() >= 0 and wfir.max() <= 100:
        log(
            "  wildfire: WFIR_RISKS is 0-100 national percentile at tract level "
            "(distribution looks correct)"
        )
    else:
        log(
            "  WARNING: WFIR_RISKS tract values are outside 0-100; "
            "scores will be min-max normalized to 0-100"
        )


def maybe_normalize_score(series: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    clean = numeric.dropna()
    if clean.empty:
        return numeric
    if clean.min() >= 0 and clean.max() <= 100:
        return numeric
    log(
        f"  WARNING: {label} outside 0-100 range "
        f"(min={clean.min():.2f}, max={clean.max():.2f}); applying min-max normalization"
    )
    span = clean.max() - clean.min()
    if span <= 0:
        return numeric
    return (numeric - clean.min()) / span * 100.0


def derive_flood_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "FLOODP_RISKS" in df.columns:
        out["flood_risk_score"] = pd.to_numeric(df["FLOODP_RISKS"], errors="coerce")
        out["flood_risk_rating"] = df["FLOODP_RISKR"]
        out["flood_eal_dollars"] = pd.to_numeric(df["FLOODP_EALS"], errors="coerce")
        return out

    if "IFLD_RISKS" not in df.columns and "CFLD_RISKS" not in df.columns:
        raise ValueError(
            "NRI CSV has no flood columns (expected FLOODP_* or CFLD_*/IFLD_*)"
        )

    ifld_score = pd.to_numeric(df.get("IFLD_RISKS"), errors="coerce")
    cfld_score = pd.to_numeric(df.get("CFLD_RISKS"), errors="coerce")
    out["flood_risk_score"] = pd.concat([ifld_score, cfld_score], axis=1).max(axis=1)

    ifld_eal = pd.to_numeric(df.get("IFLD_EALS"), errors="coerce")
    cfld_eal = pd.to_numeric(df.get("CFLD_EALS"), errors="coerce")
    out["flood_eal_dollars"] = ifld_eal.fillna(0) + cfld_eal.fillna(0)
    both_na = ifld_eal.isna() & cfld_eal.isna()
    out.loc[both_na, "flood_eal_dollars"] = pd.NA

    use_cfld = cfld_score.fillna(-1) >= ifld_score.fillna(-1)
    out["flood_risk_rating"] = np.where(
        use_cfld,
        df.get("CFLD_RISKR"),
        df.get("IFLD_RISKR"),
    )
    return out


def load_nri_csv(csv_path: Path) -> pd.DataFrame:
    usecols = available_nri_columns(csv_path)
    if "TRACTFIPS" not in usecols:
        raise ValueError("NRI CSV missing required column TRACTFIPS")

    log(f"Loading NRI CSV from {csv_path.name} ({len(usecols)} columns)...")
    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    df["TRACTFIPS"] = df["TRACTFIPS"].astype(str).str.zfill(11)

    log(f"  {len(df):,} census tracts")
    key_cols = [
        c
        for c in (
            "TRACTFIPS",
            "FLOODP_RISKS",
            "CFLD_RISKS",
            "IFLD_RISKS",
            "ERQK_RISKS",
            "WFIR_RISKS",
            "RISK_SCORE",
        )
        if c in df.columns
    ]
    log("  Null counts (key columns):")
    for col in key_cols:
        log(f"    {col}: {df[col].isna().sum():,}")

    df = derive_flood_fields(df)
    df["flood_risk_score"] = maybe_normalize_score(df["flood_risk_score"], "flood_risk_score")
    df["ERQK_RISKS"] = maybe_normalize_score(df["ERQK_RISKS"], "ERQK_RISKS")
    df["WFIR_RISKS"] = maybe_normalize_score(df["WFIR_RISKS"], "WFIR_RISKS")

    log("  flood_risk_rating value counts:")
    for rating, count in df["flood_risk_rating"].value_counts(dropna=False).items():
        log(f"    {rating}: {count:,}")

    print_raw_nri_distributions(df)
    return df


def download_census_tracts(
    cache_path: Path,
    year: int = CENSUS_TRACT_YEAR,
    force: bool = False,
) -> gpd.GeoDataFrame:
    if cache_path.exists() and not force:
        log(f"Loading Census tracts from {cache_path.name}...")
        tracts = read_gdf_parquet_safe(cache_path)
        if tracts.crs is None:
            tracts = tracts.set_crs("EPSG:4326")
        log(f"  {len(tracts):,} tracts (cached)")
        return tracts

    log(f"Downloading Census tracts for year {year} ({len(STATE_FIPS)} states)...")
    frames: list[gpd.GeoDataFrame] = []
    failed: list[str] = []

    for i, fips in enumerate(STATE_FIPS, start=1):
        try:
            state_tracts = pygris.tracts(state=fips, year=year, cb=True, cache=True)
            if state_tracts is not None and len(state_tracts):
                frames.append(state_tracts[["GEOID", "STATEFP", "geometry"]])
        except Exception as exc:
            failed.append(fips)
            log(f"  Warning: skipping state FIPS {fips}: {exc}")
        if i % 10 == 0:
            log(f"  Downloaded tracts for {i}/{len(STATE_FIPS)} states...")

    if not frames:
        raise ValueError("No Census tracts downloaded")

    tracts = pd.concat(frames, ignore_index=True)
    tracts = gpd.GeoDataFrame(tracts, geometry="geometry", crs=frames[0].crs)
    tracts["GEOID"] = tracts["GEOID"].astype(str).str.zfill(11)
    write_gdf_parquet_safe(tracts, cache_path)
    log(f"  Saved {len(tracts):,} tracts to {cache_path.name}")
    if failed:
        log(f"  Warning: {len(failed)} states skipped: {', '.join(failed)}")
    return tracts


def load_grid_centroids(grid_path: Path, sample: int | None) -> gpd.GeoDataFrame:
    log(f"Loading H3 grid from {grid_path}...")
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


def build_h3_tract_mapping(
    grid_points: gpd.GeoDataFrame,
    tracts: gpd.GeoDataFrame,
) -> pd.DataFrame:
    tract_gdf = tracts.copy()
    if tract_gdf.crs != WORK_CRS:
        tract_gdf = tract_gdf.to_crs(WORK_CRS)

    log("Spatial join: H3 centroids within census tracts...")
    joined = gpd.sjoin(
        grid_points[["h3_index", "lat", "lon", "geometry"]],
        tract_gdf[["GEOID", "geometry"]],
        how="left",
        predicate="within",
    )
    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])

    dupes = joined["h3_index"].duplicated(keep=False).sum()
    if dupes:
        joined = joined.sort_values("h3_index").drop_duplicates("h3_index", keep="first")
        log(f"  Warning: resolved {dupes:,} duplicate centroid matches")

    mapping = joined[["h3_index", "lat", "lon", "GEOID"]].rename(columns={"GEOID": "tract_fips"})
    matched = mapping["tract_fips"].notna().sum()
    log(f"  {matched:,} / {len(mapping):,} cells matched to a census tract")
    return mapping


def load_or_build_h3_tract_mapping(
    grid_points: gpd.GeoDataFrame,
    tracts: gpd.GeoDataFrame,
    mapping_path: Path,
    force_join: bool,
) -> pd.DataFrame:
    expected_cells = len(grid_points)
    if mapping_path.exists() and not force_join:
        log(f"Loading H3→tract mapping from {mapping_path.name}...")
        mapping = pd.read_parquet(mapping_path)
        if len(mapping) == expected_cells:
            log(f"  {len(mapping):,} cells (cached mapping)")
            return mapping
        log(
            f"  Warning: mapping has {len(mapping):,} rows but grid has "
            f"{expected_cells:,}; rebuilding spatial join"
        )

    mapping = build_h3_tract_mapping(grid_points, tracts)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_safe(mapping, mapping_path)
    log(f"  Saved mapping to {mapping_path.name}")
    return mapping


def build_features_from_mapping(
    mapping: pd.DataFrame,
    nri_df: pd.DataFrame,
) -> pd.DataFrame:
    attr_cols = [
        "TRACTFIPS",
        "STATEABBRV",
        "flood_risk_score",
        "flood_risk_rating",
        "flood_eal_dollars",
        "ERQK_RISKS",
        "ERQK_RISKR",
        "ERQK_EALS",
        "WFIR_RISKS",
        "WFIR_RISKR",
        "WFIR_EALS",
        "RISK_SCORE",
        "RISK_RATNG",
    ]
    nri_attrs = nri_df[[c for c in attr_cols if c in nri_df.columns]].copy()

    merged = mapping.merge(
        nri_attrs,
        left_on="tract_fips",
        right_on="TRACTFIPS",
        how="left",
    )

    result = pd.DataFrame(
        {
            "h3_index": merged["h3_index"],
            "flood_risk_score": pd.to_numeric(merged["flood_risk_score"], errors="coerce"),
            "flood_risk_rating": merged["flood_risk_rating"],
            "flood_eal_dollars": pd.to_numeric(merged["flood_eal_dollars"], errors="coerce"),
            "seismic_risk_score": pd.to_numeric(merged["ERQK_RISKS"], errors="coerce"),
            "seismic_risk_rating": merged["ERQK_RISKR"],
            "seismic_eal_dollars": pd.to_numeric(merged["ERQK_EALS"], errors="coerce"),
            "wildfire_risk_score": pd.to_numeric(merged["WFIR_RISKS"], errors="coerce"),
            "wildfire_risk_rating": merged["WFIR_RISKR"],
            "wildfire_eal_dollars": pd.to_numeric(merged["WFIR_EALS"], errors="coerce"),
            "composite_risk_score": pd.to_numeric(merged["RISK_SCORE"], errors="coerce"),
            "composite_risk_rating": merged["RISK_RATNG"],
            "tract_fips": merged["tract_fips"],
            "source": SOURCE_TAG,
            "STATEABBRV": merged.get("STATEABBRV"),
        }
    )
    result = apply_binary_flags(result)
    return result


def conus_mask_from_lat_lon(lat: pd.Series, lon: pd.Series) -> pd.Series:
    return (
        (lat >= CONUS_MIN_LAT)
        & (lat <= CONUS_MAX_LAT)
        & (lon >= CONUS_MIN_LON)
        & (lon <= CONUS_MAX_LON)
    )


def print_summary(
    result: pd.DataFrame,
    mapping: pd.DataFrame,
    nri_df: pd.DataFrame,
    grid_path: Path,
    output_path: Path,
) -> None:
    has_data = result["flood_risk_score"].notna()
    coverage_pct = 100.0 * has_data.sum() / len(result)

    conus = conus_mask_from_lat_lon(mapping["lat"], mapping["lon"])
    conus_df = result.loc[conus.reindex(result.index, fill_value=False)]

    log("\n" + "=" * 60)
    log("FEMA NRI flood / seismic / wildfire features complete")
    log("=" * 60)
    log(f"Census tracts loaded from NRI CSV:  {len(nri_df):,}")
    log(
        f"Tracts matched in mapping:          "
        f"{result['tract_fips'].notna().sum():,} cells with tract assignment"
    )
    log(f"Coverage: {has_data.sum():,} / {len(result):,} H3 cells ({coverage_pct:.2f}%)")

    log("\nRaw NRI distributions (tract level, before spatial join):")
    log(f"  flood_risk_score:    {describe_series(nri_df['flood_risk_score'])}")
    log(f"  seismic_risk_score:  {describe_series(nri_df['ERQK_RISKS'])}")
    log(f"  wildfire_risk_score: {describe_series(nri_df['WFIR_RISKS'])}")

    log("\nH3 cell distributions (after spatial join, all cells with data):")
    for col in ("flood_risk_score", "seismic_risk_score", "wildfire_risk_score"):
        log(f"  {col}: {describe_series(result.loc[has_data, col])}")

    log("\nH3 cell distributions (CONUS only):")
    for col in ("flood_risk_score", "seismic_risk_score", "wildfire_risk_score"):
        log(f"  {col}: {describe_series(conus_df.loc[conus_df[col].notna(), col])}")

    log(
        "\nNote: cell-level wildfire/flood means can exceed tract-level means because "
        "large rural tracts with higher hazard scores contain more H3 hex cells "
        "than dense urban tracts. Per-cell scores match their assigned tract values."
    )

    log("\nBinary flag counts (CONUS, post-fix):")
    for col in (
        "in_floodplain_100yr",
        "in_floodplain_500yr",
        "in_high_seismic",
        "in_high_wildfire",
    ):
        count = int(conus_df[col].sum())
        pct = 100.0 * count / len(conus_df) if len(conus_df) else 0.0
        log(f"  {col}: {count:,} ({pct:.2f}%)")

    log("\nflood_risk_rating breakdown (all cells with data):")
    for rating, count in (
        result.loc[has_data, "flood_risk_rating"].value_counts(dropna=False).items()
    ):
        log(f"  {rating}: {count:,}")

    log("\nwildfire_risk_rating breakdown (all cells with data):")
    for rating, count in (
        result.loc[has_data, "wildfire_risk_rating"].value_counts(dropna=False).items()
    ):
        log(f"  {rating}: {count:,}")

    if "STATEABBRV" in result.columns and has_data.any():
        with_state = result.loc[has_data].dropna(subset=["STATEABBRV"])
        log("\nTop 5 states by mean flood_risk_score:")
        for state, score in (
            with_state.groupby("STATEABBRV")["flood_risk_score"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
            .items()
        ):
            log(f"  {state}: {score:.2f}")

        log("\nTop 5 states by mean wildfire_risk_score:")
        for state, score in (
            with_state.groupby("STATEABBRV")["wildfire_risk_score"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
            .items()
        ):
            log(f"  {state}: {score:.2f}")

    if output_path.exists():
        log(f"\nOutput: {output_path} ({_format_bytes(output_path.stat().st_size)})")


def compute_nri_features(
    nri_csv: Path | None = None,
    grid_path: Path | None = None,
    output_path: Path | None = None,
    tracts_cache: Path | None = None,
    mapping_path: Path | None = None,
    sample: int | None = None,
    force_tracts: bool = False,
    force_join: bool = False,
) -> pd.DataFrame:
    nri_csv = nri_csv or DEFAULT_NRI_CSV
    resolution = DEFAULT_RESOLUTION
    if grid_path is None:
        grid_path = PROCESSED_DIR / f"h3_grid_res{resolution}.parquet"
    if output_path is None:
        output_path = PROCESSED_DIR / f"features_nri_res{resolution}.parquet"
    if tracts_cache is None:
        tracts_cache = TRACTS_CACHE
    if mapping_path is None:
        mapping_path = PROCESSED_DIR / f"h3_tract_mapping_res{resolution}.parquet"

    if not nri_csv.exists():
        raise FileNotFoundError(f"NRI CSV not found: {nri_csv}")
    if not grid_path.exists():
        raise FileNotFoundError(
            f"H3 grid not found: {grid_path}. "
            f"Run: python scripts/build_grid.py --resolution {resolution}"
        )

    nri_df = load_nri_csv(nri_csv)
    grid_points = load_grid_centroids(grid_path, sample)

    if sample is None and mapping_path.exists() and not force_join:
        log(f"Loading H3→tract mapping from {mapping_path.name}...")
        mapping = pd.read_parquet(mapping_path)
        if len(mapping) != len(grid_points):
            raise ValueError(
                f"Mapping cache has {len(mapping):,} rows but grid has "
                f"{len(grid_points):,}; rerun with --force-join"
            )
        log(f"  {len(mapping):,} cells (cached mapping)")
    else:
        tracts = download_census_tracts(tracts_cache, force=force_tracts)
        if sample is None:
            mapping = load_or_build_h3_tract_mapping(
                grid_points, tracts, mapping_path, force_join=force_join
            )
        else:
            mapping = build_h3_tract_mapping(grid_points, tracts)

    features = build_features_from_mapping(mapping, nri_df)
    result = features[OUTPUT_COLUMNS]

    log(f"Writing {output_path.name}...")
    write_parquet_safe(result, output_path)
    log("  Parquet write complete")

    print_summary(features, mapping, nri_df, grid_path, output_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute FEMA NRI flood, seismic, and wildfire risk features per H3 cell"
        )
    )
    parser.add_argument(
        "--nri-csv",
        type=Path,
        default=DEFAULT_NRI_CSV,
        help="FEMA NRI Census Tract CSV path",
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
        default=PROCESSED_DIR / f"features_nri_res{DEFAULT_RESOLUTION}.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--tracts-cache",
        type=Path,
        default=TRACTS_CACHE,
        help="Cached Census tract boundaries parquet",
    )
    parser.add_argument(
        "--mapping-cache",
        type=Path,
        default=PROCESSED_DIR / f"h3_tract_mapping_res{DEFAULT_RESOLUTION}.parquet",
        help="Cached H3 cell to census tract mapping parquet",
    )
    parser.add_argument(
        "--force-tracts",
        action="store_true",
        help="Re-download Census tract boundaries",
    )
    parser.add_argument(
        "--force-join",
        action="store_true",
        help="Rebuild H3→tract spatial join even if mapping cache exists",
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
        compute_nri_features(
            nri_csv=args.nri_csv,
            grid_path=args.grid,
            output_path=args.output,
            tracts_cache=args.tracts_cache,
            mapping_path=args.mapping_cache,
            sample=args.sample,
            force_tracts=args.force_tracts,
            force_join=args.force_join,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        log(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
