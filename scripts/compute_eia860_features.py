#!/usr/bin/env python3
"""Build H3 grid features from EIA-860: operating capacity and queue congestion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EIA_DIR = RAW_DIR / "eia_860"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_RESOLUTION = 6
COUNTIES_URL = "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip"

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

PLANT_FILE = "2___Plant_Y2024.xlsx"
GENERATOR_FILE = "3_1_Generator_Y2024.xlsx"

OUTPUT_COLUMNS = [
    "h3_index",
    "state",
    "county",
    "county_fips",
    "ba_code",
    "ba_name",
    "operating_capacity_mw",
    "queue_capacity_mw",
    "queue_generator_count",
    "queue_congestion_ratio",
    "operating_capacity_mw_ba",
    "queue_capacity_mw_ba",
    "queue_generator_count_ba",
    "queue_congestion_ratio_ba",
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


def normalize_county(name: object) -> str | None:
    if pd.isna(name):
        return None
    text = str(name).strip().lower()
    for suffix in (
        " county",
        " parish",
        " borough",
        " census area",
        " municipio",
        " city and borough",
        " municipality",
        " city",
    ):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    return text or None


def parse_mw(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def load_eia_excel(path: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet, header=1)
    return df.dropna(subset=["Plant Code"], how="all")


def load_plants(eia_dir: Path) -> pd.DataFrame:
    path = eia_dir / PLANT_FILE
    if not path.exists():
        raise FileNotFoundError(f"Plant file not found: {path}")
    log(f"Loading plants from {path.name}...")
    plants = load_eia_excel(path, "Plant")
    plants = plants.rename(
        columns={
            "Balancing Authority Code": "ba_code",
            "Balancing Authority Name": "ba_name",
        }
    )
    plants["state"] = plants["State"].astype(str).str.strip().str.upper()
    plants["county_norm"] = plants["County"].map(normalize_county)
    plants["plant_code"] = pd.to_numeric(plants["Plant Code"], errors="coerce")
    log(f"  {len(plants):,} plants")
    return plants[
        ["plant_code", "state", "County", "county_norm", "ba_code", "ba_name"]
    ]


def load_generators(eia_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = eia_dir / GENERATOR_FILE
    if not path.exists():
        raise FileNotFoundError(f"Generator file not found: {path}")

    log(f"Loading operable generators from {path.name}...")
    operable = load_eia_excel(path, "Operable")
    operable["plant_code"] = pd.to_numeric(operable["Plant Code"], errors="coerce")
    operable["state"] = operable["State"].astype(str).str.strip().str.upper()
    operable["county_norm"] = operable["County"].map(normalize_county)
    operable["capacity_mw"] = parse_mw(operable["Summer Capacity (MW)"])
    missing = operable["capacity_mw"] <= 0
    operable.loc[missing, "capacity_mw"] = parse_mw(
        operable.loc[missing, "Nameplate Capacity (MW)"]
    )
    operable = operable[operable["capacity_mw"] > 0]
    log(f"  {len(operable):,} operable generators")

    log(f"Loading proposed generators from {path.name}...")
    proposed = load_eia_excel(path, "Proposed")
    proposed["plant_code"] = pd.to_numeric(proposed["Plant Code"], errors="coerce")
    proposed["state"] = proposed["State"].astype(str).str.strip().str.upper()
    proposed["county_norm"] = proposed["County"].map(normalize_county)
    proposed["capacity_mw"] = parse_mw(proposed["Nameplate Capacity (MW)"])
    proposed = proposed[proposed["capacity_mw"] > 0]
    log(f"  {len(proposed):,} proposed generators")
    return operable, proposed


def aggregate_by_county(operable: pd.DataFrame, proposed: pd.DataFrame) -> pd.DataFrame:
    county_oper = (
        operable.groupby(["state", "county_norm"], as_index=False)
        .agg(
            operating_capacity_mw=("capacity_mw", "sum"),
            operating_generator_count=("capacity_mw", "count"),
        )
    )
    county_queue = (
        proposed.groupby(["state", "county_norm"], as_index=False)
        .agg(
            queue_capacity_mw=("capacity_mw", "sum"),
            queue_generator_count=("capacity_mw", "count"),
        )
    )
    county = county_oper.merge(county_queue, on=["state", "county_norm"], how="outer").fillna(
        {
            "operating_capacity_mw": 0.0,
            "operating_generator_count": 0,
            "queue_capacity_mw": 0.0,
            "queue_generator_count": 0,
        }
    )
    county["queue_congestion_ratio"] = county["queue_capacity_mw"] / county[
        "operating_capacity_mw"
    ].clip(lower=1.0)
    return county


def aggregate_by_ba(
    operable: pd.DataFrame, proposed: pd.DataFrame, plants: pd.DataFrame
) -> pd.DataFrame:
    gen_plant = plants[["plant_code", "ba_code", "ba_name"]].drop_duplicates("plant_code")

    oper = operable.merge(gen_plant, on="plant_code", how="left")
    prop = proposed.merge(gen_plant, on="plant_code", how="left")

    ba_oper = (
        oper.dropna(subset=["ba_code"])
        .groupby(["ba_code", "ba_name"], as_index=False)
        .agg(operating_capacity_mw_ba=("capacity_mw", "sum"))
    )
    ba_queue = (
        prop.dropna(subset=["ba_code"])
        .groupby(["ba_code", "ba_name"], as_index=False)
        .agg(
            queue_capacity_mw_ba=("capacity_mw", "sum"),
            queue_generator_count_ba=("capacity_mw", "count"),
        )
    )
    ba = ba_oper.merge(ba_queue, on=["ba_code", "ba_name"], how="outer").fillna(
        {
            "operating_capacity_mw_ba": 0.0,
            "queue_capacity_mw_ba": 0.0,
            "queue_generator_count_ba": 0,
        }
    )
    ba["queue_congestion_ratio_ba"] = ba["queue_capacity_mw_ba"] / ba[
        "operating_capacity_mw_ba"
    ].clip(lower=1.0)
    return ba


def county_ba_lookup(plants: pd.DataFrame) -> pd.DataFrame:
    """Most common BA per state+county from plant locations."""
    valid = plants.dropna(subset=["county_norm", "ba_code"])
    lookup = (
        valid.groupby(["state", "county_norm", "ba_code", "ba_name"], as_index=False)
        .size()
        .rename(columns={"size": "plant_count"})
    )
    lookup = lookup.sort_values(
        ["state", "county_norm", "plant_count"], ascending=[True, True, False]
    )
    return lookup.drop_duplicates(["state", "county_norm"])[
        ["state", "county_norm", "ba_code", "ba_name"]
    ]


def load_counties(cache_path: Path | None = None) -> gpd.GeoDataFrame:
    log("Loading US county boundaries...")
    if cache_path and cache_path.exists():
        counties = gpd.read_file(cache_path)
    else:
        counties = gpd.read_file(COUNTIES_URL)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            counties.to_file(cache_path, driver="GPKG")
            log(f"  Cached counties to {cache_path}")
    counties = counties.to_crs("EPSG:4326")
    counties["state"] = counties["STATEFP"].map(STATE_FIPS_TO_ABBR)
    counties = counties.dropna(subset=["state"])
    counties["county_norm"] = counties["NAME"].map(normalize_county)
    counties["county_fips"] = (
        counties["STATEFP"].astype(str).str.zfill(2)
        + counties["COUNTYFP"].astype(str).str.zfill(3)
    )
    log(f"  {len(counties):,} counties")
    return counties[["state", "NAME", "county_norm", "county_fips", "geometry"]]


def load_grid(grid_path: Path) -> gpd.GeoDataFrame:
    log(f"Loading H3 grid from {grid_path}...")
    grid = gpd.read_parquet(grid_path)
    if grid.index.name == "h3_index":
        grid = grid.reset_index()
    points = gpd.GeoDataFrame(
        grid[["h3_index", "lat", "lon"]],
        geometry=gpd.points_from_xy(grid["lon"], grid["lat"]),
        crs="EPSG:4326",
    )
    log(f"  {len(points):,} hex cells")
    return points


def assign_counties(grid_pts: gpd.GeoDataFrame, counties: gpd.GeoDataFrame) -> pd.DataFrame:
    log("Assigning counties to H3 cells...")
    joined = gpd.sjoin(
        grid_pts,
        counties,
        how="left",
        predicate="within",
    )
    joined = joined.drop(columns=["geometry", "index_right"], errors="ignore")
    joined = joined.rename(columns={"NAME": "county"})
    joined = joined.drop_duplicates(subset="h3_index", keep="first")
    unmatched = joined["county_fips"].isna().sum()
    if unmatched:
        log(f"  Warning: {unmatched:,} cells did not match a county polygon")
    return joined


def build_features(
    grid_assigned: pd.DataFrame,
    county_stats: pd.DataFrame,
    ba_stats: pd.DataFrame,
    ba_lookup: pd.DataFrame,
) -> pd.DataFrame:
    out = grid_assigned.merge(
        county_stats,
        on=["state", "county_norm"],
        how="left",
    )
    out = out.merge(ba_lookup, on=["state", "county_norm"], how="left", suffixes=("", "_lookup"))
    for col in ("ba_code", "ba_name"):
        lookup_col = f"{col}_lookup"
        if lookup_col in out.columns:
            out[col] = out[col].fillna(out[lookup_col])
            out = out.drop(columns=[lookup_col])

    out = out.merge(ba_stats, on=["ba_code", "ba_name"], how="left")

    fill_zero = [
        "operating_capacity_mw",
        "queue_capacity_mw",
        "queue_generator_count",
        "queue_congestion_ratio",
        "operating_capacity_mw_ba",
        "queue_capacity_mw_ba",
        "queue_generator_count_ba",
        "queue_congestion_ratio_ba",
    ]
    for col in fill_zero:
        if col in out.columns:
            out[col] = out[col].fillna(0.0 if "count" not in col else 0)

    out["source"] = "EIA_860_2024"
    return out[OUTPUT_COLUMNS]


def print_summary(df: pd.DataFrame, output_path: Path) -> None:
    log("\n" + "=" * 60)
    log("EIA-860 grid features complete")
    log("=" * 60)
    log(f"H3 cells:                         {len(df):,}")
    log(f"Cells with county capacity data:  {(df['operating_capacity_mw'] > 0).sum():,}")
    log(f"Cells with queue capacity data:     {(df['queue_capacity_mw'] > 0).sum():,}")
    log("\nCounty-level operating capacity (MW):")
    log(f"  median: {df['operating_capacity_mw'].median():,.1f}")
    log(f"  max:    {df['operating_capacity_mw'].max():,.1f}")
    log("\nCounty-level queue congestion ratio:")
    log(f"  median: {df['queue_congestion_ratio'].median():,.3f}")
    log(f"  max:    {df['queue_congestion_ratio'].max():,.3f}")
    log("\nTop 5 counties by queue capacity (MW):")
    top = (
        df.groupby(["state", "county"], as_index=False)["queue_capacity_mw"]
        .max()
        .sort_values("queue_capacity_mw", ascending=False)
        .head(5)
    )
    for _, row in top.iterrows():
        log(f"  {row['county']}, {row['state']}: {row['queue_capacity_mw']:,.0f} MW")
    if output_path.exists():
        log(f"\nOutput: {output_path} ({_format_bytes(output_path.stat().st_size)})")


def compute_eia860_features(
    eia_dir: Path | None = None,
    grid_path: Path | None = None,
    output_path: Path | None = None,
    counties_cache: Path | None = None,
) -> pd.DataFrame:
    eia_dir = eia_dir or EIA_DIR
    resolution = DEFAULT_RESOLUTION
    if grid_path is None:
        grid_path = PROCESSED_DIR / f"h3_grid_res{resolution}.parquet"
    if output_path is None:
        output_path = PROCESSED_DIR / f"features_eia860_grid_res{resolution}.parquet"
    if counties_cache is None:
        counties_cache = RAW_DIR / "tl_2023_us_county.gpkg"

    if not eia_dir.exists():
        raise FileNotFoundError(f"EIA-860 directory not found: {eia_dir}")
    if not grid_path.exists():
        raise FileNotFoundError(
            f"H3 grid not found: {grid_path}. "
            f"Run: python scripts/build_grid.py --resolution {resolution}"
        )

    plants = load_plants(eia_dir)
    operable, proposed = load_generators(eia_dir)
    county_stats = aggregate_by_county(operable, proposed)
    ba_stats = aggregate_by_ba(operable, proposed, plants)
    ba_lookup = county_ba_lookup(plants)

    counties = load_counties(counties_cache)
    grid_pts = load_grid(grid_path)
    grid_assigned = assign_counties(grid_pts, counties)
    result = build_features(grid_assigned, county_stats, ba_stats, ba_lookup)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result.to_parquet(output_path, index=False)
    except OSError as exc:
        raise OSError(f"Failed to write {output_path}: {exc}") from exc

    print_summary(result, output_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute EIA-860 grid capacity and queue congestion features per H3 cell"
    )
    parser.add_argument(
        "--eia-dir",
        type=Path,
        default=EIA_DIR,
        help="Directory containing EIA-860 Excel files",
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
        default=PROCESSED_DIR / f"features_eia860_grid_res{DEFAULT_RESOLUTION}.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--counties-cache",
        type=Path,
        default=RAW_DIR / "tl_2023_us_county.gpkg",
        help="Local cache path for county boundaries",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        compute_eia860_features(
            eia_dir=args.eia_dir,
            grid_path=args.grid,
            output_path=args.output,
            counties_cache=args.counties_cache,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        log(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
