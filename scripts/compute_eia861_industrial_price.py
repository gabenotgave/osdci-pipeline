#!/usr/bin/env python3
"""Compute county-level industrial electricity price from EIA-861 and join to H3 grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EIA_DIR = RAW_DIR / "eia_861"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_RESOLUTION = 6
COUNTIES_CACHE = RAW_DIR / "tl_2023_us_county.gpkg"
MIN_UTILITY_INDUSTRIAL_MWH = 1000

SALES_FILE = "Sales_Ult_Cust_2024.xlsx"
TERRITORY_FILE = "Service_Territory_2024.xlsx"

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
    "state",
    "county",
    "county_fips",
    "industrial_price_mwh",
    "industrial_revenue_usd",
    "industrial_sales_mwh",
    "utility_count",
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
    """Normalize county names while preserving independent-city distinctions."""
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
    ):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    return text or None


def _flatten_columns(columns: pd.Index) -> list[str]:
    flat: list[str] = []
    for col in columns:
        if isinstance(col, tuple):
            parts = [str(x).strip().replace("\n", " ") for x in col if str(x) != "nan"]
            flat.append("_".join(parts).strip("_"))
        else:
            flat.append(str(col))
    return flat


def load_utility_sales(eia_dir: Path) -> pd.DataFrame:
    """Load utility-level industrial revenues and sales (Part A, Bundled)."""
    path = eia_dir / SALES_FILE
    if not path.exists():
        raise FileNotFoundError(f"Sales file not found: {path}")

    log(f"Loading utility sales from {path.name}...")
    df = pd.read_excel(path, sheet_name="States", header=[0, 1, 2])
    df.columns = _flatten_columns(df.columns)

    rename = {
        "Utility Characteristics_Unnamed: 1_level_1_Utility Number": "utility_number",
        "Utility Characteristics_Unnamed: 2_level_1_Utility Name": "utility_name",
        "Utility Characteristics_Unnamed: 3_level_1_Part": "part",
        "Utility Characteristics_Unnamed: 4_level_1_Service Type": "service_type",
        "Utility Characteristics_Unnamed: 6_level_1_State": "state",
        "INDUSTRIAL_Revenues_Thousand Dollars": "ind_revenue_kusd",
        "INDUSTRIAL_Sales_Megawatthours": "ind_sales_mwh",
    }
    missing = set(rename) - set(df.columns)
    if missing:
        raise ValueError(f"Sales file missing columns: {missing}")

    df = df.rename(columns=rename)
    df = df[(df["part"] == "A") & (df["service_type"] == "Bundled")].copy()
    df["utility_number"] = pd.to_numeric(df["utility_number"], errors="coerce")
    df["state"] = df["state"].astype(str).str.strip().str.upper()
    df["ind_revenue_kusd"] = pd.to_numeric(df["ind_revenue_kusd"], errors="coerce").fillna(0)
    df["ind_sales_mwh"] = pd.to_numeric(df["ind_sales_mwh"], errors="coerce").fillna(0)

    # Collapse any duplicate utility rows
    sales = (
        df.groupby(["utility_number", "state", "utility_name"], as_index=False)
        .agg(
            ind_revenue_kusd=("ind_revenue_kusd", "sum"),
            ind_sales_mwh=("ind_sales_mwh", "sum"),
        )
    )
    sales = sales[(sales["ind_sales_mwh"] > 0) | (sales["ind_revenue_kusd"] > 0)]
    before = len(sales)
    sales = sales[sales["ind_sales_mwh"] >= MIN_UTILITY_INDUSTRIAL_MWH].copy()
    log(
        f"  {len(sales):,} utilities with ≥ {MIN_UTILITY_INDUSTRIAL_MWH:,} MWh industrial sales "
        f"({before - len(sales):,} excluded for low volume)"
    )
    return sales


def load_service_territory(eia_dir: Path) -> pd.DataFrame:
    path = eia_dir / TERRITORY_FILE
    if not path.exists():
        raise FileNotFoundError(f"Service territory file not found: {path}")

    log(f"Loading service territory from {path.name}...")
    territory = pd.read_excel(path, sheet_name="Counties_States", header=0)
    territory = territory.rename(
        columns={
            "Utility Number": "utility_number",
            "Utility Name": "utility_name_territory",
            "State": "state",
            "County": "county",
        }
    )
    territory["utility_number"] = pd.to_numeric(territory["utility_number"], errors="coerce")
    territory["state"] = territory["state"].astype(str).str.strip().str.upper()
    territory["county_norm"] = territory["county"].map(normalize_county)
    territory = territory.dropna(subset=["utility_number", "state", "county_norm"])
    log(f"  {len(territory):,} utility–county links")
    return territory[["utility_number", "state", "county", "county_norm"]]


def aggregate_county_prices(sales: pd.DataFrame, territory: pd.DataFrame) -> pd.DataFrame:
    """Allocate utility industrial sales to counties and compute $/MWh."""
    log("Aggregating industrial prices to county...")
    linked = territory.merge(sales, on=["utility_number", "state"], how="inner")
    if linked.empty:
        raise ValueError("No overlap between service territory and sales data")

    county_counts = (
        linked.groupby(["utility_number", "state"], as_index=False)
        .size()
        .rename(columns={"size": "county_count"})
    )
    linked = linked.merge(county_counts, on=["utility_number", "state"], how="left")
    linked["alloc_revenue_usd"] = linked["ind_revenue_kusd"] * 1000 / linked["county_count"]
    linked["alloc_sales_mwh"] = linked["ind_sales_mwh"] / linked["county_count"]

    county = (
        linked.groupby(["state", "county", "county_norm"], as_index=False)
        .agg(
            industrial_revenue_usd=("alloc_revenue_usd", "sum"),
            industrial_sales_mwh=("alloc_sales_mwh", "sum"),
            utility_count=("utility_number", "nunique"),
        )
    )
    county = county.drop_duplicates(subset=["state", "county_norm"], keep="first")
    county["industrial_price_mwh"] = county["industrial_revenue_usd"] / county[
        "industrial_sales_mwh"
    ].clip(lower=1e-6)
    county.loc[county["industrial_sales_mwh"] <= 0, "industrial_price_mwh"] = pd.NA

    with_price = county["industrial_price_mwh"].notna().sum()
    log(f"  {len(county):,} counties; {with_price:,} with industrial price")
    return county


def load_counties(cache_path: Path) -> gpd.GeoDataFrame:
    log("Loading US county boundaries...")
    if not cache_path.exists():
        raise FileNotFoundError(
            f"County cache not found: {cache_path}. "
            "Run: python scripts/compute_eia860_features.py"
        )
    counties = gpd.read_file(cache_path).to_crs("EPSG:4326")
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


def build_features(grid_assigned: pd.DataFrame, county_prices: pd.DataFrame) -> pd.DataFrame:
    price_data = county_prices.drop(columns=["county"], errors="ignore")
    out = grid_assigned.merge(price_data, on=["state", "county_norm"], how="left")
    out = out.drop_duplicates(subset="h3_index", keep="first")
    out["source"] = "EIA_861_2024"
    return out[OUTPUT_COLUMNS]


def print_summary(df: pd.DataFrame, output_path: Path) -> None:
    priced = df["industrial_price_mwh"].notna()
    log("\n" + "=" * 60)
    log("EIA-861 industrial price features complete")
    log("=" * 60)
    log(f"H3 cells:                         {len(df):,}")
    log(f"Cells with industrial price:      {priced.sum():,}")
    if priced.any():
        log("\nIndustrial electricity price ($/MWh):")
        log(f"  min:    ${df.loc[priced, 'industrial_price_mwh'].min():,.2f}")
        log(f"  median: ${df.loc[priced, 'industrial_price_mwh'].median():,.2f}")
        log(f"  mean:   ${df.loc[priced, 'industrial_price_mwh'].mean():,.2f}")
        log(f"  max:    ${df.loc[priced, 'industrial_price_mwh'].max():,.2f}")
        log("\nTop 5 most expensive counties ($/MWh):")
        top = (
            df.loc[priced, ["state", "county", "industrial_price_mwh"]]
            .drop_duplicates(["state", "county"])
            .sort_values("industrial_price_mwh", ascending=False)
            .head(5)
        )
        for _, row in top.iterrows():
            log(f"  {row['county']}, {row['state']}: ${row['industrial_price_mwh']:,.2f}")
    if output_path.exists():
        log(f"\nOutput: {output_path} ({_format_bytes(output_path.stat().st_size)})")


def compute_industrial_price(
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
        output_path = PROCESSED_DIR / f"features_industrial_price_res{resolution}.parquet"
    if counties_cache is None:
        counties_cache = COUNTIES_CACHE

    if not eia_dir.exists():
        raise FileNotFoundError(f"EIA-861 directory not found: {eia_dir}")
    if not grid_path.exists():
        raise FileNotFoundError(
            f"H3 grid not found: {grid_path}. "
            f"Run: python scripts/build_grid.py --resolution {resolution}"
        )

    sales = load_utility_sales(eia_dir)
    territory = load_service_territory(eia_dir)
    county_prices = aggregate_county_prices(sales, territory)

    counties = load_counties(counties_cache)
    grid_pts = load_grid(grid_path)
    grid_assigned = assign_counties(grid_pts, counties)
    result = build_features(grid_assigned, county_prices)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result.to_parquet(output_path, index=False)
    except OSError as exc:
        raise OSError(f"Failed to write {output_path}: {exc}") from exc

    print_summary(result, output_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute county-level industrial electricity price from EIA-861"
    )
    parser.add_argument(
        "--eia-dir",
        type=Path,
        default=EIA_DIR,
        help="Directory containing EIA-861 Excel files",
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
        default=PROCESSED_DIR / f"features_industrial_price_res{DEFAULT_RESOLUTION}.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--counties-cache",
        type=Path,
        default=COUNTIES_CACHE,
        help="Cached county boundaries GeoPackage",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        compute_industrial_price(
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
