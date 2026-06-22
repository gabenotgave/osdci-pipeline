#!/usr/bin/env python3
"""Compute grid carbon intensity and renewable share from EPA eGRID and join to H3 grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_RESOLUTION = 6
EGRID_XLSX = RAW_DIR / "egrid2023_data_rev2.xlsx"
EGRID_SUBREGIONS_SHP = RAW_DIR / "egrid2023_subregions" / "eGRID2023_Subregions.shp"
COUNTIES_CACHE = RAW_DIR / "tl_2023_us_county.gpkg"

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

RENEWABLE_FUELS = ("HY", "BM", "WI", "SO", "GT")
FOSSIL_FUELS = ("CL", "OL", "GS", "OF", "OP")

OUTPUT_COLUMNS = [
    "h3_index",
    "state",
    "county",
    "county_fips",
    "ba_code",
    "ba_name",
    "egrid_subregion",
    "egrid_subregion_name",
    "renewable_pct_ba",
    "nuclear_pct_ba",
    "fossil_pct_ba",
    "co2_lb_mwh_ba",
    "co2e_lb_mwh_ba",
    "net_generation_mwh_ba",
    "renewable_pct_subregion",
    "nuclear_pct_subregion",
    "fossil_pct_subregion",
    "co2_lb_mwh_subregion",
    "co2e_lb_mwh_subregion",
    "net_generation_mwh_subregion",
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


def _fuel_generation(df: pd.DataFrame, prefix: str, fuels: tuple[str, ...]) -> pd.Series:
    cols = [f"{prefix}GENA{fuel}" for fuel in fuels]
    existing = [col for col in cols if col in df.columns]
    if not existing:
        return pd.Series(0.0, index=df.index)
    return df[existing].fillna(0.0).sum(axis=1)


def _pct(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, pd.NA) * 100).astype("Float64")


def load_ba_metrics(xlsx_path: Path) -> pd.DataFrame:
    log(f"Loading BA metrics from {xlsx_path.name} (BA23)...")
    df = pd.read_excel(xlsx_path, sheet_name="BA23", header=1)
    total = df["BANGENAN"].astype("Float64")
    renewable = _fuel_generation(df, "BA", RENEWABLE_FUELS)
    fossil = _fuel_generation(df, "BA", FOSSIL_FUELS)
    nuclear = df["BAGENANC"].fillna(0.0)

    out = pd.DataFrame(
        {
            "ba_code": df["BACODE"].astype(str).str.strip(),
            "ba_name": df["BANAME"].astype(str).str.strip(),
            "renewable_pct_ba": _pct(renewable, total),
            "nuclear_pct_ba": _pct(nuclear, total),
            "fossil_pct_ba": _pct(fossil, total),
            "co2_lb_mwh_ba": pd.to_numeric(df["BACO2RTA"], errors="coerce"),
            "co2e_lb_mwh_ba": pd.to_numeric(df["BAC2ERTA"], errors="coerce"),
            "net_generation_mwh_ba": total,
        }
    )
    out = out[out["ba_code"].notna() & (out["ba_code"] != "nan")]
    log(f"  {len(out):,} balancing authorities")
    return out


def load_subregion_metrics(xlsx_path: Path) -> pd.DataFrame:
    log(f"Loading subregion metrics from {xlsx_path.name} (SRL23)...")
    df = pd.read_excel(xlsx_path, sheet_name="SRL23", header=1)
    total = df["SRNGENAN"].astype("Float64")
    renewable = _fuel_generation(df, "SR", RENEWABLE_FUELS)
    fossil = _fuel_generation(df, "SR", FOSSIL_FUELS)
    nuclear = df["SRGENANC"].fillna(0.0)

    out = pd.DataFrame(
        {
            "egrid_subregion": df["SUBRGN"].astype(str).str.strip(),
            "egrid_subregion_name": df["SRNAME"].astype(str).str.strip(),
            "renewable_pct_subregion": _pct(renewable, total),
            "nuclear_pct_subregion": _pct(nuclear, total),
            "fossil_pct_subregion": _pct(fossil, total),
            "co2_lb_mwh_subregion": pd.to_numeric(df["SRCO2RTA"], errors="coerce"),
            "co2e_lb_mwh_subregion": pd.to_numeric(df["SRC2ERTA"], errors="coerce"),
            "net_generation_mwh_subregion": total,
        }
    )
    log(f"  {len(out):,} eGRID subregions")
    return out


def load_county_ba_lookup(xlsx_path: Path) -> pd.DataFrame:
    """Most common balancing authority per county from eGRID plant locations."""
    log(f"Building county→BA lookup from {xlsx_path.name} (PLNT23)...")
    plants = pd.read_excel(xlsx_path, sheet_name="PLNT23", header=1).copy()
    plants = plants.dropna(subset=["FIPSST", "FIPSCNTY"])
    plants["county_fips"] = (
        plants["FIPSST"].astype(int).astype(str).str.zfill(2)
        + plants["FIPSCNTY"].astype(int).astype(str).str.zfill(3)
    )
    plants["ba_code"] = plants["BACODE"].astype(str).str.strip()
    plants = plants[
        plants["ba_code"].notna()
        & ~plants["ba_code"].str.startswith("NA")
        & (plants["ba_code"] != "nan")
    ]

    lookup = (
        plants.groupby(["county_fips", "ba_code", "BANAME"], as_index=False)
        .size()
        .rename(columns={"size": "plant_count", "BANAME": "ba_name"})
    )
    lookup = lookup.sort_values(["county_fips", "plant_count"], ascending=[True, False])
    lookup = lookup.drop_duplicates("county_fips")[["county_fips", "ba_code"]]
    log(f"  {len(lookup):,} counties with BA assignment")
    return lookup


def load_subregion_polygons(shp_path: Path) -> gpd.GeoDataFrame:
    log(f"Loading eGRID subregion polygons from {shp_path.name}...")
    subregions = gpd.read_file(shp_path).to_crs("EPSG:4326")
    subregions = subregions.rename(columns={"Subregion": "egrid_subregion"})
    log(f"  {len(subregions):,} subregion polygons")
    return subregions[["egrid_subregion", "geometry"]]


def load_counties(cache_path: Path) -> gpd.GeoDataFrame:
    if not cache_path.exists():
        raise FileNotFoundError(
            f"County cache not found: {cache_path}. "
            "Run: python scripts/compute_eia860_features.py"
        )
    log("Loading US county boundaries...")
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


def assign_subregions(grid_pts: gpd.GeoDataFrame, subregions: gpd.GeoDataFrame) -> pd.DataFrame:
    log("Assigning eGRID subregions to H3 cells...")
    joined = gpd.sjoin(
        grid_pts,
        subregions,
        how="left",
        predicate="within",
    )
    joined = joined.drop(columns=["geometry", "index_right"], errors="ignore")
    joined = joined.drop_duplicates(subset="h3_index", keep="first")
    unmatched = joined["egrid_subregion"].isna().sum()
    if unmatched:
        log(f"  Warning: {unmatched:,} cells did not match an eGRID subregion polygon")
    return joined[["h3_index", "egrid_subregion"]]


def build_features(
    grid_assigned: pd.DataFrame,
    subregion_assigned: pd.DataFrame,
    county_ba_lookup: pd.DataFrame,
    ba_metrics: pd.DataFrame,
    subregion_metrics: pd.DataFrame,
) -> pd.DataFrame:
    out = grid_assigned.merge(subregion_assigned, on="h3_index", how="left")
    out = out.merge(county_ba_lookup, on="county_fips", how="left")
    out = out.merge(ba_metrics, on="ba_code", how="left")
    out = out.merge(subregion_metrics, on="egrid_subregion", how="left")
    out["source"] = "EPA_eGRID_2023"
    return out[OUTPUT_COLUMNS]


def print_summary(df: pd.DataFrame, output_path: Path) -> None:
    has_ba = df["co2_lb_mwh_ba"].notna()
    has_srl = df["co2_lb_mwh_subregion"].notna()
    log("\n" + "=" * 60)
    log("EPA eGRID grid mix features complete")
    log("=" * 60)
    log(f"H3 cells:                         {len(df):,}")
    log(f"Cells with BA carbon intensity:   {has_ba.sum():,}")
    log(f"Cells with subregion carbon data: {has_srl.sum():,}")
    if has_ba.any():
        log("\nBA-level CO2 output rate (lb/MWh):")
        log(f"  min:    {df.loc[has_ba, 'co2_lb_mwh_ba'].min():,.1f}")
        log(f"  median: {df.loc[has_ba, 'co2_lb_mwh_ba'].median():,.1f}")
        log(f"  max:    {df.loc[has_ba, 'co2_lb_mwh_ba'].max():,.1f}")
        log("\nBA-level renewable share (%):")
        log(f"  min:    {df.loc[has_ba, 'renewable_pct_ba'].min():,.1f}")
        log(f"  median: {df.loc[has_ba, 'renewable_pct_ba'].median():,.1f}")
        log(f"  max:    {df.loc[has_ba, 'renewable_pct_ba'].max():,.1f}")
    if has_srl.any():
        log("\nSubregion CO2 output rate (lb/MWh):")
        log(f"  min:    {df.loc[has_srl, 'co2_lb_mwh_subregion'].min():,.1f}")
        log(f"  median: {df.loc[has_srl, 'co2_lb_mwh_subregion'].median():,.1f}")
        log(f"  max:    {df.loc[has_srl, 'co2_lb_mwh_subregion'].max():,.1f}")
    if output_path.exists():
        log(f"\nOutput: {output_path} ({_format_bytes(output_path.stat().st_size)})")


def compute_egrid_features(
    egrid_xlsx: Path | None = None,
    subregions_shp: Path | None = None,
    grid_path: Path | None = None,
    output_path: Path | None = None,
    counties_cache: Path | None = None,
) -> pd.DataFrame:
    egrid_xlsx = egrid_xlsx or EGRID_XLSX
    subregions_shp = subregions_shp or EGRID_SUBREGIONS_SHP
    resolution = DEFAULT_RESOLUTION
    if grid_path is None:
        grid_path = PROCESSED_DIR / f"h3_grid_res{resolution}.parquet"
    if output_path is None:
        output_path = PROCESSED_DIR / f"features_egrid_grid_res{resolution}.parquet"
    if counties_cache is None:
        counties_cache = COUNTIES_CACHE

    if not egrid_xlsx.exists():
        raise FileNotFoundError(f"eGRID workbook not found: {egrid_xlsx}")
    if not subregions_shp.exists():
        raise FileNotFoundError(f"eGRID subregions shapefile not found: {subregions_shp}")
    if not grid_path.exists():
        raise FileNotFoundError(
            f"H3 grid not found: {grid_path}. "
            f"Run: python scripts/build_grid.py --resolution {resolution}"
        )

    ba_metrics = load_ba_metrics(egrid_xlsx)
    subregion_metrics = load_subregion_metrics(egrid_xlsx)
    county_ba_lookup = load_county_ba_lookup(egrid_xlsx)
    subregion_polygons = load_subregion_polygons(subregions_shp)

    counties = load_counties(counties_cache)
    grid_pts = load_grid(grid_path)
    grid_assigned = assign_counties(grid_pts, counties)
    subregion_assigned = assign_subregions(grid_pts, subregion_polygons)
    result = build_features(
        grid_assigned,
        subregion_assigned,
        county_ba_lookup,
        ba_metrics,
        subregion_metrics,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result.to_parquet(output_path, index=False)
    except OSError as exc:
        raise OSError(f"Failed to write {output_path}: {exc}") from exc

    print_summary(result, output_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute EPA eGRID renewable share and carbon intensity per H3 cell"
    )
    parser.add_argument(
        "--egrid-xlsx",
        type=Path,
        default=EGRID_XLSX,
        help="EPA eGRID Excel workbook",
    )
    parser.add_argument(
        "--subregions-shp",
        type=Path,
        default=EGRID_SUBREGIONS_SHP,
        help="eGRID subregions shapefile",
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
        default=PROCESSED_DIR / f"features_egrid_grid_res{DEFAULT_RESOLUTION}.parquet",
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
        compute_egrid_features(
            egrid_xlsx=args.egrid_xlsx,
            subregions_shp=args.subregions_shp,
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
