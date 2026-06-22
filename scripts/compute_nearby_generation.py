#!/usr/bin/env python3
"""Compute nearby operating generation capacity from EIA-860 plant locations."""

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
DEFAULT_RADII_KM = (5.0, 25.0, 50.0)
DEFAULT_COLOCATION_M = 2000.0

PLANT_FILE = "2___Plant_Y2024.xlsx"
GENERATOR_FILE = "3_1_Generator_Y2024.xlsx"

RENEWABLE_SOURCES = frozenset(
    {"SUN", "WND", "WAT", "GEO", "WDS", "BLQ", "LFG", "MSW", "OBG", "AB", "WH"}
)

OUTPUT_COLUMNS = [
    "h3_index",
    "dist_nearest_plant_m",
    "dist_nearest_plant_km",
    "nearest_plant_operating_mw",
    "operating_capacity_mw_5km",
    "operating_capacity_mw_25km",
    "operating_capacity_mw_50km",
    "renewable_capacity_mw_25km",
    "plant_count_25km",
    "colocated_with_power",
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


def parse_mw(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def load_eia_excel(path: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet, header=1)
    return df.dropna(subset=["Plant Code"], how="all")


def load_plant_sites(eia_dir: Path) -> gpd.GeoDataFrame:
    plant_path = eia_dir / PLANT_FILE
    gen_path = eia_dir / GENERATOR_FILE
    if not plant_path.exists():
        raise FileNotFoundError(f"Plant file not found: {plant_path}")
    if not gen_path.exists():
        raise FileNotFoundError(f"Generator file not found: {gen_path}")

    log(f"Loading plants from {plant_path.name}...")
    plants = load_eia_excel(plant_path, "Plant")
    plants["plant_code"] = pd.to_numeric(plants["Plant Code"], errors="coerce")
    plants["lat"] = pd.to_numeric(plants["Latitude"], errors="coerce")
    plants["lon"] = pd.to_numeric(plants["Longitude"], errors="coerce")
    plants = plants.dropna(subset=["plant_code", "lat", "lon"]).drop_duplicates("plant_code")
    plants = plants[
        (plants["lat"].between(-90, 90))
        & (plants["lon"].between(-180, 180))
        & ~((plants["lat"] == 0) & (plants["lon"] == 0))
    ]
    log(f"  {len(plants):,} plants with coordinates")

    log(f"Loading operable generators from {gen_path.name}...")
    generators = load_eia_excel(gen_path, "Operable")
    generators["plant_code"] = pd.to_numeric(generators["Plant Code"], errors="coerce")
    generators["capacity_mw"] = parse_mw(generators["Summer Capacity (MW)"])
    missing = generators["capacity_mw"] <= 0
    generators.loc[missing, "capacity_mw"] = parse_mw(
        generators.loc[missing, "Nameplate Capacity (MW)"]
    )
    generators = generators[
        (generators["Status"] == "OP") & (generators["capacity_mw"] > 0)
    ]
    generators["energy_source"] = (
        generators["Energy Source 1"].astype(str).str.strip().str.upper()
    )
    generators["is_renewable"] = generators["energy_source"].isin(RENEWABLE_SOURCES)
    generators["renewable_capacity_mw"] = generators["capacity_mw"].where(
        generators["is_renewable"], 0.0
    )
    log(f"  {len(generators):,} operating generators (Status=OP)")

    plant_capacity = generators.groupby("plant_code", as_index=False).agg(
        operating_mw=("capacity_mw", "sum"),
        renewable_mw=("renewable_capacity_mw", "sum"),
        generator_count=("capacity_mw", "count"),
    )
    # Dominant fuel by capacity at each plant
    dominant = (
        generators.groupby(["plant_code", "energy_source"], as_index=False)["capacity_mw"]
        .sum()
        .sort_values(["plant_code", "capacity_mw"], ascending=[True, False])
        .drop_duplicates("plant_code")
        .rename(columns={"energy_source": "primary_fuel", "capacity_mw": "primary_fuel_mw"})
    )

    sites = plants.merge(plant_capacity, on="plant_code", how="inner")
    sites = sites.merge(
        dominant[["plant_code", "primary_fuel"]], on="plant_code", how="left"
    )
    sites = sites.rename(columns={"Plant Name": "plant_name", "State": "state"})
    sites["renewable_mw"] = sites["renewable_mw"].fillna(0.0)

    log(f"  {len(sites):,} plants with operating capacity")
    log(f"  Total operating MW: {sites['operating_mw'].sum():,.0f}")

    return gpd.GeoDataFrame(
        sites[
            [
                "plant_code",
                "plant_name",
                "state",
                "primary_fuel",
                "operating_mw",
                "renewable_mw",
                "generator_count",
            ]
        ],
        geometry=gpd.points_from_xy(sites["lon"], sites["lat"]),
        crs="EPSG:4326",
    )


def load_grid(grid_path: Path) -> gpd.GeoDataFrame:
    log(f"Loading H3 grid from {grid_path}...")
    grid = gpd.read_parquet(grid_path)
    if grid.index.name == "h3_index":
        grid = grid.reset_index()
    if "h3_index" not in grid.columns:
        raise ValueError(f"Grid missing h3_index: {grid.columns.tolist()}")

    points = gpd.GeoDataFrame(
        grid[["h3_index", "lat", "lon"]],
        geometry=gpd.points_from_xy(grid["lon"], grid["lat"]),
        crs="EPSG:4326",
    )
    log(f"  {len(points):,} hex cells")
    return points


def compute_nearest_plant(
    grid_pts: gpd.GeoDataFrame, plants: gpd.GeoDataFrame
) -> pd.DataFrame:
    log("Computing distance to nearest operating plant...")
    plants_m = plants.to_crs(epsg=3857)
    grid_m = grid_pts.to_crs(epsg=3857)

    nearest = gpd.sjoin_nearest(
        grid_m[["h3_index", "geometry"]],
        plants_m[
            ["plant_code", "operating_mw", "geometry"]
        ],
        how="left",
        distance_col="dist_nearest_plant_m",
    )
    nearest = nearest.rename(columns={"operating_mw": "nearest_plant_operating_mw"})
    nearest = nearest[
        ["h3_index", "dist_nearest_plant_m", "nearest_plant_operating_mw"]
    ].drop_duplicates(subset="h3_index", keep="first")
    nearest["dist_nearest_plant_km"] = nearest["dist_nearest_plant_m"] / 1000.0
    return nearest


def sum_capacity_within_radius(
    grid_pts: gpd.GeoDataFrame,
    plants: gpd.GeoDataFrame,
    radius_km: float,
    mw_col: str = "operating_mw",
) -> pd.Series:
    radius_m = radius_km * 1000.0
    log(f"  Summing {mw_col} within {radius_km:g} km...")
    plants_m = plants.to_crs(epsg=3857)
    grid_m = grid_pts.to_crs(epsg=3857)

    buffered = plants_m[["plant_code", mw_col, "geometry"]].copy()
    buffered["geometry"] = buffered.geometry.buffer(radius_m)

    joined = gpd.sjoin(
        grid_m[["h3_index", "geometry"]],
        buffered,
        how="left",
        predicate="within",
    )
    if joined.empty:
        return pd.Series(0.0, index=grid_pts["h3_index"])

    totals = joined.groupby("h3_index")[mw_col].sum()
    return totals.reindex(grid_pts["h3_index"], fill_value=0.0)


def count_plants_within_radius(
    grid_pts: gpd.GeoDataFrame,
    plants: gpd.GeoDataFrame,
    radius_km: float,
) -> pd.Series:
    radius_m = radius_km * 1000.0
    log(f"  Counting plants within {radius_km:g} km...")
    plants_m = plants.to_crs(epsg=3857)
    grid_m = grid_pts.to_crs(epsg=3857)

    buffered = plants_m[["plant_code", "geometry"]].copy()
    buffered["geometry"] = buffered.geometry.buffer(radius_m)

    joined = gpd.sjoin(
        grid_m[["h3_index", "geometry"]],
        buffered,
        how="left",
        predicate="within",
    )
    counts = joined.groupby("h3_index")["plant_code"].nunique()
    return counts.reindex(grid_pts["h3_index"], fill_value=0).astype(int)


def build_features(
    grid_pts: gpd.GeoDataFrame,
    plants: gpd.GeoDataFrame,
    radii_km: tuple[float, ...],
    colocation_m: float,
) -> pd.DataFrame:
    if len(radii_km) != 3:
        raise ValueError("Exactly three radii are required for output column names (5, 25, 50 km)")

    nearest = compute_nearest_plant(grid_pts, plants)

    log("Summing nearby operating capacity by radius...")
    cap_5 = sum_capacity_within_radius(grid_pts, plants, radii_km[0])
    cap_25 = sum_capacity_within_radius(grid_pts, plants, radii_km[1])
    cap_50 = sum_capacity_within_radius(grid_pts, plants, radii_km[2])
    renewable_25 = sum_capacity_within_radius(
        grid_pts, plants, radii_km[1], mw_col="renewable_mw"
    )
    plant_count_25 = count_plants_within_radius(grid_pts, plants, radii_km[1])

    out = grid_pts[["h3_index"]].merge(nearest, on="h3_index", how="left")
    out["operating_capacity_mw_5km"] = cap_5.values
    out["operating_capacity_mw_25km"] = cap_25.values
    out["operating_capacity_mw_50km"] = cap_50.values
    out["renewable_capacity_mw_25km"] = renewable_25.values
    out["plant_count_25km"] = plant_count_25.values
    out["colocated_with_power"] = out["dist_nearest_plant_m"] <= colocation_m
    out["source"] = "EIA_860_2024"

    for col in (
        "operating_capacity_mw_5km",
        "operating_capacity_mw_25km",
        "operating_capacity_mw_50km",
        "renewable_capacity_mw_25km",
    ):
        out[col] = out[col].fillna(0.0)
    out["plant_count_25km"] = out["plant_count_25km"].fillna(0).astype(int)
    out["colocated_with_power"] = out["colocated_with_power"].fillna(False)

    return out[OUTPUT_COLUMNS]


def print_summary(
    df: pd.DataFrame,
    radii_km: tuple[float, ...],
    colocation_m: float,
    output_path: Path,
) -> None:
    log("\n" + "=" * 60)
    log("Nearby generation capacity features complete")
    log("=" * 60)
    log(f"H3 cells:                         {len(df):,}")
    log(
        f"Cells colocated with power (≤{colocation_m / 1000:g} km): "
        f"{df['colocated_with_power'].sum():,}"
    )
    log("\nDistance to nearest plant (km):")
    log(f"  min:    {df['dist_nearest_plant_km'].min():,.2f}")
    log(f"  median: {df['dist_nearest_plant_km'].median():,.2f}")
    log(f"  max:    {df['dist_nearest_plant_km'].max():,.2f}")
    log(f"\nOperating capacity within {radii_km[1]:g} km (MW):")
    log(f"  min:    {df['operating_capacity_mw_25km'].min():,.1f}")
    log(f"  median: {df['operating_capacity_mw_25km'].median():,.1f}")
    log(f"  max:    {df['operating_capacity_mw_25km'].max():,.1f}")
    log(f"  > 0:    {(df['operating_capacity_mw_25km'] > 0).sum():,} cells")
    if output_path.exists():
        log(f"\nOutput: {output_path} ({_format_bytes(output_path.stat().st_size)})")


def parse_radii_km(value: str) -> tuple[float, ...]:
    parts = sorted(float(x.strip()) for x in value.split(",") if x.strip())
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Provide exactly three comma-separated radii in km, e.g. 5,25,50")
    if any(r <= 0 for r in parts):
        raise argparse.ArgumentTypeError("Radii must be positive")
    return tuple(parts)


def compute_nearby_generation(
    eia_dir: Path | None = None,
    grid_path: Path | None = None,
    output_path: Path | None = None,
    radii_km: tuple[float, ...] = DEFAULT_RADII_KM,
    colocation_m: float = DEFAULT_COLOCATION_M,
) -> pd.DataFrame:
    eia_dir = eia_dir or EIA_DIR
    resolution = DEFAULT_RESOLUTION
    if grid_path is None:
        grid_path = PROCESSED_DIR / f"h3_grid_res{resolution}.parquet"
    if output_path is None:
        output_path = PROCESSED_DIR / f"features_nearby_generation_res{resolution}.parquet"

    if not eia_dir.exists():
        raise FileNotFoundError(f"EIA-860 directory not found: {eia_dir}")
    if not grid_path.exists():
        raise FileNotFoundError(
            f"H3 grid not found: {grid_path}. "
            f"Run: python scripts/build_grid.py --resolution {resolution}"
        )

    plants = load_plant_sites(eia_dir)
    if plants.empty:
        raise ValueError("No plant locations with operating capacity")

    grid_pts = load_grid(grid_path)
    result = build_features(grid_pts, plants, radii_km, colocation_m)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result.to_parquet(output_path, index=False)
    except OSError as exc:
        raise OSError(f"Failed to write {output_path}: {exc}") from exc

    print_summary(result, radii_km, colocation_m, output_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute nearby operating generation capacity from EIA-860 per H3 cell"
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
        default=PROCESSED_DIR / f"features_nearby_generation_res{DEFAULT_RESOLUTION}.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--radii-km",
        type=parse_radii_km,
        metavar="KM,KM,KM",
        help="Three comma-separated search radii in km, default: 5,25,50",
    )
    parser.add_argument(
        "--colocation-m",
        type=float,
        default=DEFAULT_COLOCATION_M,
        help="Distance threshold (m) for colocated_with_power flag (default: 2000)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        compute_nearby_generation(
            eia_dir=args.eia_dir,
            grid_path=args.grid,
            output_path=args.output,
            radii_km=args.radii_km or DEFAULT_RADII_KM,
            colocation_m=args.colocation_m,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        log(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
