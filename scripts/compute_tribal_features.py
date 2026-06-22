#!/usr/bin/env python3
"""Compute tribal land exclusion flags from Census TIGER AIANNH polygons."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_RESOLUTION = 6
DEFAULT_SHAPEFILE = RAW_DIR / "tl_2024_us_aiannh" / "tl_2024_us_aiannh.shp"
WORK_CRS = "EPSG:5070"

KEEP_TYPES = [
    "American Indian Reservation",
    "Off-Reservation Trust Land",
    "Joint Use Area",
    "Hawaiian Home Land",
]

# Census TIGER AIANNH LSAD → entity type (2024 vintage)
LSAD_ENTITY_TYPE = {
    "78": "Hawaiian Home Land",
    "81": "American Indian Reservation",
    "82": "Joint Use Area",
    "83": "Joint Use Area",
    "84": "American Indian Reservation",
    "85": "American Indian Reservation",
    "86": "American Indian Reservation",
    "89": "Off-Reservation Trust Land",
    "90": "Joint Use Area",
    "94": "American Indian Reservation",
    "95": "American Indian Reservation",
    "96": "American Indian Reservation",
    "97": "American Indian Reservation",
    "98": "American Indian Reservation",
    "99": "American Indian Reservation",
    "9C": "American Indian Reservation",
    "9D": "American Indian Reservation",
    "00": "American Indian Reservation",
    "OT": "Off-Reservation Trust Land",
}

OUTPUT_COLUMNS = ["h3_index", "in_tribal_land", "tribal_name"]


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


def read_grid_parquet_safe(grid_path: Path) -> gpd.GeoDataFrame:
    with grid_path.open("rb") as handle:
        return gpd.read_parquet(handle)


def derive_entity_type(df: gpd.GeoDataFrame) -> pd.Series:
    lsad_type = df["LSAD"].map(LSAD_ENTITY_TYPE)
    name = df["NAMELSAD"].fillna("").astype(str)
    fallback = pd.Series(pd.NA, index=df.index, dtype="object")
    reservation_mask = name.str.contains(
        r"Reservation|Pueblo|Rancheria|Colony|Indian Community|Indian Village",
        case=False,
        regex=True,
    ) & ~name.str.contains(
        r"ANVSA|OTSA|SDTSA|TDSA|statistical",
        case=False,
        regex=True,
    )
    fallback.loc[reservation_mask] = "American Indian Reservation"
    trust_mask = name.str.contains("Trust Land", case=False, regex=False) & ~name.str.contains(
        "joint-use", case=False, regex=False
    )
    fallback.loc[trust_mask] = "Off-Reservation Trust Land"
    hawaiian_mask = name.str.contains("Hawaiian Home Land", case=False, regex=False)
    fallback.loc[hawaiian_mask] = "Hawaiian Home Land"
    joint_mask = name.str.contains("joint-use", case=False, regex=False)
    fallback.loc[joint_mask] = "Joint Use Area"
    return lsad_type.fillna(fallback)


def load_aiannh(shapefile_path: Path) -> tuple[gpd.GeoDataFrame, int]:
    log(f"Loading {shapefile_path}...")
    aiannh = gpd.read_file(shapefile_path)
    log(f"Loaded {len(aiannh)} AIANNH polygons")
    log(f"CRS: {aiannh.crs}")
    log(f"Columns: {aiannh.columns.tolist()}")

    aiannh["entity_type"] = derive_entity_type(aiannh)
    log("\nEntity type breakdown (all polygons):")
    for entity_type, count in aiannh["entity_type"].value_counts(dropna=False).items():
        log(f"  {entity_type}: {count}")

    type_col = "entity_type"
    if "AIANTYPE" in aiannh.columns:
        type_col = "AIANTYPE"
        log(f"\nAIANTYPE value_counts:\n{aiannh[type_col].value_counts()}")
    elif "LSAD" in aiannh.columns:
        log(f"\nLSAD value_counts:\n{aiannh['LSAD'].value_counts()}")

    total = len(aiannh)
    aiannh = aiannh[aiannh["entity_type"].isin(KEEP_TYPES)].copy()
    log(f"\nKept {len(aiannh)} of {total} polygons after type filter")

    aiannh = aiannh[~aiannh.geometry.isna() & ~aiannh.geometry.is_empty]
    aiannh = aiannh[aiannh.geometry.is_valid].copy()
    log(f"  {len(aiannh)} polygons after geometry filter")
    return aiannh, total


def load_grid(grid_path: Path, sample: int | None) -> gpd.GeoDataFrame:
    log(f"Loading H3 grid from {grid_path.name}...")
    grid = read_grid_parquet_safe(grid_path)
    if grid.index.name == "h3_index":
        grid = grid.reset_index()
    if grid.crs is None:
        grid = grid.set_crs("EPSG:4326")
    if sample is not None:
        grid = grid.head(sample).copy()
        log(f"  Sample mode: using first {len(grid):,} cells")
    else:
        log(f"  {len(grid):,} hex cells")
    return grid


def centroid_within_join(
    grid: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    attr_cols: list[str],
) -> gpd.GeoDataFrame:
    grid_proj = grid.to_crs(WORK_CRS)
    poly_proj = polygons.to_crs(WORK_CRS)

    grid_points = grid_proj[["h3_index", "geometry"]].copy()
    grid_points["geometry"] = grid_proj.geometry.centroid

    log("Spatial join: H3 centroids within polygons...")
    joined = gpd.sjoin(
        grid_points,
        poly_proj[["geometry", *attr_cols]],
        how="left",
        predicate="within",
    )
    return joined


def compute_tribal_features(
    shapefile_path: Path | None = None,
    grid_path: Path | None = None,
    output_path: Path | None = None,
    sample: int | None = None,
) -> pd.DataFrame:
    resolution = DEFAULT_RESOLUTION
    if shapefile_path is None:
        shapefile_path = DEFAULT_SHAPEFILE
    if grid_path is None:
        grid_path = PROCESSED_DIR / f"h3_grid_res{resolution}.parquet"
    if output_path is None:
        output_path = PROCESSED_DIR / f"features_tribal_res{resolution}.parquet"

    if not shapefile_path.exists():
        raise FileNotFoundError(f"AIANNH shapefile not found: {shapefile_path}")
    if not grid_path.exists():
        raise FileNotFoundError(
            f"H3 grid not found: {grid_path}. "
            f"Run: python scripts/build_grid.py --resolution {resolution}"
        )

    aiannh, total_loaded = load_aiannh(shapefile_path)
    grid = load_grid(grid_path, sample)

    joined = centroid_within_join(grid, aiannh, ["NAMELSAD"])
    tribal_cells = set(joined.loc[joined["index_right"].notna(), "h3_index"])

    out = grid[["h3_index"]].copy()
    out["in_tribal_land"] = out["h3_index"].isin(tribal_cells).astype(int)

    name_map = (
        joined.loc[joined["index_right"].notna(), ["h3_index", "NAMELSAD"]]
        .drop_duplicates("h3_index", keep="first")
        .set_index("h3_index")["NAMELSAD"]
    )
    out["tribal_name"] = out["h3_index"].map(name_map)

    log(f"Writing {output_path.name}...")
    write_parquet_safe(out[OUTPUT_COLUMNS], output_path)

    flagged = out["in_tribal_land"] == 1
    log("\n" + "=" * 60)
    log("Tribal land features complete")
    log("=" * 60)
    log(f"Total AIANNH polygons loaded:   {total_loaded:,}")
    log(f"Polygons after type filter:     {len(aiannh):,}")
    log(f"Total H3 cells processed:         {len(out):,}")
    log(
        f"Cells with in_tribal_land=1:      {flagged.sum():,} "
        f"({100.0 * flagged.sum() / len(out):.2f}%)"
    )
    if flagged.any():
        log("\nTop 10 tribal entities by cell count:")
        top = (
            out.loc[flagged, "tribal_name"]
            .value_counts()
            .head(10)
        )
        for name, count in top.items():
            log(f"  {name}: {count:,}")
    if output_path.exists():
        log(f"\nOutput: {output_path} ({_format_bytes(output_path.stat().st_size)})")
    return out[OUTPUT_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute tribal land exclusion flags per H3 cell from Census AIANNH"
    )
    parser.add_argument(
        "--shapefile",
        type=Path,
        default=DEFAULT_SHAPEFILE,
        help="Census TIGER AIANNH shapefile path",
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
        default=PROCESSED_DIR / f"features_tribal_res{DEFAULT_RESOLUTION}.parquet",
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
    started = time.perf_counter()
    try:
        compute_tribal_features(
            shapefile_path=args.shapefile,
            grid_path=args.grid,
            output_path=args.output,
            sample=args.sample,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        log(f"Error: {exc}")
        return 1
    elapsed = time.perf_counter() - started
    log(f"\nRuntime: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
