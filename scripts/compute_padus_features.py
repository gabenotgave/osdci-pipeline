#!/usr/bin/env python3
"""Compute PAD-US protected area exclusion features per H3 cell."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_RESOLUTION = 6
DEFAULT_GDB = RAW_DIR / "PADUS4_1Geodatabase.gdb"
GAP_CACHE = RAW_DIR / "padus_gap12_5070.parquet"
WORK_CRS = "EPSG:5070"

GAP_FIELD = "GAP_Sts"
NAME_FIELD = "Unit_Nm"
HARD_EXCLUSION_STATUSES = frozenset({"1", "2"})

OUTPUT_COLUMNS = [
    "h3_index",
    "in_protected_area",
    "protected_area_pct",
    "protected_area_name",
    "gap_status_min",
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


def read_gdf_parquet_safe(path: Path) -> gpd.GeoDataFrame:
    with path.open("rb") as handle:
        return gpd.read_parquet(handle)


def read_grid_parquet_safe(grid_path: Path) -> gpd.GeoDataFrame:
    with grid_path.open("rb") as handle:
        return gpd.read_parquet(handle)


def list_gdb_layers(gdb_path: Path) -> list[str]:
    import fiona

    log(f"Listing layers in {gdb_path.name}...")
    layers = fiona.listlayers(gdb_path)
    log("Available layers:")
    for layer in layers:
        log(f"  {layer}")
    return layers


def resolve_layer_name(layers: list[str], layer_override: str | None) -> str:
    if layer_override:
        if layer_override not in layers:
            raise ValueError(
                f"Layer {layer_override!r} not found. Available: {', '.join(layers)}"
            )
        return layer_override

    combined = [layer for layer in layers if "combined" in layer.lower()]
    if combined:
        return combined[0]
    raise FileNotFoundError(
        "No layer with 'combined' in the name was found. "
        "Available layers:\n  "
        + "\n  ".join(layers)
        + "\nRe-run with --layer <name> to specify the correct layer."
    )


def normalize_gap_status(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def count_layer_rows(gdb_path: Path, layer_name: str) -> int:
    import fiona

    with fiona.open(gdb_path, layer=layer_name) as src:
        return len(src)


def load_padus_gap12(
    gdb_path: Path,
    layer_name: str,
    cache_path: Path,
    refresh_cache: bool,
) -> tuple[gpd.GeoDataFrame, int]:
    total_rows = count_layer_rows(gdb_path, layer_name)

    if cache_path.exists() and not refresh_cache:
        log(f"Loading cached GAP 1/2 polygons from {cache_path.name}...")
        protected = read_gdf_parquet_safe(cache_path)
        if protected.crs is None:
            protected = protected.set_crs(WORK_CRS)
        log(f"  {len(protected):,} polygons (cached)")
        return protected, total_rows

    log(f"Loading PAD-US layer {layer_name!r} (GAP Status 1 & 2 filter)...")
    log("  This may take several minutes on first run...")
    protected = gpd.read_file(
        gdb_path,
        layer=layer_name,
        where=f"{GAP_FIELD} IN ('1', '2')",
    )
    log(
        f"  Loaded {len(protected):,} / {total_rows:,} polygons "
        f"({100.0 * len(protected) / total_rows:.1f}%) with GAP Status 1 or 2"
    )
    log(f"  CRS: {protected.crs}")
    log(f"  Columns: {', '.join(protected.columns)}")

    if GAP_FIELD not in protected.columns:
        raise ValueError(f"Expected column {GAP_FIELD!r} not found in layer")
    if NAME_FIELD not in protected.columns:
        raise ValueError(f"Expected column {NAME_FIELD!r} not found in layer")

    protected[GAP_FIELD] = normalize_gap_status(protected[GAP_FIELD])
    protected = protected[protected[GAP_FIELD].isin(HARD_EXCLUSION_STATUSES)].copy()
    protected = protected[~protected.geometry.isna() & ~protected.geometry.is_empty]
    protected = protected[protected.geometry.is_valid].copy()
    if protected.empty:
        raise ValueError("No valid GAP Status 1 or 2 polygons after filtering")

    log(f"  {len(protected):,} valid polygons after geometry filter")
    protected = protected.to_crs(WORK_CRS)
    protected = protected[[GAP_FIELD, NAME_FIELD, "geometry"]].copy()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_gdf_parquet_safe(protected, cache_path)
    log(f"  Cached to {cache_path.name}")
    return protected, total_rows


def load_grid(grid_path: Path, sample: int | None) -> gpd.GeoDataFrame:
    log(f"Loading H3 grid from {grid_path}...")
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
    grid = grid[["h3_index", "geometry"]].to_crs(WORK_CRS)
    grid["cell_area_m2"] = grid.geometry.area
    return grid


def compute_overlap_features(
    grid: gpd.GeoDataFrame,
    protected: gpd.GeoDataFrame,
) -> pd.DataFrame:
    log("Pass 1: spatial join (intersects) to find candidate cells...")
    candidates = gpd.sjoin(
        grid[["h3_index", "geometry", "cell_area_m2"]],
        protected[[GAP_FIELD, NAME_FIELD, "geometry"]],
        how="inner",
        predicate="intersects",
    )
    candidate_ids = candidates["h3_index"].unique()
    log(f"  {len(candidate_ids):,} cells intersect protected areas")

    if len(candidate_ids) == 0:
        return _empty_features(grid)

    log("Pass 2: overlay (intersection) on candidate cells for area fractions...")
    grid_candidates = grid[grid["h3_index"].isin(candidate_ids)].copy()
    intersections = gpd.overlay(
        grid_candidates,
        protected,
        how="intersection",
        keep_geom_type=False,
    )
    intersections["piece_area_m2"] = intersections.geometry.area

    grouped = intersections.groupby("h3_index", as_index=False).agg(
        protected_area_m2=("piece_area_m2", "sum"),
    )
    grid_areas = grid[["h3_index", "cell_area_m2"]]
    grouped = grouped.merge(grid_areas, on="h3_index", how="left")
    grouped["protected_area_pct"] = (
        grouped["protected_area_m2"] / grouped["cell_area_m2"]
    ).clip(upper=1.0)

    largest = (
        intersections.sort_values("piece_area_m2", ascending=False)
        .drop_duplicates("h3_index")[["h3_index", NAME_FIELD]]
        .rename(columns={NAME_FIELD: "protected_area_name"})
    )

    gap_rank = intersections.copy()
    gap_rank["gap_rank"] = gap_rank[GAP_FIELD].map({"1": 1, "2": 2})
    gap_min = (
        gap_rank.sort_values("gap_rank")
        .drop_duplicates("h3_index")[["h3_index", GAP_FIELD]]
        .rename(columns={GAP_FIELD: "gap_status_min"})
    )

    overlap = grouped.merge(largest, on="h3_index", how="left")
    overlap = overlap.merge(gap_min, on="h3_index", how="left")
    overlap["in_protected_area"] = (overlap["protected_area_pct"] > 0).astype(int)

    out = grid[["h3_index"]].merge(
        overlap[
            [
                "h3_index",
                "in_protected_area",
                "protected_area_pct",
                "protected_area_name",
                "gap_status_min",
            ]
        ],
        on="h3_index",
        how="left",
    )
    out["in_protected_area"] = out["in_protected_area"].fillna(0).astype(int)
    out["protected_area_pct"] = out["protected_area_pct"].fillna(0.0)
    return out[OUTPUT_COLUMNS]


def _empty_features(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "h3_index": grid["h3_index"],
            "in_protected_area": 0,
            "protected_area_pct": 0.0,
            "protected_area_name": pd.NA,
            "gap_status_min": pd.NA,
        }
    )


def print_summary(
    df: pd.DataFrame,
    total_polygons: int,
    filtered_polygons: int,
    output_path: Path,
) -> None:
    protected_cells = df["in_protected_area"] == 1
    log("\n" + "=" * 60)
    log("PAD-US protected area features complete")
    log("=" * 60)
    log(f"Total PAD-US polygons loaded:     {total_polygons:,}")
    log(f"Polygons after GAP 1/2 filter:    {filtered_polygons:,}")
    log(f"Total H3 cells processed:         {len(df):,}")
    log(
        f"Cells with in_protected_area=1:   {protected_cells.sum():,} "
        f"({100.0 * protected_cells.sum() / len(df):.2f}%)"
    )

    if protected_cells.any():
        pct = df.loc[protected_cells, "protected_area_pct"]
        log(f"\nMean protected_area_pct (overlap): {pct.mean():.4f}")

        log("\nTop 10 protected_area_name by cell count:")
        top_names = (
            df.loc[protected_cells, "protected_area_name"]
            .value_counts()
            .head(10)
        )
        for name, count in top_names.items():
            log(f"  {name}: {count:,}")

        gap_counts = df.loc[protected_cells, "gap_status_min"].value_counts()
        log("\ngap_status_min breakdown:")
        for status in ("1", "2"):
            log(f"  {status}: {gap_counts.get(status, 0):,}")

    if output_path.exists():
        log(f"\nOutput: {output_path} ({_format_bytes(output_path.stat().st_size)})")


def compute_padus_features(
    gdb_path: Path | None = None,
    grid_path: Path | None = None,
    output_path: Path | None = None,
    layer_name: str | None = None,
    cache_path: Path | None = None,
    sample: int | None = None,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    gdb_path = gdb_path or DEFAULT_GDB
    resolution = DEFAULT_RESOLUTION
    if grid_path is None:
        grid_path = PROCESSED_DIR / f"h3_grid_res{resolution}.parquet"
    if output_path is None:
        output_path = PROCESSED_DIR / f"features_padus_res{resolution}.parquet"
    if cache_path is None:
        cache_path = GAP_CACHE

    if not gdb_path.exists():
        raise FileNotFoundError(f"PAD-US geodatabase not found: {gdb_path}")
    if not grid_path.exists():
        raise FileNotFoundError(
            f"H3 grid not found: {grid_path}. "
            f"Run: python scripts/build_grid.py --resolution {resolution}"
        )

    grid = load_grid(grid_path, sample)
    log(f"H3 grid reprojected to {WORK_CRS}")

    layers = list_gdb_layers(gdb_path)
    layer = resolve_layer_name(layers, layer_name)
    log(f"Using layer: {layer}")

    protected, total_count = load_padus_gap12(gdb_path, layer, cache_path, refresh_cache)

    result = compute_overlap_features(grid, protected)

    log(f"Writing {output_path.name}...")
    write_parquet_safe(result, output_path)
    log("  Parquet write complete")

    print_summary(result, total_count, len(protected), output_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PAD-US GAP 1/2 protected area exclusion features per H3 cell"
    )
    parser.add_argument(
        "--gdb",
        type=Path,
        default=DEFAULT_GDB,
        help="PAD-US file geodatabase path",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Override auto-detected Combined layer name",
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
        default=PROCESSED_DIR / f"features_padus_res{DEFAULT_RESOLUTION}.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=GAP_CACHE,
        help="Cached GAP 1/2 polygons parquet (EPSG:5070)",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Re-read GDB and rebuild GAP 1/2 cache",
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
        compute_padus_features(
            gdb_path=args.gdb,
            grid_path=args.grid,
            output_path=args.output,
            layer_name=args.layer,
            cache_path=args.cache,
            sample=args.sample,
            refresh_cache=args.refresh_cache,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        log(f"Error: {exc}")
        return 1
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
