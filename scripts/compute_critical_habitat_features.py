#!/usr/bin/env python3
"""Compute USFWS ESA critical habitat exclusion flags per H3 cell."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry import shape as shapely_shape
from shapely.ops import unary_union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_RESOLUTION = 6
CACHE_PATH = RAW_DIR / "critical_habitat_usfws.parquet"
FEATURE_SERVER = (
    "https://services.arcgis.com/QVENGdaPbd4LUkLV/arcgis/rest/services/"
    "USFWS_Critical_Habitat/FeatureServer/0/query"
)
WORK_CRS = "EPSG:5070"

OUTPUT_COLUMNS = [
    "h3_index",
    "in_critical_habitat",
    "critical_habitat_species_count",
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


def normalize_crithab_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    rename = {}
    for col in gdf.columns:
        upper = col.upper()
        if upper in {"COMNAME", "SCINAME", "STATUS", "EFFECTDATE"} and col != upper:
            rename[col] = upper
    if rename:
        gdf = gdf.rename(columns=rename)
    return gdf


def esri_geometry_to_shapely(geom: dict | None):
    if not geom:
        return None
    if "x" in geom and "y" in geom:
        return Point(geom["x"], geom["y"])
    if "rings" in geom:
        rings = geom["rings"]
        polygons = [Polygon(ring) for ring in rings if len(ring) >= 3]
        if not polygons:
            return None
        if len(polygons) == 1:
            return polygons[0]
        return unary_union(polygons)
    if geom.get("type"):
        return shapely_shape(geom)
    return None


def download_critical_habitat(cache_path: Path) -> gpd.GeoDataFrame:
    records: list[dict] = []
    offset = 0
    page_size = 100

    while True:
        last_error: Exception | None = None
        data: dict | None = None
        for attempt in range(3):
            try:
                response = requests.get(
                    FEATURE_SERVER,
                    params={
                        "where": "1=1",
                        "outFields": "COMNAME,SCINAME,STATUS,EFFECTDATE",
                        "returnGeometry": "true",
                        "outSR": "4326",
                        "f": "json",
                        "resultOffset": offset,
                        "resultRecordCount": page_size,
                    },
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()
                if "error" in data:
                    raise RuntimeError(f"FeatureServer error: {data['error']}")
                break
            except (requests.RequestException, RuntimeError) as exc:
                last_error = exc
                if attempt < 2:
                    wait = 5 * (attempt + 1)
                    log(f"  Request failed ({exc}); retrying in {wait}s...")
                    time.sleep(wait)
        if data is None:
            raise RuntimeError(f"FeatureServer download failed: {last_error}")

        features = data.get("features", [])
        if not features:
            break

        for feature in features:
            try:
                geom = esri_geometry_to_shapely(feature.get("geometry"))
                if geom is None or geom.is_empty:
                    continue
                rec = dict(feature["attributes"])
                rec["geometry"] = geom
                records.append(rec)
            except Exception:
                pass

        offset += len(features)
        log(f"  Downloaded {offset} critical habitat polygons...")

        if len(features) < page_size:
            break

    if not records:
        raise RuntimeError("No valid critical habitat geometries downloaded")

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf = normalize_crithab_columns(gdf)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_gdf_parquet_safe(gdf, cache_path)
    log(f"  Cached {len(gdf)} polygons → {cache_path}")
    return gdf


def load_critical_habitat(cache_path: Path, refresh_cache: bool) -> gpd.GeoDataFrame:
    if cache_path.exists() and not refresh_cache:
        log(f"Loading critical habitat from cache ({cache_path.name})...")
        crithab = read_gdf_parquet_safe(cache_path)
        crithab = normalize_crithab_columns(crithab)
        log(f"  {len(crithab):,} polygons (cached)")
        return crithab

    log("Downloading critical habitat from USFWS FeatureServer...")
    return download_critical_habitat(cache_path)


def filter_final_crithab(crithab: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    if "STATUS" not in crithab.columns:
        raise ValueError("Expected STATUS column in critical habitat data")

    status = crithab["STATUS"].astype(str).str.strip().str.title()
    counts = {
        "total": len(crithab),
        "final": int((status == "Final").sum()),
        "proposed": int((status == "Proposed").sum()),
    }
    log(
        f"{counts['final']} final / {counts['proposed']} proposed / "
        f"{counts['total']} total polygons"
    )

    final = crithab[status == "Final"].copy()
    final = final[~final.geometry.isna() & ~final.geometry.is_empty]
    final = final[final.geometry.is_valid].copy()
    log(f"  {len(final):,} final polygons after geometry filter")
    return final, counts


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

    log("Spatial join: H3 centroids within critical habitat polygons...")
    joined = gpd.sjoin(
        grid_points,
        poly_proj[["geometry", *attr_cols]],
        how="left",
        predicate="within",
    )
    return joined


def compute_critical_habitat_features(
    grid_path: Path | None = None,
    output_path: Path | None = None,
    cache_path: Path | None = None,
    sample: int | None = None,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    resolution = DEFAULT_RESOLUTION
    if grid_path is None:
        grid_path = PROCESSED_DIR / f"h3_grid_res{resolution}.parquet"
    if output_path is None:
        output_path = (
            PROCESSED_DIR / f"features_critical_habitat_res{resolution}.parquet"
        )
    if cache_path is None:
        cache_path = CACHE_PATH

    if not grid_path.exists():
        raise FileNotFoundError(
            f"H3 grid not found: {grid_path}. "
            f"Run: python scripts/build_grid.py --resolution {resolution}"
        )

    crithab = load_critical_habitat(cache_path, refresh_cache)
    log(f"Critical habitat polygons: {len(crithab):,}")

    final, status_counts = filter_final_crithab(crithab)
    grid = load_grid(grid_path, sample)

    joined = centroid_within_join(grid, final, ["COMNAME"])
    crithab_cells = set(joined.loc[joined["index_right"].notna(), "h3_index"])

    out = grid[["h3_index"]].copy()
    out["in_critical_habitat"] = out["h3_index"].isin(crithab_cells).astype(int)

    species_count = (
        joined.loc[joined["index_right"].notna()]
        .groupby("h3_index")["COMNAME"]
        .nunique()
    )
    out["critical_habitat_species_count"] = (
        out["h3_index"].map(species_count).fillna(0).astype(int)
    )

    log(f"Writing {output_path.name}...")
    write_parquet_safe(out[OUTPUT_COLUMNS], output_path)

    flagged = out["in_critical_habitat"] == 1
    log("\n" + "=" * 60)
    log("Critical habitat features complete")
    log("=" * 60)
    log(f"Total polygons loaded:            {status_counts['total']:,}")
    log(
        f"Final / proposed:               "
        f"{status_counts['final']:,} / {status_counts['proposed']:,}"
    )
    log(f"Total H3 cells processed:         {len(out):,}")
    log(
        f"Cells with in_critical_habitat=1: {flagged.sum():,} "
        f"({100.0 * flagged.sum() / len(out):.2f}%)"
    )
    if flagged.any():
        mean_species = out.loc[flagged, "critical_habitat_species_count"].mean()
        log(
            f"Mean critical_habitat_species_count (flagged cells): "
            f"{mean_species:.2f}"
        )
        log("\nTop 10 species by cell count:")
        top_species = (
            joined.loc[joined["index_right"].notna(), "COMNAME"]
            .value_counts()
            .head(10)
        )
        for species, count in top_species.items():
            log(f"  {species}: {count:,}")
    if output_path.exists():
        log(f"\nOutput: {output_path} ({_format_bytes(output_path.stat().st_size)})")
    return out[OUTPUT_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute USFWS ESA critical habitat exclusion flags per H3 cell"
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
        default=PROCESSED_DIR
        / f"features_critical_habitat_res{DEFAULT_RESOLUTION}.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=CACHE_PATH,
        help="Cached critical habitat parquet",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force re-download from USFWS FeatureServer",
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
        compute_critical_habitat_features(
            grid_path=args.grid,
            output_path=args.output,
            cache_path=args.cache,
            sample=args.sample,
            refresh_cache=args.refresh_cache,
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        log(f"Error: {exc}")
        return 1
    elapsed = time.perf_counter() - started
    log(f"\nRuntime: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
