#!/usr/bin/env python3
"""Compute metro proximity and cloud region distance features per H3 cell."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pygris
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from osdci.cloud_regions import CITIES_100K, CITIES_50K, CLOUD_REGIONS

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_RESOLUTION = 6
CENSUS_YEAR = 2023
CLOUD_COUNT_RADIUS_KM = 500.0

STATE_FIPS = [
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13", "15", "16",
    "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
    "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42",
    "44", "45", "46", "47", "48", "49", "50", "51", "53", "54", "55", "56",
]

PLACES_CACHE = RAW_DIR / "census_places_2023.parquet"
CLOUD_REGIONS_CACHE = RAW_DIR / "cloud_regions.parquet"

OUTPUT_COLUMNS = [
    "h3_index",
    "dist_nearest_city_50k_km",
    "nearest_city_50k_name",
    "dist_nearest_city_100k_km",
    "nearest_city_100k_name",
    "dist_nearest_cloud_km",
    "nearest_cloud_provider",
    "nearest_cloud_region",
    "nearest_cloud_region_name",
    "cloud_regions_within_500km",
]


def log(message: str) -> None:
    print(message, flush=True)


def _format_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.1f} GB"


def normalize_place_name(name: object) -> str:
    if pd.isna(name):
        return ""
    text = str(name).strip()
    if "(" in text:
        text = text.split("(", 1)[0].strip()
    text = re.sub(r"\s+city$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+town$", "", text, flags=re.IGNORECASE)
    return text.strip()


def expand_city_names(cities: frozenset[str]) -> frozenset[str]:
    expanded: set[str] = set(cities)
    for city in cities:
        expanded.add(city)
        expanded.add(normalize_place_name(city))
        if not city.lower().endswith(" city"):
            expanded.add(f"{city} city")
    return frozenset(x for x in expanded if x)


CITIES_50K_MATCH = expand_city_names(CITIES_50K)
CITIES_100K_MATCH = expand_city_names(CITIES_100K)


def place_in_city_list(place_name: object, city_names: frozenset[str]) -> bool:
    if pd.isna(place_name):
        return False
    name = str(place_name).strip()
    norm = normalize_place_name(name)
    return name in city_names or norm in city_names


def download_census_places(
    year: int = CENSUS_YEAR,
    places_cache: Path | None = None,
    force: bool = False,
) -> gpd.GeoDataFrame:
    places_cache = places_cache or PLACES_CACHE
    if places_cache.exists() and not force:
        log(f"Loading Census places from {places_cache.name}...")
        places = gpd.read_parquet(places_cache)
        if places.crs is None:
            places = places.set_crs("EPSG:4326")
        return places

    log(f"Downloading Census places for year {year} ({len(STATE_FIPS)} states)...")
    frames: list[gpd.GeoDataFrame] = []
    failed: list[str] = []

    for i, fips in enumerate(STATE_FIPS, start=1):
        try:
            state_places = pygris.places(state=fips, year=year, cache=True)
            if state_places is not None and len(state_places):
                frames.append(state_places)
        except Exception as exc:
            failed.append(fips)
            log(f"  Warning: skipping state FIPS {fips}: {exc}")
        if i % 10 == 0:
            log(f"  Downloaded places for {i}/{len(STATE_FIPS)} states...")

    if not frames:
        raise ValueError("No Census places downloaded")

    places = pd.concat(frames, ignore_index=True)
    places = gpd.GeoDataFrame(places, geometry="geometry", crs=frames[0].crs)
    places_cache.parent.mkdir(parents=True, exist_ok=True)
    places.to_parquet(places_cache)
    log(f"  Saved {len(places):,} places to {places_cache}")
    if failed:
        log(f"  Warning: {len(failed)} states skipped: {', '.join(failed)}")
    return places


def filter_places_by_population(
    places: gpd.GeoDataFrame, city_names: frozenset[str]
) -> gpd.GeoDataFrame:
    mask = places["NAME"].map(lambda n: place_in_city_list(n, city_names))
    return places.loc[mask].copy()


def places_to_centroid_gdf(places: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    projected = places.to_crs(epsg=3857)
    centroids = projected.geometry.centroid
    out = places[["NAME"]].copy()
    out["geometry"] = centroids
    return gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:3857")


def load_city_centroids(
    places: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, int, int]:
    places_50k = filter_places_by_population(places, CITIES_50K_MATCH)
    places_100k = filter_places_by_population(places, CITIES_100K_MATCH)
    return (
        places_to_centroid_gdf(places_50k),
        places_to_centroid_gdf(places_100k),
        len(places_50k),
        len(places_100k),
    )


def load_cloud_regions_gdf(cache_path: Path | None = None, force: bool = False) -> gpd.GeoDataFrame:
    cache_path = cache_path or CLOUD_REGIONS_CACHE
    if cache_path.exists() and not force:
        log(f"Loading cloud regions from {cache_path.name}...")
        regions = gpd.read_parquet(cache_path)
        if regions.crs is None:
            regions = regions.set_crs("EPSG:4326")
        return regions.to_crs(epsg=3857)

    log("Building cloud regions GeoDataFrame...")
    df = pd.DataFrame(CLOUD_REGIONS)
    regions = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    regions.to_parquet(cache_path, index=False)
    log(f"  Saved {len(regions):,} cloud regions to {cache_path.name}")
    return regions


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
    ).to_crs(epsg=3857)
    log(f"  {len(points):,} hex cells")
    return points


def nearest_with_name(
    grid_pts: gpd.GeoDataFrame,
    sites: gpd.GeoDataFrame,
    distance_col: str,
    name_col: str = "NAME",
) -> pd.DataFrame:
    if sites.empty:
        return pd.DataFrame(
            {
                "h3_index": grid_pts["h3_index"],
                distance_col: pd.NA,
                name_col: pd.NA,
            }
        )

    joined = gpd.sjoin_nearest(
        grid_pts[["h3_index", "geometry"]],
        sites[[name_col, "geometry"]],
        how="left",
        distance_col=distance_col,
    )
    joined = joined.rename(columns={name_col: name_col})
    out = joined[["h3_index", distance_col, name_col]].drop_duplicates(
        subset="h3_index", keep="first"
    )
    out[distance_col] = out[distance_col] / 1000.0
    return out


def count_sites_within_radius(
    grid_pts: gpd.GeoDataFrame,
    sites: gpd.GeoDataFrame,
    radius_km: float,
    id_col: str | None = None,
) -> pd.Series:
    radius_m = radius_km * 1000.0
    buffered = sites.copy()
    buffered["geometry"] = buffered.geometry.buffer(radius_m)

    joined = gpd.sjoin(
        grid_pts[["h3_index", "geometry"]],
        buffered,
        how="left",
        predicate="within",
    )
    if joined.empty:
        return pd.Series(0, index=grid_pts["h3_index"], dtype=int)

    if id_col and id_col in joined.columns:
        counts = joined.groupby("h3_index")[id_col].nunique()
    else:
        counts = joined.groupby("h3_index").size()
    return counts.reindex(grid_pts["h3_index"], fill_value=0).astype(int)


def nearest_cloud(
    grid_pts: gpd.GeoDataFrame, cloud: gpd.GeoDataFrame
) -> pd.DataFrame:
    joined = gpd.sjoin_nearest(
        grid_pts[["h3_index", "geometry"]],
        cloud[
            [
                "provider",
                "region",
                "name",
                "geometry",
            ]
        ],
        how="left",
        distance_col="dist_nearest_cloud_m",
    )
    out = joined[
        [
            "h3_index",
            "dist_nearest_cloud_m",
            "provider",
            "region",
            "name",
        ]
    ].drop_duplicates(subset="h3_index", keep="first")
    out["dist_nearest_cloud_km"] = out["dist_nearest_cloud_m"] / 1000.0
    out = out.drop(columns=["dist_nearest_cloud_m"])
    out = out.rename(
        columns={
            "provider": "nearest_cloud_provider",
            "region": "nearest_cloud_region",
            "name": "nearest_cloud_region_name",
        }
    )
    return out


def build_features(
    grid_pts: gpd.GeoDataFrame,
    cities_50k: gpd.GeoDataFrame,
    cities_100k: gpd.GeoDataFrame,
    cloud: gpd.GeoDataFrame,
    cloud_count_radius_km: float,
) -> pd.DataFrame:
    log("Computing nearest city (pop > 50k)...")
    near_50k = nearest_with_name(
        grid_pts,
        cities_50k,
        "dist_nearest_city_50k_m",
    )
    near_50k = near_50k.rename(
        columns={
            "dist_nearest_city_50k_m": "dist_nearest_city_50k_km",
            "NAME": "nearest_city_50k_name",
        }
    )

    log("Computing nearest city (pop > 100k)...")
    near_100k = nearest_with_name(
        grid_pts,
        cities_100k,
        "dist_nearest_city_100k_m",
    )
    near_100k = near_100k.rename(
        columns={
            "dist_nearest_city_100k_m": "dist_nearest_city_100k_km",
            "NAME": "nearest_city_100k_name",
        }
    )

    log("Computing nearest cloud region...")
    near_cloud = nearest_cloud(grid_pts, cloud)

    log(f"Counting cloud regions within {cloud_count_radius_km:g} km...")
    cloud_counts = count_sites_within_radius(
        grid_pts,
        cloud,
        cloud_count_radius_km,
        id_col="region",
    )

    out = grid_pts[["h3_index"]].merge(near_50k, on="h3_index", how="left")
    out = out.merge(
        near_100k[["h3_index", "dist_nearest_city_100k_km", "nearest_city_100k_name"]],
        on="h3_index",
        how="left",
    )
    out = out.merge(near_cloud, on="h3_index", how="left")
    out["cloud_regions_within_500km"] = cloud_counts.values

    for col in ("dist_nearest_city_50k_km", "dist_nearest_city_100k_km", "dist_nearest_cloud_km"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["cloud_regions_within_500km"] = (
        out["cloud_regions_within_500km"].fillna(0).astype(int)
    )
    return out[OUTPUT_COLUMNS]


def print_summary(
    df: pd.DataFrame,
    total_places: int,
    n_50k: int,
    n_100k: int,
    cloud: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    log("\n" + "=" * 60)
    log("Proximity features complete")
    log("=" * 60)
    log(f"Census places loaded:           {total_places:,}")
    log(f"Places matched pop > 50k:       {n_50k:,}")
    log(f"Places matched pop > 100k:        {n_100k:,}")
    log("Cloud regions by provider:")
    for provider, count in cloud["provider"].value_counts().sort_index().items():
        log(f"  {provider}: {count}")

    for col in ("dist_nearest_city_50k_km", "dist_nearest_cloud_km"):
        valid = df[col].notna()
        log(f"\n{col}:")
        if valid.any():
            log(f"  min:    {df.loc[valid, col].min():,.2f}")
            log(f"  median: {df.loc[valid, col].median():,.2f}")
            log(f"  max:    {df.loc[valid, col].max():,.2f}")
        else:
            log("  (no values)")

    log("\ncloud_regions_within_500km:")
    log(f"  median: {df['cloud_regions_within_500km'].median():.0f}")
    log(f"  cells with 0: {(df['cloud_regions_within_500km'] == 0).sum():,}")

    if output_path.exists():
        log(f"\nOutput: {output_path} ({_format_bytes(output_path.stat().st_size)})")


def compute_proximity_features(
    grid_path: Path | None = None,
    output_path: Path | None = None,
    places_cache: Path | None = None,
    cloud_cache: Path | None = None,
    refresh_places: bool = False,
    refresh_cloud: bool = False,
    cloud_count_radius_km: float = CLOUD_COUNT_RADIUS_KM,
) -> pd.DataFrame:
    resolution = DEFAULT_RESOLUTION
    if grid_path is None:
        grid_path = PROCESSED_DIR / f"h3_grid_res{resolution}.parquet"
    if output_path is None:
        output_path = PROCESSED_DIR / f"features_proximity_res{resolution}.parquet"

    if not grid_path.exists():
        raise FileNotFoundError(
            f"H3 grid not found: {grid_path}. "
            f"Run: python scripts/build_grid.py --resolution {resolution}"
        )

    places = download_census_places(places_cache=places_cache, force=refresh_places)
    cities_50k, cities_100k, n_50k, n_100k = load_city_centroids(places)
    log(
        f"Loaded {len(places):,} Census places, "
        f"{n_50k:,} with pop > 50k, {n_100k:,} with pop > 100k"
    )

    cloud = load_cloud_regions_gdf(cache_path=cloud_cache, force=refresh_cloud)
    grid_pts = load_grid(grid_path)
    result = build_features(
        grid_pts,
        cities_50k,
        cities_100k,
        cloud,
        cloud_count_radius_km,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result.to_parquet(output_path, index=False)
    except OSError as exc:
        raise OSError(f"Failed to write {output_path}: {exc}") from exc

    print_summary(result, len(places), n_50k, n_100k, cloud, output_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute metro and cloud region proximity features per H3 cell"
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
        default=PROCESSED_DIR / f"features_proximity_res{DEFAULT_RESOLUTION}.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--places-cache",
        type=Path,
        default=PLACES_CACHE,
        help="Cached Census places parquet",
    )
    parser.add_argument(
        "--cloud-cache",
        type=Path,
        default=CLOUD_REGIONS_CACHE,
        help="Cached cloud regions parquet",
    )
    parser.add_argument(
        "--refresh-places",
        action="store_true",
        help="Re-download Census places from pygris",
    )
    parser.add_argument(
        "--refresh-cloud",
        action="store_true",
        help="Rebuild cloud regions cache from constants",
    )
    parser.add_argument(
        "--cloud-count-radius-km",
        type=float,
        default=CLOUD_COUNT_RADIUS_KM,
        help="Radius for cloud_regions_within_500km (default: 500)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        compute_proximity_features(
            grid_path=args.grid,
            output_path=args.output,
            places_cache=args.places_cache,
            cloud_cache=args.cloud_cache,
            refresh_places=args.refresh_places,
            refresh_cloud=args.refresh_cloud,
            cloud_count_radius_km=args.cloud_count_radius_km,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        log(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
