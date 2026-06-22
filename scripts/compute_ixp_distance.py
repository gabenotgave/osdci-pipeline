#!/usr/bin/env python3
"""Compute distance to nearest IXP and IXP density from PeeringDB."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PEERINGDB_DIR = RAW_DIR / "peeringdb"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_RESOLUTION = 6
DEFAULT_COUNT_RADIUS_KM = 500.0
PEERINGDB_BASE = "https://www.peeringdb.com/api"
CACHE_IXFAC = PEERINGDB_DIR / "ixfac_us.json"
CACHE_FAC = PEERINGDB_DIR / "fac_us.json"
CACHE_SITES = PEERINGDB_DIR / "ixp_sites_us.parquet"

OUTPUT_COLUMNS = [
    "h3_index",
    "dist_nearest_ixp_m",
    "dist_nearest_ixp_km",
    "ixp_count_500km",
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


def fetch_peeringdb_json(url: str, retries: int = 3) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=120) as response:
                return json.loads(response.read())
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise OSError(f"PeeringDB request failed for {url}: {last_error}") from last_error


def download_peeringdb_cache(country: str, force: bool) -> None:
    PEERINGDB_DIR.mkdir(parents=True, exist_ok=True)
    endpoints = {
        CACHE_IXFAC: f"{PEERINGDB_BASE}/ixfac?country={country}",
        CACHE_FAC: f"{PEERINGDB_BASE}/fac?country={country}",
    }
    for path, url in endpoints.items():
        if path.exists() and not force:
            log(f"  Using cached {path.name}")
            continue
        log(f"  Downloading {url}...")
        payload = fetch_peeringdb_json(url)
        records = payload.get("data", [])
        path.write_text(json.dumps(records), encoding="utf-8")
        log(f"    Saved {len(records):,} records to {path.name}")


def load_ixp_sites(
    country: str = "US",
    cache_sites: Path | None = None,
    force_download: bool = False,
) -> gpd.GeoDataFrame:
    cache_sites = cache_sites or CACHE_SITES
    if cache_sites.exists() and not force_download:
        log(f"Loading IXP sites from {cache_sites.name}...")
        sites = gpd.read_parquet(cache_sites)
        if sites.crs is None:
            sites = sites.set_crs("EPSG:4326")
        log(f"  {len(sites):,} peering locations ({sites['ix_id'].nunique():,} unique IXPs)")
        return sites

    log(f"Building IXP site list for country={country}...")
    download_peeringdb_cache(country, force=force_download)

    ixfac = json.loads(CACHE_IXFAC.read_text(encoding="utf-8"))
    facilities = json.loads(CACHE_FAC.read_text(encoding="utf-8"))
    fac_lookup = {
        row["id"]: row
        for row in facilities
        if row.get("latitude") is not None and row.get("longitude") is not None
    }

    rows: list[dict] = []
    for link in ixfac:
        fac = fac_lookup.get(link["fac_id"])
        if fac is None:
            continue
        lat = float(fac["latitude"])
        lon = float(fac["longitude"])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        if lat == 0 and lon == 0:
            continue
        rows.append(
            {
                "ix_id": int(link["ix_id"]),
                "fac_id": int(link["fac_id"]),
                "ix_name": link.get("name") or fac.get("name"),
                "city": link.get("city") or fac.get("city"),
                "country": link.get("country") or fac.get("country"),
                "lat": lat,
                "lon": lon,
            }
        )

    if not rows:
        raise ValueError(f"No IXP facility locations found for country={country}")

    sites = pd.DataFrame(rows).drop_duplicates(subset=["ix_id", "fac_id"])
    geo = gpd.GeoDataFrame(
        sites,
        geometry=gpd.points_from_xy(sites["lon"], sites["lat"]),
        crs="EPSG:4326",
    )
    PEERINGDB_DIR.mkdir(parents=True, exist_ok=True)
    geo.to_parquet(cache_sites, index=False)
    log(
        f"  {len(geo):,} peering locations cached "
        f"({geo['ix_id'].nunique():,} unique IXPs)"
    )
    return geo


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


def compute_nearest_ixp(
    grid_pts: gpd.GeoDataFrame, ixp_sites: gpd.GeoDataFrame
) -> pd.DataFrame:
    log("Computing distance to nearest IXP...")
    sites_m = ixp_sites.to_crs(epsg=3857)
    grid_m = grid_pts.to_crs(epsg=3857)

    nearest = gpd.sjoin_nearest(
        grid_m[["h3_index", "geometry"]],
        sites_m[["ix_id", "geometry"]],
        how="left",
        distance_col="dist_nearest_ixp_m",
    )
    nearest = nearest[["h3_index", "dist_nearest_ixp_m"]].drop_duplicates(
        subset="h3_index", keep="first"
    )
    nearest["dist_nearest_ixp_km"] = nearest["dist_nearest_ixp_m"] / 1000.0
    return nearest


def count_ixps_within_radius(
    grid_pts: gpd.GeoDataFrame,
    ixp_sites: gpd.GeoDataFrame,
    radius_km: float,
) -> pd.Series:
    log(f"Counting unique IXPs within {radius_km:g} km...")
    radius_m = radius_km * 1000.0
    sites_m = ixp_sites.to_crs(epsg=3857)
    grid_m = grid_pts.to_crs(epsg=3857)

    buffered = sites_m[["ix_id", "geometry"]].copy()
    buffered["geometry"] = buffered.geometry.buffer(radius_m)

    joined = gpd.sjoin(
        grid_m[["h3_index", "geometry"]],
        buffered,
        how="left",
        predicate="within",
    )
    if joined.empty:
        return pd.Series(0, index=grid_pts["h3_index"], dtype=int)

    counts = joined.groupby("h3_index")["ix_id"].nunique()
    return counts.reindex(grid_pts["h3_index"], fill_value=0).astype(int)


def build_features(
    grid_pts: gpd.GeoDataFrame,
    ixp_sites: gpd.GeoDataFrame,
    count_radius_km: float,
) -> pd.DataFrame:
    nearest = compute_nearest_ixp(grid_pts, ixp_sites)
    ixp_counts = count_ixps_within_radius(grid_pts, ixp_sites, count_radius_km)

    out = grid_pts[["h3_index"]].merge(nearest, on="h3_index", how="left")
    out["ixp_count_500km"] = ixp_counts.values
    out["source"] = "PeeringDB"
    out["ixp_count_500km"] = out["ixp_count_500km"].fillna(0).astype(int)
    return out[OUTPUT_COLUMNS]


def print_summary(df: pd.DataFrame, count_radius_km: float, output_path: Path) -> None:
    log("\n" + "=" * 60)
    log("IXP distance features complete")
    log("=" * 60)
    log(f"H3 cells:                         {len(df):,}")
    log("\nDistance to nearest IXP (km):")
    log(f"  min:    {df['dist_nearest_ixp_km'].min():,.2f}")
    log(f"  median: {df['dist_nearest_ixp_km'].median():,.2f}")
    log(f"  mean:   {df['dist_nearest_ixp_km'].mean():,.2f}")
    log(f"  max:    {df['dist_nearest_ixp_km'].max():,.2f}")
    log(f"  ≤ 25 km: {(df['dist_nearest_ixp_km'] <= 25).sum():,} cells")
    log(f"  ≤ 50 km: {(df['dist_nearest_ixp_km'] <= 50).sum():,} cells")
    log(f"\nIXP count within {count_radius_km:g} km:")
    log(f"  min:    {df['ixp_count_500km'].min()}")
    log(f"  median: {df['ixp_count_500km'].median():.0f}")
    log(f"  max:    {df['ixp_count_500km'].max()}")
    log(f"  > 0:    {(df['ixp_count_500km'] > 0).sum():,} cells")
    if output_path.exists():
        log(f"\nOutput: {output_path} ({_format_bytes(output_path.stat().st_size)})")


def compute_ixp_distance(
    grid_path: Path | None = None,
    output_path: Path | None = None,
    cache_sites: Path | None = None,
    country: str = "US",
    count_radius_km: float = DEFAULT_COUNT_RADIUS_KM,
    refresh: bool = False,
) -> pd.DataFrame:
    resolution = DEFAULT_RESOLUTION
    if grid_path is None:
        grid_path = PROCESSED_DIR / f"h3_grid_res{resolution}.parquet"
    if output_path is None:
        output_path = PROCESSED_DIR / f"features_ixp_distance_res{resolution}.parquet"
    if cache_sites is None:
        cache_sites = CACHE_SITES

    if not grid_path.exists():
        raise FileNotFoundError(
            f"H3 grid not found: {grid_path}. "
            f"Run: python scripts/build_grid.py --resolution {resolution}"
        )

    ixp_sites = load_ixp_sites(country=country, cache_sites=cache_sites, force_download=refresh)
    grid_pts = load_grid(grid_path)
    result = build_features(grid_pts, ixp_sites, count_radius_km)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result.to_parquet(output_path, index=False)
    except OSError as exc:
        raise OSError(f"Failed to write {output_path}: {exc}") from exc

    print_summary(result, count_radius_km, output_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute H3 distance to nearest IXP and IXP count within 500 km (PeeringDB)"
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
        default=PROCESSED_DIR / f"features_ixp_distance_res{DEFAULT_RESOLUTION}.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--cache-sites",
        type=Path,
        default=CACHE_SITES,
        help="Cached IXP site locations parquet",
    )
    parser.add_argument(
        "--country",
        default="US",
        help="PeeringDB country filter for IX-facility links (default: US)",
    )
    parser.add_argument(
        "--count-radius-km",
        type=float,
        default=DEFAULT_COUNT_RADIUS_KM,
        help="Radius for ixp_count feature in km (default: 500)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download PeeringDB API data and rebuild site cache",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        compute_ixp_distance(
            grid_path=args.grid,
            output_path=args.output,
            cache_sites=args.cache_sites,
            country=args.country,
            count_radius_km=args.count_radius_km,
            refresh=args.refresh,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        log(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
