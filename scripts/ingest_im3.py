#!/usr/bin/env python3
"""Ingest IM3 Open Source Data Center Atlas CSV and write H3-labeled parquet."""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import h3
import pandas as pd
from shapely.geometry import Point

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT / "data" / "raw" / "im3_open_source_data_center_atlas_v2026.02.09.csv"
)
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "labels_im3.parquet"

H3_RESOLUTION = 6
BUFFER_METERS = 100  # 100 m buffers → facilities within 200 m share a cluster

US_STATE_ABBRS = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL",
    "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
    "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
    "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
}

TYPE_PRIORITY = {"campus": 3, "building": 2, "point": 1}

OUTPUT_COLUMNS = [
    "h3_index",
    "lat",
    "lon",
    "name",
    "operator",
    "state",
    "state_abb",
    "county",
    "sqft",
    "type",
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


def _geo_to_h3(lat: float, lon: float, resolution: int) -> str:
    if hasattr(h3, "geo_to_h3"):
        return h3.geo_to_h3(lat, lon, resolution)
    return h3.latlng_to_cell(lat, lon, resolution)


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"id": str})
    df["id"] = df["id"].astype(str).str.strip()
    for column in df.columns:
        if df[column].dtype == object or str(df[column].dtype) == "string":
            df[column] = df[column].astype("string").str.strip()
    return df


def drop_null_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["lat", "lon"]).copy()


def warn_non_us_states(df: pd.DataFrame) -> None:
    unknown = sorted(
        {
            str(abb).strip().upper()
            for abb in df["state_abb"].dropna().unique()
            if str(abb).strip().upper() not in US_STATE_ABBRS
        }
    )
    if unknown:
        log(f"Warning: non-US state abbreviations found: {', '.join(unknown)}")


def _norm_field(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text.casefold() if text else None


def _should_merge(
    name_i: object,
    operator_i: object,
    name_j: object,
    operator_j: object,
) -> bool:
    """Merge only when rows likely represent the same facility (same name or operator)."""
    name_a, name_b = _norm_field(name_i), _norm_field(name_j)
    operator_a, operator_b = _norm_field(operator_i), _norm_field(operator_j)

    if name_a and name_b and name_a == name_b:
        return True
    if operator_a and operator_b and operator_a == operator_b:
        return True
    return False


def _cluster_overlapping_buffers(
    buffers: gpd.GeoSeries,
    names: pd.Series,
    operators: pd.Series,
) -> list[int]:
    """Cluster rows whose 100 m buffers overlap and share a name or operator."""
    geoms = buffers.values
    n = len(geoms)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    tree = buffers.sindex
    name_vals = names.reset_index(drop=True)
    operator_vals = operators.reset_index(drop=True)

    for i, geom in enumerate(geoms):
        for j in tree.query(geom):
            if i >= j or not geom.intersects(geoms[j]):
                continue
            if _should_merge(
                name_vals.iloc[i],
                operator_vals.iloc[i],
                name_vals.iloc[j],
                operator_vals.iloc[j],
            ):
                union(i, j)

    roots = [find(i) for i in range(n)]
    root_to_cluster = {root: idx for idx, root in enumerate(dict.fromkeys(roots))}
    return [root_to_cluster[root] for root in roots]


def deduplicate_by_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """Dedup IM3 layer duplicates: within 200 m and same name or operator."""
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=[Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])],
        crs="EPSG:4326",
    )
    gdf_m = gdf.to_crs(epsg=3857).reset_index(drop=True)
    buffers = gdf_m.geometry.buffer(BUFFER_METERS)
    gdf_m["cluster_id"] = _cluster_overlapping_buffers(
        buffers, gdf_m["name"], gdf_m["operator"]
    )
    gdf_m["type_priority"] = gdf_m["type"].map(TYPE_PRIORITY).fillna(0)

    # campus > building > point; break ties by largest sqft, then stable id
    gdf_m = gdf_m.sort_values(
        ["cluster_id", "type_priority", "sqft", "id"],
        ascending=[True, False, False, True],
    )
    deduped = gdf_m.drop_duplicates(subset="cluster_id", keep="first")
    return pd.DataFrame(deduped.drop(columns=["geometry", "cluster_id", "type_priority"]))


def assign_h3_indices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["h3_index"] = [
        _geo_to_h3(lat, lon, H3_RESOLUTION) for lat, lon in zip(out["lat"], out["lon"])
    ]
    return out


def build_output(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["source"] = "IM3_DOE_2026"
    out["sqft"] = out["sqft"].where(out["sqft"].notna(), None)
    out["operator"] = out["operator"].where(out["operator"].notna(), None)
    return out[OUTPUT_COLUMNS]


def print_summary(
    loaded: int,
    after_coords: int,
    deduped: pd.DataFrame,
    removed: int,
    output_path: Path,
) -> None:
    log("\n" + "=" * 60)
    log("IM3 ingestion complete")
    log("=" * 60)
    log(f"Rows loaded from CSV:              {loaded:,}")
    log(f"Rows after dropping null coords:   {after_coords:,}")
    log(f"Rows after deduplication:          {len(deduped):,} ({removed:,} removed)")
    log("\nCount by type:")
    for facility_type, count in deduped["type"].value_counts().sort_index().items():
        log(f"  {facility_type}: {count:,}")
    log("\nTop 5 states by data center count:")
    for state, count in deduped["state"].value_counts().head(5).items():
        log(f"  {state}: {count:,}")
    if output_path.exists():
        log(f"\nOutput: {output_path} ({_format_bytes(output_path.stat().st_size)})")


def ingest(input_path: Path = DEFAULT_INPUT, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    log(f"Loading {input_path}...")
    loaded = load_csv(input_path)
    loaded_count = len(loaded)

    log("Dropping rows with null coordinates...")
    with_coords = drop_null_coordinates(loaded)
    coords_count = len(with_coords)

    warn_non_us_states(with_coords)

    log("Deduplicating layer duplicates (same name or operator within 200 m)...")
    deduped = deduplicate_by_proximity(with_coords)
    removed = coords_count - len(deduped)

    log("Assigning H3 indices (resolution 6)...")
    labeled = assign_h3_indices(deduped)
    output = build_output(labeled)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.to_parquet(output_path, index=False)
    except OSError as exc:
        raise OSError(f"Failed to write {output_path}: {exc}") from exc

    print_summary(loaded_count, coords_count, output, removed, output_path)
    return output


def main() -> int:
    try:
        ingest()
    except (FileNotFoundError, OSError) as exc:
        log(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
