"""Build an H3 hexagonal grid covering the United States boundary."""

from __future__ import annotations

from pathlib import Path

import geodatasets
import geopandas as gpd
import h3
from shapely.geometry import Polygon, mapping

NATURALEARTH_LOWRES_COUNTRIES_URL = (
    "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
)
USA_NAME = "United States of America"
DEFAULT_RESOLUTION = 5


def _load_naturalearth_lowres() -> gpd.GeoDataFrame:
    """Load Natural Earth 110m country boundaries (formerly geopandas.datasets)."""
    try:
        path = geodatasets.get_path("naturalearth_lowres")
        return gpd.read_file(path)
    except (ValueError, KeyError, AttributeError):
        return gpd.read_file(NATURALEARTH_LOWRES_COUNTRIES_URL)


def _polyfill_geojson(geojson: dict, resolution: int) -> set[str]:
    if hasattr(h3, "polyfill_geojson"):
        return set(h3.polyfill_geojson(geojson, resolution))
    h3shape = h3.geo_to_h3shape(geojson)
    return set(h3.polygon_to_cells(h3shape, resolution))


def _h3_to_geo(h3_index: str) -> tuple[float, float]:
    if hasattr(h3, "h3_to_geo"):
        return h3.h3_to_geo(h3_index)
    return h3.cell_to_latlng(h3_index)


def _h3_to_geo_boundary(h3_index: str) -> list[tuple[float, float]]:
    if hasattr(h3, "h3_to_geo_boundary"):
        return h3.h3_to_geo_boundary(h3_index, geo_json=True)
    return [(lon, lat) for lat, lon in h3.cell_to_boundary(h3_index)]


def _filter_usa(world: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    for column in ("name", "NAME", "ADMIN", "SOVEREIGNT"):
        if column in world.columns:
            usa = world.loc[world[column] == USA_NAME]
            if not usa.empty:
                return usa.copy()
    raise ValueError(
        f"Could not find '{USA_NAME}' in naturalearth_lowres country boundaries."
    )


def load_usa_boundary() -> Polygon:
    """Download USA boundary from Natural Earth, dissolve to a single geometry."""
    world = _load_naturalearth_lowres()
    usa = _filter_usa(world)

    usa = usa.to_crs(epsg=4326)
    dissolved = usa.dissolve()
    return dissolved.geometry.iloc[0]


def polyfill_boundary(boundary, resolution: int) -> set[str]:
    """Fill a boundary geometry with H3 cells at the given resolution."""
    geojson = mapping(boundary)
    cells = _polyfill_geojson(geojson, resolution)
    if not cells:
        raise RuntimeError(
            "H3 polyfill returned 0 cells. The input geometry may not be in WGS84 "
            "(EPSG:4326). Ensure the boundary is reprojected before polyfill."
        )
    return cells


def cells_to_geodataframe(cells: set[str]) -> gpd.GeoDataFrame:
    """Convert H3 cell indices to a GeoDataFrame with centroids and hex boundaries."""
    records: list[dict] = []
    for h3_index in sorted(cells):
        lat, lon = _h3_to_geo(h3_index)
        boundary = _h3_to_geo_boundary(h3_index)
        records.append(
            {
                "h3_index": h3_index,
                "lat": lat,
                "lon": lon,
                "geometry": Polygon(boundary),
            }
        )

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    gdf = gdf.set_index("h3_index")
    return gdf


def _format_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.1f} GB"


def save_grid(gdf: gpd.GeoDataFrame, resolution: int, output_dir: Path) -> tuple[Path, Path]:
    """Save grid to parquet (with geometry) and CSV (without geometry)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"h3_grid_res{resolution}.parquet"
    csv_path = output_dir / f"h3_grid_res{resolution}.csv"

    try:
        gdf.to_parquet(parquet_path, geometry_encoding="WKB")
    except OSError as exc:
        raise OSError(f"Failed to write parquet grid to {parquet_path}: {exc}") from exc

    try:
        gdf[["lat", "lon"]].to_csv(csv_path)
    except OSError as exc:
        raise OSError(f"Failed to write CSV grid to {csv_path}: {exc}") from exc

    return parquet_path, csv_path


def print_summary(gdf: gpd.GeoDataFrame, parquet_path: Path, csv_path: Path) -> None:
    """Print cell count, bounding box, and output file sizes."""
    minx, miny, maxx, maxy = gdf.total_bounds
    print(f"Total cells: {len(gdf):,}")
    print(
        "Bounding box (lon/lat): "
        f"({minx:.4f}, {miny:.4f}) to ({maxx:.4f}, {maxy:.4f})"
    )
    print(f"Parquet: {parquet_path} ({_format_bytes(parquet_path.stat().st_size)})")
    print(f"CSV: {csv_path} ({_format_bytes(csv_path.stat().st_size)})")


def build_h3_grid(
    resolution: int = DEFAULT_RESOLUTION,
    output_dir: Path | str = "data/processed",
) -> gpd.GeoDataFrame:
    """Build, save, and summarize an H3 grid covering the USA boundary."""
    output_dir = Path(output_dir)

    print(f"Loading USA boundary from naturalearth_lowres...")
    boundary = load_usa_boundary()

    print(f"Polyfilling boundary at H3 resolution {resolution}...")
    cells = polyfill_boundary(boundary, resolution)

    print(f"Building GeoDataFrame for {len(cells):,} cells...")
    gdf = cells_to_geodataframe(cells)

    print(f"Saving outputs to {output_dir}...")
    parquet_path, csv_path = save_grid(gdf, resolution, output_dir)

    print("\nGrid build complete.")
    print_summary(gdf, parquet_path, csv_path)
    return gdf
