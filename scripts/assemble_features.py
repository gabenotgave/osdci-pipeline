#!/usr/bin/env python3
"""Assemble all feature parquets into a single CONUS training-ready master table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from osdci.features import (  # noqa: E402
    ALL_FEATURES,
    CLIMATE_FEATURES,
    EJ_FEATURES,
    EXCLUSION_FLAGS,
    GRID_FEATURES,
    ID_COLS,
    LABEL_COL,
    LABEL_META,
    LAND_FEATURES,
    NETWORK_FEATURES,
    POPULATION_FEATURES,
    POWER_FEATURES,
    PROXIMITY_FEATURES,
    RISK_FEATURES,
    WATER_FEATURES,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_GRID = PROCESSED_DIR / "h3_grid_res6.parquet"
DEFAULT_LABELS = PROCESSED_DIR / "labels_aggregated_res6.parquet"
DEFAULT_OUTPUT = PROCESSED_DIR / "features_master_res6.parquet"
DEFAULT_INDEX = PROCESSED_DIR / "features_master_res6_index.csv"

CONUS_MIN_LAT = 24.0
CONUS_MAX_LAT = 50.0
CONUS_MIN_LON = -125.0
CONUS_MAX_LON = -66.0

FEATURE_FILES = [
    "features_industrial_price_res6.parquet",
    "features_egrid_grid_res6.parquet",
    "features_eia860_grid_res6.parquet",
    "features_tx_distance_res6.parquet",
    "features_substation_distance_res6.parquet",
    "features_nearby_generation_res6.parquet",
    "features_ixp_distance_res6.parquet",
    "features_proximity_res6.parquet",
    "features_nlcd_res6.parquet",
    "features_impervious_res6.parquet",
    "features_slope_res6.parquet",
    "features_padus_res6.parquet",
    "features_wetland_res6.parquet",
    "features_nri_res6.parquet",
    "features_climate_res6.parquet",
    "features_water_stress_res6.parquet",
    "features_population_res6.parquet",
    "features_ejscreen_res6.parquet",
    "features_tribal_res6.parquet",
    "features_critical_habitat_res6.parquet",
]

# Map source parquet column names → canonical names in osdci/features.py
COLUMN_ALIASES = {
    "dist_tx_line_km": "dist_transmission_km",
    "industrial_price_mwh": "elec_price_mwh",
    "co2_lb_mwh_subregion": "co2_rate_lb_mwh",
    "renewable_pct_subregion": "renewable_pct",
    "renewable_pct_ba": "ba_renewable_pct",
}

JOIN_COLUMNS = list(dict.fromkeys(list(ALL_FEATURES) + list(EXCLUSION_FLAGS)))

# Metadata columns joined but not counted as model features
METADATA_COLUMNS = [
    "tribal_name",
    "critical_habitat_species_count",
]

FEATURE_GROUPS = {
    "Power": POWER_FEATURES,
    "Grid": GRID_FEATURES,
    "Network": NETWORK_FEATURES,
    "Proximity": PROXIMITY_FEATURES,
    "Land": LAND_FEATURES,
    "Climate": CLIMATE_FEATURES,
    "Water": WATER_FEATURES,
    "Risk": RISK_FEATURES,
    "Population": POPULATION_FEATURES,
    "EJ": EJ_FEATURES,
}


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


def load_conus_grid(grid_path: Path) -> pd.DataFrame:
    log(f"Loading H3 grid from {grid_path.name}...")
    grid = gpd.read_parquet(grid_path)
    if grid.index.name == "h3_index":
        grid = grid.reset_index()

    grid["is_conus"] = (
        (grid["lat"] >= CONUS_MIN_LAT)
        & (grid["lat"] <= CONUS_MAX_LAT)
        & (grid["lon"] >= CONUS_MIN_LON)
        & (grid["lon"] <= CONUS_MAX_LON)
    ).astype(int)

    conus = grid.loc[grid["is_conus"] == 1, ["h3_index", "lat", "lon", "is_conus"]].copy()
    log(f"CONUS cells: {len(conus):,} of {len(grid):,} total")
    return conus


def prepare_feature_frame(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    df = df.copy()
    if "h3_index" not in df.columns and df.index.name == "h3_index":
        df = df.reset_index()

    rename = {c: COLUMN_ALIASES[c] for c in df.columns if c in COLUMN_ALIASES}
    if rename:
        df = df.rename(columns=rename)

    keep = ["h3_index"] + [
        c for c in df.columns if c in JOIN_COLUMNS or c in METADATA_COLUMNS
    ]
    return df[keep]


def join_feature_file(base: pd.DataFrame, path: Path) -> pd.DataFrame:
    filename = path.name
    df = pd.read_parquet(path)
    feat = prepare_feature_frame(df, filename)
    feat_cols = [c for c in feat.columns if c != "h3_index"]

    if not feat_cols:
        log(f"  Joined {filename}: 0 columns (skipped)")
        return base

    overlap = [c for c in feat_cols if c in base.columns]
    new_cols = [c for c in feat_cols if c not in base.columns]
    if overlap:
        log(f"  Warning: {filename} skips duplicate columns: {', '.join(overlap)}")

    if not new_cols:
        log(f"  Joined {filename}: 0 new columns")
        return base

    join_df = feat[["h3_index"] + new_cols]
    merged = base.merge(join_df, on="h3_index", how="left")
    matches = join_df["h3_index"].isin(base["h3_index"]).sum()
    log(
        f"  Joined {filename}: {len(new_cols)} columns, "
        f"{matches:,} source rows"
    )
    return merged


def attach_labels(base: pd.DataFrame, labels_path: Path) -> pd.DataFrame:
    log(f"Loading labels from {labels_path.name}...")
    labels = pd.read_parquet(labels_path)
    label_cols = ["h3_index", LABEL_COL] + [c for c in LABEL_META if c in labels.columns]
    labels = labels[label_cols]

    merged = base.merge(labels, on="h3_index", how="left")
    merged[LABEL_COL] = merged[LABEL_COL].fillna(0).astype(int)
    if "facility_count" in merged.columns:
        merged["facility_count"] = merged["facility_count"].fillna(0).astype(int)
    if "has_hyperscaler" in merged.columns:
        merged["has_hyperscaler"] = merged["has_hyperscaler"].fillna(0).astype(int)

    positives = int((merged[LABEL_COL] == 1).sum())
    negatives = int((merged[LABEL_COL] == 0).sum())
    pos_rate = 100.0 * positives / len(merged) if len(merged) else 0.0
    log(f"  Positive cells (has_dc=1): {positives:,}")
    log(f"  Negative cells (has_dc=0): {negatives:,}")
    log(f"  Positive rate: {pos_rate:.4f}%")
    return merged


def null_summary(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(dtype=float)
    present = [c for c in columns if c in df.columns]
    if not present:
        return pd.Series(dtype=float)
    return df[present].isna().mean().sort_values(ascending=False)


def group_coverage(df: pd.DataFrame, features: list[str]) -> float:
    present = [c for c in features if c in df.columns]
    if not present:
        return float("nan")
    return 100.0 * (1.0 - df[present].isna().mean().mean())


def print_join_diagnostics(df: pd.DataFrame) -> None:
    feature_cols = [c for c in JOIN_COLUMNS if c in df.columns]
    log(f"\nTotal columns in assembled dataset: {len(df.columns)}")
    log(f"Feature columns present: {len(feature_cols)} of {len(JOIN_COLUMNS)} canonical")

    missing_canonical = [c for c in JOIN_COLUMNS if c not in df.columns]
    if missing_canonical:
        log("\nCanonical columns not found in any feature file:")
        for col in missing_canonical:
            log(f"  - {col}")

    if feature_cols:
        null_pct = (df[feature_cols].isna().mean() * 100).sort_values(ascending=False)
        log("\nTop 20 most null feature columns (%):")
        for col, pct in null_pct.head(20).items():
            log(f"  {col}: {pct:.2f}%")

        high_null = null_pct[null_pct > 50]
        if not high_null.empty:
            log("\nWarning: columns with > 50% null:")
            for col, pct in high_null.items():
                log(f"  {col}: {pct:.2f}%")


def print_final_report(
    df: pd.DataFrame,
    output_path: Path,
    feature_col_count: int,
) -> None:
    positives = int((df[LABEL_COL] == 1).sum())
    negatives = int((df[LABEL_COL] == 0).sum())
    pos_pct = 100.0 * positives / len(df) if len(df) else 0.0
    neg_pct = 100.0 * negatives / len(df) if len(df) else 0.0
    hyperscaler = int(df.get("has_hyperscaler", pd.Series(dtype=int)).sum())

    feature_cols = [c for c in JOIN_COLUMNS if c in df.columns]
    null_pct = (df[feature_cols].isna().mean() * 100).sort_values(ascending=False)
    top_null = [f"{c} ({v:.1f}%)" for c, v in null_pct.head(10).items()]

    log("\n" + "=" * 60)
    log("Feature assembly complete")
    log("=" * 60)
    log(f"Total H3 cells (CONUS):        {len(df):,}")
    log(f"Feature columns:                {feature_col_count}")
    log("Label distribution:")
    log(f"  has_dc = 1 (positive):       {positives:,} ({pos_pct:.4f}%)")
    log(f"  has_dc = 0 (negative):       {negatives:,} ({neg_pct:.4f}%)")
    log(f"  has_hyperscaler = 1:         {hyperscaler:,}")
    log("Coverage summary (% non-null per feature group):")
    for name, feats in FEATURE_GROUPS.items():
        cov = group_coverage(df, feats)
        cov_str = f"{cov:.1f}" if not np.isnan(cov) else "n/a"
        log(f"  {name} features:              {cov_str}%")
    log(f"Top 10 most null columns:      {', '.join(top_null)}")
    for col in ("in_tribal_land", "in_critical_habitat"):
        if col in df.columns:
            coverage = 100.0 * df[col].notna().mean()
            log(f"  {col} coverage:            {coverage:.1f}% non-null")
    if output_path.exists():
        log(f"Output: {output_path} ({_format_bytes(output_path.stat().st_size)})")


def assemble_features(
    grid_path: Path | None = None,
    labels_path: Path | None = None,
    output_path: Path | None = None,
    index_path: Path | None = None,
    processed_dir: Path | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    grid_path = grid_path or DEFAULT_GRID
    labels_path = labels_path or DEFAULT_LABELS
    output_path = output_path or DEFAULT_OUTPUT
    index_path = index_path or DEFAULT_INDEX
    processed_dir = processed_dir or PROCESSED_DIR

    if not grid_path.exists():
        raise FileNotFoundError(f"H3 grid not found: {grid_path}")

    conus = load_conus_grid(grid_path)

    log("\nJoining feature files...")
    for filename in FEATURE_FILES:
        path = processed_dir / filename
        if not path.exists():
            log(f"  Warning: missing {filename} — skipped")
            continue
        conus = join_feature_file(conus, path)

    print_join_diagnostics(conus)

    if labels_path.exists():
        conus = attach_labels(conus, labels_path)
    else:
        log(f"Warning: labels not found at {labels_path}; has_dc will be missing")

    # Final column order: IDs, features, exclusions, labels
    feature_cols = [c for c in JOIN_COLUMNS if c in conus.columns]
    label_cols = [c for c in [LABEL_COL] + LABEL_META if c in conus.columns]
    ordered = ID_COLS + feature_cols + label_cols
    extra = [c for c in conus.columns if c not in ordered]
    master = conus[ordered + extra]

    feature_col_count = len(feature_cols)

    if dry_run:
        log("\nDry run — no files written.")
        print_final_report(master, output_path, feature_col_count)
        return master

    log(f"\nWriting {output_path.name}...")
    write_parquet_safe(master, output_path)

    index_cols = ["h3_index", "lat", "lon", "is_conus", LABEL_COL]
    if "facility_count" in master.columns:
        index_cols.append("facility_count")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    master[index_cols].to_csv(index_path, index=False)
    log(f"Wrote index: {index_path.name}")

    print_final_report(master, output_path, feature_col_count)
    return master


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble CONUS feature parquets into a master training table"
    )
    parser.add_argument(
        "--grid",
        type=Path,
        default=DEFAULT_GRID,
        help="H3 grid parquet",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help="Aggregated labels parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Master features parquet output path",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
        help="Lightweight index CSV output path",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory containing feature parquets",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print assembly report without writing output files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        assemble_features(
            grid_path=args.grid,
            labels_path=args.labels,
            output_path=args.output,
            index_path=args.index,
            processed_dir=args.processed_dir,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        log(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
