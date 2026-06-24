#!/usr/bin/env python3
"""Combine model viability scores with community impact penalties."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import h3
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_MASTER = PROCESSED_DIR / "features_master_res6.parquet"
DEFAULT_SCORES = MODELS_DIR / "spatial_residuals.csv"
DEFAULT_OUTPUT = PROCESSED_DIR / "scored_cells.parquet"
DEFAULT_GEOJSON = MODELS_DIR / "top_candidates.geojson"
DEFAULT_BASEMAP = MODELS_DIR / "basemap.csv"
DEFAULT_EXPLANATIONS = MODELS_DIR / "cell_explanations.json"
DEFAULT_MODEL_VERSION = "v2_regularized"
VIABILITY_THRESHOLD = 0.15
BASEMAP_VIABILITY_THRESHOLD = 0.02
BASEMAP_SIZE_LIMIT_MB = 20.0
GEOJSON_SIZE_WARN_MB = 40.0

BASEMAP_COLS = [
    "h3_index",
    "lat",
    "lon",
    "viability_score",
    "final_score",
    "final_score_pct",
    "display_category",
    "has_dc",
    "c_protected_area",
    "c_wetland",
    "c_tribal_land",
    "c_critical_habitat",
    "c_floodplain",
    "c_wildfire",
    "c_seismic",
    "c_water_stress",
    "c_ej_burden",
    "c_population",
    "dist_nearest_ixp_km",
    "dist_transmission_km",
]

get_cell_boundary = getattr(
    h3,
    "cell_to_boundary",
    getattr(h3, "h3_to_geo_boundary", None),
)

DEFAULT_WEIGHTS = {
    "in_protected_area": 1.0,
    "in_wetland": 1.0,
    "in_floodplain_100yr": 0.6,
    "in_high_wildfire": 0.5,
    "in_high_seismic": 0.4,
    "high_water_stress": 0.5,
    "demographic_index": 0.7,
    "population": 0.6,
    "in_tribal_land": 1.0,
    "in_critical_habitat": 0.9,
}

HARD_EXCLUSION_COLS = [
    "in_protected_area",
    "in_wetland",
    "in_tribal_land",
    "in_critical_habitat",
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


def normalize_population(series: pd.Series) -> pd.Series:
    values = series.fillna(0).astype(float)
    vmax = values.max()
    if vmax <= 0:
        return values * 0.0
    return values / vmax


def compute_impact_penalty(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    weights = weights or DEFAULT_WEIGHTS
    pop_norm = normalize_population(df["pop_within_25mi"])

    components = {
        "in_protected_area": df["in_protected_area"].fillna(0),
        "in_wetland": df["in_wetland"].fillna(0),
        "in_floodplain_100yr": df["in_floodplain_100yr"].fillna(0),
        "in_high_wildfire": df["in_high_wildfire"].fillna(0),
        "in_high_seismic": df["in_high_seismic"].fillna(0),
        "high_water_stress": df["high_water_stress"].fillna(0),
        "demographic_index": df["demographic_index_norm"].fillna(0),
        "population": pop_norm.fillna(0),
        "in_tribal_land": df["in_tribal_land"].fillna(0),
        "in_critical_habitat": df["in_critical_habitat"].fillna(0),
    }

    penalty = pd.Series(0.0, index=df.index, dtype=float)
    weight_sum = sum(weights.values())
    for name, weight in weights.items():
        penalty += components[name] * weight
    return (penalty / weight_sum).clip(0.0, 1.0)


def display_category(df: pd.DataFrame) -> pd.Series:
    hard = pd.Series(False, index=df.index)
    for col in HARD_EXCLUSION_COLS:
        if col in df.columns:
            hard |= df[col].fillna(0).astype(int) == 1

    categories = pd.Series("Low Viability", index=df.index, dtype="object")
    categories.loc[df["has_dc"] == 1] = "Existing Data Center"
    categories.loc[(df["has_dc"] == 0) & hard] = "Hard Exclusion"
    categories.loc[
        (df["has_dc"] == 0)
        & ~hard
        & (df["impact_penalty"] >= 0.5)
    ] = "High Community Impact"
    categories.loc[
        (df["has_dc"] == 0)
        & ~hard
        & (df["impact_penalty"] < 0.5)
        & (df["final_score"] >= 0.5)
        & (df.get("is_soft_positive", 0) == 0)
    ] = "Candidate"
    return categories


def safe_round(val, digits: int):
    return round(float(val), digits) if pd.notnull(val) else None


def safe_int(val):
    return int(val) if pd.notnull(val) else None


def filter_viable_pool(scored_df: pd.DataFrame) -> pd.DataFrame:
    return scored_df[
        (scored_df["viability_score"] >= VIABILITY_THRESHOLD)
        & (scored_df["has_dc"] == 0)
        & (scored_df["is_soft_positive"] == 0)
    ].copy()


def add_penalty_components(scored_df: pd.DataFrame) -> pd.DataFrame:
    df = scored_df.copy()
    pop_q99 = df["pop_within_25mi"].fillna(0).quantile(0.99)
    if pop_q99 <= 0:
        c_population = df["pop_within_25mi"].fillna(0).astype(float) * 0.0
    else:
        c_population = (df["pop_within_25mi"].fillna(0) / pop_q99).clip(0.0, 1.0)

    df["c_protected_area"] = df["in_protected_area"].fillna(0).astype(float)
    df["c_wetland"] = df["in_wetland"].fillna(0).astype(float)
    df["c_tribal_land"] = df["in_tribal_land"].fillna(0).astype(float)
    df["c_critical_habitat"] = df["in_critical_habitat"].fillna(0).astype(float)
    df["c_floodplain"] = df["in_floodplain_100yr"].fillna(0).astype(float)
    df["c_wildfire"] = df["in_high_wildfire"].fillna(0).astype(float)
    df["c_seismic"] = df["in_high_seismic"].fillna(0).astype(float)
    df["c_water_stress"] = df["high_water_stress"].fillna(0).astype(float)
    df["c_ej_burden"] = df["demographic_index_norm"].fillna(0).astype(float)
    df["c_population"] = c_population
    return df


def build_feature_properties(row: pd.Series) -> dict:
    return {
        "h3_index": row["h3_index"],
        "lat": safe_round(row["lat"], 5),
        "lon": safe_round(row["lon"], 5),
        "viability_score": safe_round(row["viability_score"], 4),
        "impact_penalty": safe_round(row["impact_penalty"], 4),
        "final_score": safe_round(row["final_score"], 4),
        "final_score_pct": safe_round(row["final_score"] * 100, 1),
        "c_protected_area": safe_round(row["c_protected_area"], 3),
        "c_wetland": safe_round(row["c_wetland"], 3),
        "c_tribal_land": safe_round(row["c_tribal_land"], 3),
        "c_critical_habitat": safe_round(row["c_critical_habitat"], 3),
        "c_floodplain": safe_round(row["c_floodplain"], 3),
        "c_wildfire": safe_round(row["c_wildfire"], 3),
        "c_seismic": safe_round(row["c_seismic"], 3),
        "c_water_stress": safe_round(row["c_water_stress"], 3),
        "c_ej_burden": safe_round(row["c_ej_burden"], 3),
        "c_population": safe_round(row["c_population"], 3),
        "elec_price_mwh": safe_round(row.get("elec_price_mwh"), 2),
        "renewable_pct": safe_round(row.get("renewable_pct"), 3),
        "co2_rate_lb_mwh": safe_round(row.get("co2_rate_lb_mwh"), 1),
        "mean_temp_c": safe_round(row.get("mean_temp_c"), 1),
        "annual_cdd": safe_round(row.get("annual_cdd"), 0),
        "cooling_favorable": safe_int(row.get("cooling_favorable")),
        "dist_nearest_ixp_km": safe_round(row.get("dist_nearest_ixp_km"), 1),
        "dist_nearest_cloud_km": safe_round(row.get("dist_nearest_cloud_km"), 1),
        "dist_transmission_km": safe_round(row.get("dist_transmission_km"), 1),
        "dist_substation_km": safe_round(row.get("dist_substation_km"), 1),
        "water_stress_score": safe_round(row.get("water_stress_score"), 3),
        "pop_density_km2": safe_round(row.get("pop_density_km2"), 1),
        "mean_slope_deg": safe_round(row.get("mean_slope_deg"), 1),
        "mean_impervious_pct": safe_round(row.get("mean_impervious_pct"), 1),
        "pm25_pctile": safe_round(row.get("pm25_pctile"), 1),
        "demographic_index_norm": safe_round(row.get("demographic_index_norm"), 3),
    }


def export_top_candidates_geojson(
    scored_df: pd.DataFrame,
    geojson_path: Path,
    model_version: str = DEFAULT_MODEL_VERSION,
) -> tuple[list[dict], Path]:
    if get_cell_boundary is None:
        raise RuntimeError("h3 library missing cell_to_boundary / h3_to_geo_boundary")

    scored_with_components = add_penalty_components(scored_df)
    viable = filter_viable_pool(scored_with_components)
    features: list[dict] = []

    for _, row in viable.iterrows():
        boundary = get_cell_boundary(row["h3_index"])
        coords = [[lon, lat] for lat, lon in boundary]
        coords.append(coords[0])

        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": build_feature_properties(row),
            }
        )

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "generated": datetime.now().isoformat(),
            "model_version": model_version,
            "viability_threshold": VIABILITY_THRESHOLD,
            "total_candidates": len(features),
            "conus_cells": len(scored_df),
            "existing_dc_cells": int(scored_df["has_dc"].sum()),
            "h3_resolution": 6,
            "description": (
                "OSDCI candidate cells — "
                "viability_score >= 0.15, no existing DC, "
                "not a k-ring soft label neighbor"
            ),
            "penalty_components": [
                "c_protected_area",
                "c_wetland",
                "c_tribal_land",
                "c_critical_habitat",
                "c_floodplain",
                "c_wildfire",
                "c_seismic",
                "c_water_stress",
                "c_ej_burden",
                "c_population",
            ],
        },
    }

    geojson_path.parent.mkdir(parents=True, exist_ok=True)
    with geojson_path.open("w") as handle:
        json.dump(geojson, handle)

    size_mb = os.path.getsize(geojson_path) / 1_048_576

    log(f"\nViable candidate pool:    {len(viable):,} cells "
        f"(viability >= {VIABILITY_THRESHOLD})")
    log("At default weights:")
    top = viable.sort_values("final_score", ascending=False).iloc[0]
    log(
        f"  Top candidate:          {top['h3_index']} "
        f"lat={top['lat']:.5f} lon={top['lon']:.5f} "
        f"score={top['final_score']:.4f}"
    )
    for threshold in (0.5, 0.3, 0.1):
        count = int((viable["final_score"] >= threshold).sum())
        log(f"  score >= {threshold:.1f}:           {count:,} cells")
    log("Hard exclusion overlap (still in GeoJSON, penalized by weights):")
    log(f"  c_protected_area=1:     {int((viable['c_protected_area'] == 1).sum()):,} cells")
    log(f"  c_tribal_land=1:        {int((viable['c_tribal_land'] == 1).sum()):,} cells")
    log(
        f"  c_critical_habitat=1: "
        f"{int((viable['c_critical_habitat'] == 1).sum()):,} cells"
    )
    log(f"GeoJSON written:          {geojson_path} ({size_mb:.1f} MB)")

    if size_mb > GEOJSON_SIZE_WARN_MB:
        log(
            f"WARNING: GeoJSON is {size_mb:.1f} MB — frontend may need "
            "chunking or a backend API"
        )

    return features, geojson_path


def build_basemap_row(row: dict) -> dict:
    return {
        "h3_index": row["h3_index"],
        "lat": round(float(row["lat"]), 5),
        "lon": round(float(row["lon"]), 5),
        "viability_score": round(float(row["viability_score"]), 4)
        if pd.notnull(row.get("viability_score"))
        else 0.0,
        "final_score": round(float(row["final_score"]), 4)
        if pd.notnull(row.get("final_score"))
        else 0.0,
        "final_score_pct": round(float(row["final_score"]) * 100, 1)
        if pd.notnull(row.get("final_score"))
        else 0.0,
        "display_category": str(row.get("display_category", "Low Viability")),
        "has_dc": int(row.get("has_dc", 0)),
        "c_protected_area": int(row.get("in_protected_area", 0) or 0),
        "c_wetland": int(row.get("in_wetland", 0) or 0),
        "c_tribal_land": int(row.get("in_tribal_land", 0) or 0),
        "c_critical_habitat": int(row.get("in_critical_habitat", 0) or 0),
        "c_floodplain": int(row.get("in_floodplain_100yr", 0) or 0),
        "c_wildfire": int(row.get("in_high_wildfire", 0) or 0),
        "c_seismic": int(row.get("in_high_seismic", 0) or 0),
        "c_water_stress": int(row.get("high_water_stress", 0) or 0),
        "c_ej_burden": round(float(row["demographic_index_norm"]), 4)
        if pd.notnull(row.get("demographic_index_norm"))
        else 0.0,
        "c_population": round(float(row["pop_within_25mi_norm"]), 4)
        if pd.notnull(row.get("pop_within_25mi_norm"))
        else 0.0,
        "dist_nearest_ixp_km": round(float(row["dist_nearest_ixp_km"]), 1)
        if pd.notnull(row.get("dist_nearest_ixp_km"))
        else "",
        "dist_transmission_km": round(float(row["dist_transmission_km"]), 1)
        if pd.notnull(row.get("dist_transmission_km"))
        else "",
    }


def export_basemap_csv(
    scored_df: pd.DataFrame,
    basemap_path: Path,
) -> tuple[int, Path, float]:
    # Frontend usage:
    # import Papa from 'papaparse'
    # Papa.parse('/basemap.csv', {
    #   header: true, dynamicTyping: true,
    #   complete: ({ data }) => initBasemapLayer(data)
    # })
    # deck.gl H3HexagonLayer: getHexagon: d => d.h3_index

    for threshold in (BASEMAP_VIABILITY_THRESHOLD, 0.05):
        basemap_df = scored_df[scored_df["viability_score"] >= threshold].copy()
        log(f"\nBasemap cells (viability >= {threshold}): {len(basemap_df):,}")

        rows: list[dict] = []
        records = basemap_df.to_dict("records")
        for i, row in enumerate(records):
            if i % 50000 == 0:
                log(f"  {i:,} / {len(basemap_df):,}...")
            rows.append(build_basemap_row(row))

        basemap_path.parent.mkdir(parents=True, exist_ok=True)
        with basemap_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=BASEMAP_COLS)
            writer.writeheader()
            writer.writerows(rows)

        size_mb = os.path.getsize(basemap_path) / 1_048_576
        log(f"Basemap CSV written: {len(rows):,} cells ({size_mb:.1f} MB)")

        if size_mb <= BASEMAP_SIZE_LIMIT_MB:
            log("Size OK — suitable for static file delivery")
            return len(rows), basemap_path, threshold

        if threshold == BASEMAP_VIABILITY_THRESHOLD:
            log(
                f"WARNING: basemap.csv is {size_mb:.1f} MB at viability >= {threshold} "
                "— retrying with viability >= 0.05"
            )
            continue

        log(
            "WARNING: still above 20 MB — consider gzip compression "
            "or raising viability threshold further"
        )
        return len(rows), basemap_path, threshold

    return 0, basemap_path, BASEMAP_VIABILITY_THRESHOLD


def explain_cell(row: dict) -> list[str]:
    reasons: list[str] = []

    if row.get("in_protected_area"):
        reasons.append("Inside protected federal land (PAD-US GAP 1/2)")
    if row.get("in_tribal_land"):
        reasons.append("Sovereign tribal territory (AIANNH)")
    if row.get("in_wetland"):
        reasons.append("Wetland area — permitting barrier")
    if row.get("in_critical_habitat"):
        reasons.append("ESA critical habitat designation")

    if row.get("in_floodplain_100yr"):
        reasons.append("100-year flood zone (FEMA NRI)")
    if row.get("in_high_wildfire"):
        reasons.append("High wildfire hazard")
    if row.get("in_high_seismic"):
        reasons.append("High seismic risk zone")
    if row.get("high_water_stress"):
        reasons.append("High baseline water stress")

    ixp_km = row.get("dist_nearest_ixp_km")
    if pd.notnull(ixp_km) and ixp_km > 500:
        reasons.append(
            f"Distant from backbone internet ({ixp_km:.0f} km to nearest IXP)"
        )

    tx_km = row.get("dist_transmission_km")
    if pd.notnull(tx_km) and tx_km > 75:
        reasons.append(f"Far from high-voltage transmission ({tx_km:.0f} km)")

    sub_km = row.get("dist_substation_km")
    if pd.notnull(sub_km) and sub_km > 75:
        reasons.append(f"Far from electrical substation ({sub_km:.0f} km)")

    cdd = row.get("annual_cdd")
    if pd.notnull(cdd) and cdd > 2000:
        reasons.append(f"High cooling load ({cdd:.0f} CDD annually)")

    pop = row.get("pop_within_25mi")
    if pd.notnull(pop) and pop < 5000:
        reasons.append("Very low population — limited workforce access")

    slope = row.get("mean_slope_deg")
    if pd.notnull(slope) and slope > 10:
        reasons.append(f"Steep terrain ({slope:.1f}° avg slope)")

    if row.get("high_ej_burden"):
        reasons.append("High environmental justice burden area")

    viability = row.get("viability_score", 0) or 0
    if viability >= 0.6 and not reasons:
        if pd.notnull(ixp_km) and ixp_km < 100:
            reasons.append(
                f"Strong backbone connectivity ({ixp_km:.0f} km to IXP)"
            )
        elec = row.get("elec_price_mwh")
        if pd.notnull(elec) and elec < 60:
            reasons.append(f"Low industrial electricity cost (${elec:.0f}/MWh)")
        if row.get("cooling_favorable"):
            reasons.append("Climate favorable for free-air cooling")

    if not reasons:
        if viability < 0.15:
            reasons.append(
                "Low combined infrastructure and market accessibility score"
            )
        else:
            reasons.append("Moderate suitability — review feature details")

    return reasons


def export_cell_explanations(
    scored_df: pd.DataFrame,
    explanations_path: Path,
) -> tuple[dict[str, list[str]], Path]:
    log(f"\nBuilding cell explanations ({len(scored_df):,} cells)...")
    records = scored_df.to_dict("records")
    explanations: dict[str, list[str]] = {}

    for i, row in enumerate(records):
        if i % 50000 == 0:
            log(f"  {i:,} / {len(records):,}...")
        explanations[row["h3_index"]] = explain_cell(row)

    explanations_path.parent.mkdir(parents=True, exist_ok=True)
    with explanations_path.open("w") as handle:
        json.dump(explanations, handle)

    size_mb = os.path.getsize(explanations_path) / 1_048_576
    log(
        f"Cell explanations written: {len(explanations):,} cells "
        f"({size_mb:.1f} MB)"
    )
    return explanations, explanations_path


def compute_final_scores(
    master_path: Path | None = None,
    scores_path: Path | None = None,
    output_path: Path | None = None,
    weights: dict[str, float] | None = None,
    geojson_path: Path | None = None,
    basemap_path: Path | None = None,
    explanations_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], int, float, dict[str, list[str]]]:
    master_path = master_path or DEFAULT_MASTER
    scores_path = scores_path or DEFAULT_SCORES
    output_path = output_path or DEFAULT_OUTPUT
    geojson_path = geojson_path or DEFAULT_GEOJSON
    basemap_path = basemap_path or DEFAULT_BASEMAP
    explanations_path = explanations_path or DEFAULT_EXPLANATIONS
    weights = weights or DEFAULT_WEIGHTS

    log(f"Loading master features from {master_path.name}...")
    master = pd.read_parquet(master_path)
    log(f"  {len(master):,} cells")

    log(f"Loading viability scores from {scores_path.name}...")
    scores = pd.read_csv(scores_path)
    score_cols = [
        "h3_index",
        "viability_score",
        "has_dc",
        "is_soft_positive",
    ]
    scores = scores[[c for c in score_cols if c in scores.columns]]

    df = master.merge(scores, on="h3_index", how="left", suffixes=("", "_score"))
    if "has_dc_score" in df.columns:
        df["has_dc"] = df["has_dc"].fillna(df["has_dc_score"]).astype(int)
        df = df.drop(columns=["has_dc_score"])
    df["viability_score"] = df["viability_score"].fillna(0.0)
    df["is_soft_positive"] = df.get("is_soft_positive", 0).fillna(0).astype(int)

    missing = [c for c in HARD_EXCLUSION_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Master parquet missing required columns: {missing}")

    df["pop_within_25mi_norm"] = normalize_population(df["pop_within_25mi"])
    df["impact_penalty"] = compute_impact_penalty(df, weights)
    df["final_score"] = (df["viability_score"] * (1.0 - df["impact_penalty"])).clip(
        0.0, 1.0
    )
    df["display_category"] = display_category(df)

    output_cols = [
        "h3_index",
        "lat",
        "lon",
        "has_dc",
        "viability_score",
        "impact_penalty",
        "final_score",
        "display_category",
        "in_protected_area",
        "in_wetland",
        "in_floodplain_100yr",
        "in_tribal_land",
        "in_critical_habitat",
        "in_high_wildfire",
        "in_high_seismic",
        "high_water_stress",
        "is_soft_positive",
    ]
    optional_cols = ["tribal_name", "critical_habitat_species_count"]
    output_cols.extend(c for c in optional_cols if c in df.columns)
    result = df[[c for c in output_cols if c in df.columns]].copy()

    log(f"Writing {output_path.name}...")
    write_parquet_safe(result, output_path)

    model_version = DEFAULT_MODEL_VERSION
    baseline_results = MODELS_DIR / "baseline_results.json"
    if baseline_results.exists():
        with baseline_results.open() as handle:
            model_version = json.load(handle).get("model_version", model_version)

    features, _ = export_top_candidates_geojson(df, geojson_path, model_version)
    basemap_count, _, basemap_threshold = export_basemap_csv(df, basemap_path)
    explanations, _ = export_cell_explanations(df, explanations_path)

    return result, df, features, basemap_count, basemap_threshold, explanations


def print_summary(
    result: pd.DataFrame,
    output_path: Path,
    geojson_path: Path,
    geojson_features: list[dict],
    basemap_path: Path,
    basemap_count: int,
    basemap_threshold: float,
    explanations_path: Path,
    explanations: dict[str, list[str]],
) -> None:
    log("\n" + "=" * 60)
    log("Final scoring complete")
    log("=" * 60)

    log("\nCell category breakdown:")
    counts = result["display_category"].value_counts()
    for category, count in counts.items():
        pct = 100.0 * count / len(result)
        log(f"  {category}: {count:,} ({pct:.2f}%)")

    hard = result["display_category"] == "Hard Exclusion"
    if hard.any() and "in_tribal_land" in result.columns:
        tribal = int((hard & (result["in_tribal_land"] == 1)).sum())
        crithab = int((hard & (result["in_critical_habitat"] == 1)).sum())
        log(f"\nHard Exclusion breakdown:")
        log(f"  in_tribal_land=1:      {tribal:,}")
        log(f"  in_critical_habitat=1: {crithab:,}")

    candidates = result[
        (result["display_category"] == "Candidate")
        & (result["has_dc"] == 0)
    ].sort_values("final_score", ascending=False)

    log("\nTop 20 candidate cells (final_score, lat/lon):")
    top = candidates.head(20)
    if top.empty:
        log("  (none)")
    else:
        for _, row in top.iterrows():
            log(
                f"  {row['h3_index']}  "
                f"{row['lat']:.4f}, {row['lon']:.4f}  "
                f"score={row['final_score']:.4f}"
            )

    log("\nArtifacts:")
    artifacts = [
        (geojson_path, len(geojson_features), "full properties"),
        (
            basemap_path,
            basemap_count,
            f"H3 indexes, viability >= {basemap_threshold}",
        ),
        (explanations_path, len(explanations), "reason strings"),
        (MODELS_DIR / "baseline_results.json", None, None),
        (MODELS_DIR / "feature_importances.csv", None, None),
        (output_path, None, None),
    ]
    for path, count, label in artifacts:
        if not path.exists():
            continue
        size = _format_bytes(path.stat().st_size)
        if count is not None and label:
            log(f"  {path} ({size}, {count:,} cells — {label})")
        else:
            log(f"  {path} ({size})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply community impact penalties to model viability scores"
    )
    parser.add_argument(
        "--master",
        type=Path,
        default=DEFAULT_MASTER,
        help="Master features parquet",
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=DEFAULT_SCORES,
        help="Model viability scores CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Scored cells parquet output",
    )
    parser.add_argument(
        "--geojson",
        type=Path,
        default=DEFAULT_GEOJSON,
        help="Top candidates GeoJSON output for OSDCI frontend",
    )
    parser.add_argument(
        "--basemap",
        type=Path,
        default=DEFAULT_BASEMAP,
        help="Basemap CSV output for OSDCI frontend (H3 indexes, viability >= 0.02)",
    )
    parser.add_argument(
        "--explanations",
        type=Path,
        default=DEFAULT_EXPLANATIONS,
        help="Cell explanations JSON output for OSDCI frontend",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    try:
        result, _, geojson_features, basemap_count, basemap_threshold, explanations = (
            compute_final_scores(
                master_path=args.master,
                scores_path=args.scores,
                output_path=args.output,
                geojson_path=args.geojson,
                basemap_path=args.basemap,
                explanations_path=args.explanations,
            )
        )
        print_summary(
            result,
            args.output,
            args.geojson,
            geojson_features,
            args.basemap,
            basemap_count,
            basemap_threshold,
            args.explanations,
            explanations,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        log(f"Error: {exc}")
        return 1
    elapsed = time.perf_counter() - started
    log(f"\nRuntime: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
