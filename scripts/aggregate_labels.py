#!/usr/bin/env python3
"""Aggregate IM3 facility labels from one row per facility to one row per H3 cell."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_INPUT = PROCESSED_DIR / "labels_im3.parquet"
DEFAULT_OUTPUT = PROCESSED_DIR / "labels_aggregated_res6.parquet"

HYPERSCALERS = {
    "Google",
    "Amazon",
    "Microsoft",
    "Meta",
    "Apple",
    "AWS",
    "Google LLC",
    "Amazon Web Services",
    "Microsoft Corporation",
    "Meta Platforms",
    "Facebook",
    "Google Inc",
}

OUTPUT_COLUMNS = [
    "h3_index",
    "has_dc",
    "facility_count",
    "max_sqft",
    "total_sqft",
    "has_hyperscaler",
    "dominant_type",
    "operators",
    "state_abb",
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


def is_hyperscaler(operator: object) -> int:
    if pd.isna(operator):
        return 0
    op = str(operator).strip().lower()
    return int(any(h.lower() in op for h in HYPERSCALERS))


def dominant_type(types: pd.Series) -> str:
    values = set(types.dropna().astype(str))
    if "campus" in values:
        return "campus"
    if "building" in values:
        return "building"
    return "point"


def inspect_labels(df: pd.DataFrame) -> None:
    log(f"Shape: {len(df):,} rows x {len(df.columns)} columns")
    log(f"Columns: {', '.join(df.columns.tolist())}")

    unique_h3 = df["h3_index"].nunique()
    log(f"Unique h3_index values: {unique_h3:,} (total rows: {len(df):,})")

    if unique_h3 == len(df):
        log(
            "\nWarning: labels are already cell-level (one row per h3_index). "
            "Nothing to aggregate — exiting."
        )
        sys.exit(0)

    log("\nSample (5 rows):")
    log(df.head(5).to_string())

    if "type" in df.columns:
        log("\nValue counts of type:")
        for label, count in df["type"].value_counts().items():
            log(f"  {label}: {count:,}")

    if "operator" in df.columns:
        log("\nTop 10 operators:")
        for label, count in df["operator"].value_counts().head(10).items():
            log(f"  {label}: {count:,}")


def aggregate_labels(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("h3_index", sort=True)

    out = pd.DataFrame(
        {
            "h3_index": grouped.size().index.astype(str),
            "has_dc": 1,
            "facility_count": grouped.size().values.astype(int),
            "max_sqft": grouped["sqft"].max().values,
            "total_sqft": grouped["sqft"].sum(min_count=1).values,
            "has_hyperscaler": grouped["operator"].apply(
                lambda ops: int(ops.map(is_hyperscaler).any())
            ).values,
            "dominant_type": grouped["type"].apply(dominant_type).values,
            "operators": grouped["operator"].apply(
                lambda ops: ", ".join(sorted(ops.dropna().astype(str).unique()))
            ).values,
        }
    )

    if "state_abb" in df.columns:
        out["state_abb"] = grouped["state_abb"].apply(
            lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA
        ).values
    else:
        out["state_abb"] = pd.NA

    return out[OUTPUT_COLUMNS]


def run_sanity_checks(result: pd.DataFrame, input_rows: int) -> None:
    checks = [
        ((result["has_dc"] == 1).all(), "All has_dc values must be 1"),
        (result["h3_index"].is_unique, "No duplicate h3_index values in output"),
        ((result["facility_count"] >= 1).all(), "facility_count >= 1 for all rows"),
        (len(result) <= input_rows, "Output row count must be <= input row count"),
    ]
    failed = [msg for ok, msg in checks if not ok]
    if failed:
        log("\nSanity check FAILED:")
        for msg in failed:
            log(f"  - {msg}")
        sys.exit(1)
    log("\nSanity checks passed.")


def print_summary(result: pd.DataFrame, input_rows: int, output_path: Path) -> None:
    colocated = (result["facility_count"] > 1).sum()
    hyperscaler = int(result["has_hyperscaler"].sum())
    hyperscaler_pct = 100.0 * hyperscaler / len(result) if len(result) else 0.0

    log("\n" + "=" * 60)
    log("Label aggregation complete")
    log("=" * 60)
    log(f"Input rows (facilities):           {input_rows:,}")
    log(f"Output rows (H3 cells with DCs):    {len(result):,}")
    log(f"Cells with facility_count > 1:      {colocated:,}")
    log(f"Max facility_count in any cell:     {result['facility_count'].max()}")
    log(f"has_hyperscaler=1: {hyperscaler:,} ({hyperscaler_pct:.2f}%)")

    log("\ndominant_type breakdown:")
    for label, count in result["dominant_type"].value_counts().items():
        log(f"  {label}: {count:,}")

    max_sqft = pd.to_numeric(result["max_sqft"], errors="coerce").dropna()
    if not max_sqft.empty:
        log("\nmax_sqft (non-null):")
        log(f"  min:    {max_sqft.min():,.0f}")
        log(f"  median: {max_sqft.median():,.0f}")
        log(f"  mean:   {max_sqft.mean():,.0f}")
        log(f"  max:    {max_sqft.max():,.0f}")

    log("\nTop 10 cells by facility_count:")
    for _, row in result.nlargest(10, "facility_count").iterrows():
        log(
            f"  {row['h3_index']}: count={row['facility_count']} "
            f"type={row['dominant_type']} operators={row['operators'][:60]}"
        )

    top_sqft = result.dropna(subset=["max_sqft"]).nlargest(10, "max_sqft")
    if not top_sqft.empty:
        log("\nTop 10 cells by max_sqft:")
        for _, row in top_sqft.iterrows():
            log(
                f"  {row['h3_index']}: max_sqft={row['max_sqft']:,.0f} "
                f"count={row['facility_count']} state={row['state_abb']}"
            )

    if result["state_abb"].notna().any():
        log("\nTop 10 states by cell count:")
        for state, count in result["state_abb"].value_counts().head(10).items():
            log(f"  {state}: {count:,}")

    if output_path.exists():
        log(f"\nOutput: {output_path} ({_format_bytes(output_path.stat().st_size)})")


def aggregate_labels_file(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    input_path = input_path or DEFAULT_INPUT
    output_path = output_path or DEFAULT_OUTPUT

    if not input_path.exists():
        raise FileNotFoundError(f"Labels file not found: {input_path}")

    log(f"Loading labels from {input_path.name}...")
    df = pd.read_parquet(input_path)
    inspect_labels(df)

    input_rows = len(df)
    result = aggregate_labels(df)

    log(f"\nWriting {output_path.name}...")
    write_parquet_safe(result, output_path)

    run_sanity_checks(result, input_rows)
    print_summary(result, input_rows, output_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate IM3 facility labels to H3 cell level"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Facility-level labels parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Aggregated cell-level labels parquet",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        aggregate_labels_file(args.input, args.output)
    except (FileNotFoundError, ValueError, OSError) as exc:
        log(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
