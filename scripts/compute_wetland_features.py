#!/usr/bin/env python3
"""Derive binary wetland exclusion flags from NLCD pct_wetland."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_RESOLUTION = 6
DEFAULT_NLCD = PROCESSED_DIR / f"features_nlcd_res{DEFAULT_RESOLUTION}.parquet"

PRIMARY_THRESHOLD = 0.25
STRICT_THRESHOLD = 0.50

SOURCE = "NLCD_2024_proxy"
SOURCE_NOTE = (
    "Derived from NLCD classes 90+95. "
    "USFWS NWI not available as national bulk download at time of data collection."
)

OUTPUT_COLUMNS = [
    "h3_index",
    "pct_wetland",
    "in_wetland",
    "in_wetland_strict",
    "in_wetland_any",
    "source",
    "source_note",
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


def read_parquet_safe(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    with path.open("rb") as handle:
        return pd.read_parquet(handle, columns=columns)


def compute_wetland_features(
    nlcd_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    resolution = DEFAULT_RESOLUTION
    if nlcd_path is None:
        nlcd_path = PROCESSED_DIR / f"features_nlcd_res{resolution}.parquet"
    if output_path is None:
        output_path = PROCESSED_DIR / f"features_wetland_res{resolution}.parquet"

    if not nlcd_path.exists():
        raise FileNotFoundError(
            f"NLCD features not found: {nlcd_path}. "
            f"Run: python scripts/compute_nlcd_features.py"
        )

    log(f"Loading NLCD features from {nlcd_path.name}...")
    nlcd = read_parquet_safe(nlcd_path, columns=["h3_index", "pct_wetland"])
    log(f"  {len(nlcd):,} H3 cells")

    pct = pd.to_numeric(nlcd["pct_wetland"], errors="coerce").fillna(0.0)

    result = pd.DataFrame(
        {
            "h3_index": nlcd["h3_index"],
            "pct_wetland": nlcd["pct_wetland"],
            "in_wetland": (pct >= PRIMARY_THRESHOLD).astype(int),
            "in_wetland_strict": (pct >= STRICT_THRESHOLD).astype(int),
            "in_wetland_any": (pct > 0.0).astype(int),
            "source": SOURCE,
            "source_note": SOURCE_NOTE,
        }
    )

    log(f"Writing {output_path.name}...")
    write_parquet_safe(result[OUTPUT_COLUMNS], output_path)
    log("  Parquet write complete")

    print_summary(result, output_path)
    return result[OUTPUT_COLUMNS]


def print_summary(result: pd.DataFrame, output_path: Path) -> None:
    pct = pd.to_numeric(result["pct_wetland"], errors="coerce").fillna(0.0)
    any_wetland = pct > 0.0

    log("\n" + "=" * 60)
    log("NLCD wetland proxy features complete")
    log("=" * 60)
    log(f"Total H3 cells loaded:              {len(result):,}")
    log(f"Cells with pct_wetland > 0:           {any_wetland.sum():,}")

    for col in ("in_wetland", "in_wetland_strict", "in_wetland_any"):
        count = int(result[col].sum())
        pct_cells = 100.0 * count / len(result) if len(result) else 0.0
        log(f"{col}=1:                          {count:,} ({pct_cells:.2f}%)")

    primary = result["in_wetland"] == 1
    if primary.any():
        mean_pct = pct.loc[primary].mean()
        log(f"\nMean pct_wetland (in_wetland=1):    {mean_pct:.4f}")

    if output_path.exists():
        log(f"\nOutput: {output_path} ({_format_bytes(output_path.stat().st_size)})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive binary wetland flags from NLCD pct_wetland"
    )
    parser.add_argument(
        "--nlcd",
        type=Path,
        default=DEFAULT_NLCD,
        help="NLCD features parquet with pct_wetland",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / f"features_wetland_res{DEFAULT_RESOLUTION}.parquet",
        help="Output parquet path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        compute_wetland_features(nlcd_path=args.nlcd, output_path=args.output)
    except (FileNotFoundError, ValueError, OSError) as exc:
        log(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
