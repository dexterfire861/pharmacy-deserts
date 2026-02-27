#!/usr/bin/env python3
"""
Build pharmacy/pharmacist ZIP-detail caches from an NPPES pfile.

Example:
    python scripts/build_npi_provider_cache.py \
      --source "/Users/me/Downloads/NPPES_Data_Dissemination_February_2026_V2/npidata_pfile_20050523-20260208.csv"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.loaders import build_nppes_zip_detail_cache, _resolve_nppes_pfile_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build NPPES provider ZIP-detail caches")
    parser.add_argument(
        "--source",
        default="",
        help="Path to npidata_pfile_*.csv (optional; auto-detected if omitted)",
    )
    parser.add_argument(
        "--cache-dir",
        default="raw_data/npi_cache",
        help="Output cache directory (default: raw_data/npi_cache)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="CSV chunk size for streaming reads (default: 250000)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Optional limit for testing/debugging.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if cache metadata matches source file.",
    )
    args = parser.parse_args()

    source_path = _resolve_nppes_pfile_path(args.source or None)
    if source_path is None:
        print("No NPPES pfile found. Pass --source or set NPPES_PFILE_PATH.", file=sys.stderr)
        return 1

    result = build_nppes_zip_detail_cache(
        csv_path=source_path,
        cache_dir=args.cache_dir,
        chunksize=args.chunksize,
        max_chunks=args.max_chunks,
        force=args.force,
    )

    print("NPPES cache build complete:")
    print(f"  Source: {source_path}")
    print(f"  Pharmacist rows: {result.get('pharmacist_rows', 0):,}")
    print(f"  Pharmacy rows:   {result.get('pharmacy_rows', 0):,}")
    print(f"  Pharmacist file: {result.get('pharmacist_path')}")
    print(f"  Pharmacy file:   {result.get('pharmacy_path')}")
    print(f"  From cache:      {result.get('from_cache', False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

