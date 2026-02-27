#!/usr/bin/env python3
"""
Prepare canonical Walgreens model input files under data/.

This copies/converts the latest uploaded platform dataset files into:
  - data/financial_data.csv
  - data/health_data.csv
  - data/population_data.csv
  - data/insurance.csv
  - data/houseprice.csv
  - data/pharmacy_data.csv

Source files are never modified.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_latest_version_dir(root: Path) -> Path:
    latest_path = root / "raw_data/datasets/pharmacy_data/LATEST.json"
    if not latest_path.exists():
        raise FileNotFoundError(f"Missing latest version pointer: {latest_path}")
    latest = json.loads(latest_path.read_text()).get("latest_version")
    if not latest:
        raise ValueError(f"'latest_version' missing in {latest_path}")
    version_dir = root / "raw_data/datasets/pharmacy_data/versions" / latest
    if not version_dir.exists():
        raise FileNotFoundError(f"Version directory not found: {version_dir}")
    return version_dir


def _pick_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _resolve_sources(version_dir: Path) -> dict[str, Path]:
    files_dir = version_dir / "files"
    if not files_dir.exists():
        raise FileNotFoundError(f"Missing files directory: {files_dir}")

    cfg = {}
    cfg_path = version_dir / "dataset_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
    uploaded = (cfg.get("uploaded_files") or {}) if isinstance(cfg, dict) else {}

    def slot(name: str) -> Path | None:
        fname = uploaded.get(name)
        if not fname:
            return None
        p = files_dir / fname
        return p if p.exists() else None

    financial = _pick_existing(
        [p for p in [slot("financial"), files_dir / "financial_data.csv"] if p is not None]
    )
    health = _pick_existing(
        [p for p in [slot("health"), files_dir / "health_data.csv"] if p is not None]
    )
    population = _pick_existing(
        [p for p in [slot("population"), files_dir / "population_data.csv"] if p is not None]
    )
    insurance = _pick_existing(
        [
            p
            for p in [
                files_dir / "custom_insurance.csv",
                files_dir / "insurance.csv",
            ]
            if p is not None
        ]
    )
    houseprice = _pick_existing(
        [
            p
            for p in [
                files_dir / "custom_houseprice.csv",
                files_dir / "houseprice.csv",
            ]
            if p is not None
        ]
    )
    pharmacy = _pick_existing(
        [
            p
            for p in [
                slot("pharmacy"),
                files_dir / "pharmacy_data.csv",
                files_dir / "pharmacy.csv",
            ]
            if p is not None
        ]
    )

    required = {
        "financial_data.csv": financial,
        "health_data.csv": health,
        "population_data.csv": population,
        "pharmacy_data.csv": pharmacy,
    }
    missing_required = [name for name, src in required.items() if src is None]
    if missing_required:
        raise FileNotFoundError(
            "Missing required source files in latest version: " + ", ".join(missing_required)
        )

    out = {
        "financial_data.csv": financial,
        "health_data.csv": health,
        "population_data.csv": population,
        "pharmacy_data.csv": pharmacy,
    }
    if insurance is not None:
        out["insurance.csv"] = insurance
    if houseprice is not None:
        out["houseprice.csv"] = houseprice
    return out


def _copy_or_convert_csv(src: Path, dst: Path, force: bool) -> str:
    if dst.exists() and not force:
        return f"skip (exists): {dst}"

    dst.parent.mkdir(parents=True, exist_ok=True)
    ext = src.suffix.lower()
    if ext in {".xlsx", ".xlsm", ".xls"}:
        df = pd.read_excel(src, dtype=str)
        df.to_csv(dst, index=False)
        return f"converted Excel -> CSV: {src} -> {dst} ({len(df):,} rows)"

    shutil.copy2(src, dst)
    return f"copied: {src} -> {dst}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare data/*.csv inputs for Walgreens model scripts from latest uploaded dataset."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files under data/ if they already exist.",
    )
    args = parser.parse_args()

    root = _repo_root()
    data_dir = root / "data"
    try:
        version_dir = _find_latest_version_dir(root)
        sources = _resolve_sources(version_dir)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Using source version: {version_dir.name}")
    print(f"Writing canonical inputs to: {data_dir}")
    for out_name, src in sources.items():
        dst = data_dir / out_name
        try:
            msg = _copy_or_convert_csv(src, dst, force=args.force)
            print(f"  - {msg}")
        except Exception as exc:
            print(f"ERROR writing {dst}: {exc}")
            return 1

    print("\nDone.")
    print("You can now run:")
    print("  python deployment/walgreens_portfolio/run_complete_system.py --npi data/pharmacy_data.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
