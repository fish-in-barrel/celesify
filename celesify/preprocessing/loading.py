"""
Phase 1: Data Loading

Handles CSV file discovery, Kaggle API fallback download, and raw data loading.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from celesify.core.constants import (
    KAGGLE_DATASET,
    KAGGLE_EXPECTED_FILE,
)
from celesify.core.logging import log


SERVICE = "preprocessing"


def has_kaggle_credentials() -> bool:
    """Check if Kaggle API credentials are available in environment."""
    return bool(os.getenv("KAGGLE_USERNAME")) and bool(os.getenv("KAGGLE_KEY"))


def download_from_kaggle(raw_dir: Path) -> list[Path]:
    """
    Download dataset from Kaggle if not present locally.

    Args:
        raw_dir: Directory to download into

    Returns:
        List of CSV files found in raw_dir after download

    Raises:
        RuntimeError: If credentials missing or download fails
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    raw_dir.mkdir(parents=True, exist_ok=True)

    if not has_kaggle_credentials():
        raise RuntimeError(
            "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY "
            "for the preprocessing container to enable auto-download."
        )

    log(SERVICE, f"No local CSV detected. Downloading dataset {KAGGLE_DATASET} via Kaggle API.")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(raw_dir), unzip=True, quiet=False)

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(
            f"Kaggle download completed but no CSV files were found in {raw_dir}."
        )

    preferred_path = raw_dir / KAGGLE_EXPECTED_FILE
    if preferred_path.exists():
        log(SERVICE, f"Kaggle download complete. Found expected file: {preferred_path.name}")
    else:
        log(
            SERVICE,
            "Kaggle download complete. Expected file name was not found; "
            f"available files: {[p.name for p in csv_files]}",
        )

    return csv_files


def select_csv_file(raw_dir: Path) -> Path:
    """
    Select a CSV file to load from raw_dir, downloading from Kaggle if necessary.

    Args:
        raw_dir: Directory containing CSV files (or empty if download needed)

    Returns:
        Path to selected CSV file
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(raw_dir.glob("*.csv")) if raw_dir.exists() else []
    if not csv_files:
        csv_files = download_from_kaggle(raw_dir)

    selected_csv = csv_files[0]
    if len(csv_files) > 1:
        log(SERVICE, f"Found {len(csv_files)} CSV files; selecting {selected_csv.name} (alphabetical first).")
    else:
        log(SERVICE, f"Found 1 CSV file: {selected_csv.name}")

    return selected_csv


def load_raw_dataframe(csv_path: Path) -> pd.DataFrame:
    """Load raw DataFrame from CSV file."""
    df = pd.read_csv(csv_path)
    log(SERVICE, f"Loaded {csv_path.name} with shape {df.shape}")
    return df
