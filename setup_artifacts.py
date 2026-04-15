#!/usr/bin/env python
"""
Download prebuilt training artifacts from GitHub releases.

Runs at Streamlit Community Cloud startup to fetch model and metrics.
"""

from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

REPO = "fish-in-barrel/celesify"
RELEASE_TAG = "latest"
ARTIFACTS_ZIP = "celesify-artifacts.zip"
OUTPUT_DIR = Path("outputs")


def download_artifacts() -> bool:
    """Download and extract training artifacts from GitHub release."""
    output_dir = OUTPUT_DIR / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip if artifacts already exist
    if (output_dir / "model.joblib").exists():
        print("✓ Artifacts already present, skipping download")
        return True

    try:
        url = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/{ARTIFACTS_ZIP}"
        print(f"Downloading artifacts from {url}...")

        with urlopen(url) as response:
            with open(ARTIFACTS_ZIP, "wb") as out_file:
                out_file.write(response.read())

        # Extract to outputs/
        with zipfile.ZipFile(ARTIFACTS_ZIP, "r") as zip_ref:
            zip_ref.extractall(str(OUTPUT_DIR))

        # Clean up
        Path(ARTIFACTS_ZIP).unlink()
        print("✓ Artifacts downloaded and extracted successfully")
        return True

    except Exception as e:
        print(f"✗ Failed to download artifacts: {e}", file=sys.stderr)
        return False


def verify_artifacts() -> bool:
    """Verify that all required artifacts are present."""
    required_files = [
        "models/model.joblib",
        "models/baseline_metrics.json",
        "models/tuned_metrics.json",
        "models/clean_tuned_metrics.json",
        "models/best_params.json",
        "models/feature_importance.json",
        "processed/preprocessing_report.json",
    ]

    missing = []
    for file_path in required_files:
        full_path = OUTPUT_DIR / file_path
        if not full_path.exists():
            missing.append(file_path)

    if missing:
        print(f"✗ Missing artifacts: {missing}", file=sys.stderr)
        return False

    print(f"✓ All {len(required_files)} artifacts verified")
    return True


if __name__ == "__main__":
    # Only run in cloud environment (check for Streamlit's env var)
    if os.getenv("STREAMLIT_SERVER_HEADLESS") != "true":
        print("Not running in Streamlit Community Cloud, skipping artifact download")
        sys.exit(0)

    if not download_artifacts():
        sys.exit(1)
    if not verify_artifacts():
        sys.exit(1)

    print("\n✓ Artifact setup complete!")
