"""
Data loading and preparation for model training.

This module handles:
- Loading Parquet datasets from the preprocessing pipeline
- Selecting feature columns and target variables
- Splitting data for different model variants (clean vs engineered)
- Subsampling data for quick validation runs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from celesify.core.constants import CLASS_LABEL_ORDER, RANDOM_STATE
from celesify.core.logging import log

SERVICE = "training"


def load_split_variant(
    processed_dir: Path,
    candidates: list[tuple[str, str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, str, str, str]:
    """
    Load train/test split from available Parquet files.

    Tries candidates in order until both files exist. Returns the data
    along with variant metadata for tracking which split was used.

    Args:
        processed_dir: Directory containing Parquet files (from preprocessing).
        candidates: List of (train_file, test_file, variant_name) tuples
                   tried in order.

    Returns:
        Tuple of (train_df, test_df, variant_name, train_filename, test_filename).

    Raises:
        FileNotFoundError: If no matching pair of files is found.
    """
    for train_name, test_name, variant_name in candidates:
        train_file = processed_dir / train_name
        test_file = processed_dir / test_name
        if train_file.exists() and test_file.exists():
            return (
                pd.read_parquet(train_file),
                pd.read_parquet(test_file),
                variant_name,
                train_file.name,
                test_file.name,
            )

    candidate_list = ", ".join(f"{train}/{test}" for train, test, _ in candidates)
    raise FileNotFoundError(f"No matching parquet split found. Checked: {candidate_list}")


def load_preprocessing_report(report_file: Path) -> dict[str, Any]:
    """
    Load metadata report from preprocessing stage.

    Args:
        report_file: Path to preprocessing_report.json.

    Returns:
        Dictionary of preprocessing metadata, or empty dict if file missing.
    """
    if report_file.exists():
        data = json.loads(report_file.read_text(encoding="utf-8"))
        log(SERVICE, "Loaded preprocessing_report.json for metadata and imbalance guidance.")
        return data

    log(SERVICE, "preprocessing_report.json not found; continuing with safe defaults.")
    return {}


def load_datasets(
    processed_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """
    Load both clean and engineered dataset variants.

    Tries to load both the "clean" variant (minimal features) and the
    "engineered" variant (with feature engineering). Falls back to alternates
    if primary variants are unavailable.

    Args:
        processed_dir: Directory containing Parquet files.

    Returns:
        Tuple of (clean_train, clean_test, engineered_train, engineered_test, metadata).
        Metadata dict contains: variant_clean, variant_engineered, split file names.

    Raises:
        ValueError: If required 'class' column is missing from splits.
        FileNotFoundError: If no matching parquet splits are found.
    """
    clean_candidates = [
        ("train_clean.parquet", "test_clean.parquet", "cleaned"),
        ("train.parquet", "test.parquet", "engineered"),
    ]
    engineered_candidates = [
        ("train.parquet", "test.parquet", "engineered"),
        ("train_clean.parquet", "test_clean.parquet", "cleaned"),
    ]

    clean_train, clean_test, clean_variant, clean_train_name, clean_test_name = load_split_variant(
        processed_dir,
        clean_candidates,
    )
    engineered_train, engineered_test, engineered_variant, engineered_train_name, engineered_test_name = load_split_variant(
        processed_dir,
        engineered_candidates,
    )

    # Validate required columns
    for df_name, df in [
        ("clean_train", clean_train),
        ("clean_test", clean_test),
        ("engineered_train", engineered_train),
        ("engineered_test", engineered_test),
    ]:
        if "class" not in df.columns:
            raise ValueError(f"Expected target column 'class' in {df_name}, got columns: {df.columns.tolist()}")

    metadata = {
        "variant_clean": clean_variant,
        "variant_engineered": engineered_variant,
        "clean_train_file": clean_train_name,
        "clean_test_file": clean_test_name,
        "engineered_train_file": engineered_train_name,
        "engineered_test_file": engineered_test_name,
    }

    return clean_train, clean_test, engineered_train, engineered_test, metadata


def extract_features_and_target(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    """
    Separate features and target variable from datasets.

    Args:
        train_df: Training dataframe with 'class' target column.
        test_df: Test dataframe with 'class' target column.

    Returns:
        Tuple of (x_train, y_train, x_test, y_test, feature_columns).

    Raises:
        ValueError: If no feature columns exist (only 'class' column).
    """
    feature_columns = [str(col) for col in train_df.columns if col != "class"]

    if not feature_columns:
        raise ValueError("No feature columns found after removing target column 'class'.")

    x_train = cast(pd.DataFrame, train_df[feature_columns])
    y_train = train_df["class"].astype(int)
    x_test = cast(pd.DataFrame, test_df[feature_columns])
    y_test = test_df["class"].astype(int)

    return x_train, y_train, x_test, y_test, feature_columns


def subsample_train_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified subsampling of training data while preserving class proportions.

    Used for quick validation runs. Test set is not modified.

    Args:
        train_df: Original training dataframe.
        test_df: Original test dataframe (returned unchanged).
        target_rows: Target number of training rows to sample.

    Returns:
        Tuple of (subsampled_train_df, test_df).
    """
    if target_rows <= 0 or len(train_df) <= target_rows:
        return train_df, test_df

    class_counts = cast(pd.Series, train_df["class"].value_counts().sort_index())
    target_counts = cast(pd.Series, ((class_counts / len(train_df)) * target_rows).round().astype(int))
    target_counts[target_counts < 1] = 1

    # Reduce from largest classes if we exceed target
    while int(target_counts.sum()) > target_rows:
        reducible = target_counts[target_counts > 1]
        if reducible.empty:
            break
        reducible_index = reducible.index[0]
        target_counts.loc[reducible_index] = int(target_counts.loc[reducible_index]) - 1

    # Sample stratified by class
    sampled_index_parts: list[int] = []
    for cls, count in target_counts.items():
        class_slice = train_df[train_df["class"] == cls]
        take_n = min(int(count), len(class_slice))
        sampled_index_parts.extend(
            class_slice.sample(n=take_n, random_state=RANDOM_STATE, replace=False).index.tolist()
        )

    sampled_indices = pd.Index(sampled_index_parts)
    sampled_train = train_df.loc[sampled_indices].sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    log(SERVICE, f"Subsampled train from {len(train_df)} to {len(sampled_train)} rows (target={target_rows})")

    return sampled_train, test_df


def get_imbalance_recommendation(preprocess_report: dict) -> bool:
    """
    Determine if class imbalance warrants using balanced class weights.

    Checks preprocessing report for imbalance recommendation or computes
    from majority/minority ratio (threshold: > 2.0).

    Args:
        preprocess_report: Metadata from preprocessing stage.

    Returns:
        True if class weighting should be balanced, False otherwise.
    """
    # Check explicit recommendation first
    recommendation = preprocess_report.get("imbalance_recommendation")
    if isinstance(recommendation, str) and "balanced" in recommendation.lower():
        return True

    nested_assessment = preprocess_report.get("imbalance_assessment")
    if isinstance(nested_assessment, dict):
        nested_recommendation = nested_assessment.get("recommendation")
        if isinstance(nested_recommendation, str) and "balanced" in nested_recommendation.lower():
            return True

    # Try to extract ratio from various locations in report
    ratio = preprocess_report.get("majority_minority_ratio")
    if ratio is None:
        ratio = preprocess_report.get("imbalance_ratio")
    if ratio is None and isinstance(preprocess_report.get("class_balance"), dict):
        ratio = preprocess_report["class_balance"].get("majority_minority_ratio")
    if ratio is None and isinstance(nested_assessment, dict):
        ratio = nested_assessment.get("majority_to_minority_ratio")

    # Threshold: if ratio > 2.0, recommend balanced weighting
    if ratio is not None:
        try:
            return float(ratio) > 2.0
        except (TypeError, ValueError):
            return False

    return False


def get_class_mapping(preprocess_report: dict) -> dict[str, int]:
    """
    Extract class label to integer encoding from preprocessing report.

    Args:
        preprocess_report: Metadata from preprocessing stage.

    Returns:
        Dictionary mapping class names (str) to integer codes.
    """
    from celesify.core.constants import CLASS_ENCODING

    class_mapping = preprocess_report.get("class_mapping")
    if class_mapping is None:
        class_mapping = preprocess_report.get("target_encoding")

    if isinstance(class_mapping, dict):
        parsed = {}
        for name, encoded in class_mapping.items():
            try:
                parsed[str(name)] = int(encoded)
            except (TypeError, ValueError):
                continue
        if parsed:
            return parsed

    return CLASS_ENCODING


def log_dataset_info(
    clean_train: pd.DataFrame,
    clean_test: pd.DataFrame,
    engineered_train: pd.DataFrame,
    engineered_test: pd.DataFrame,
) -> None:
    """Log summary statistics for loaded datasets."""
    log(
        SERVICE,
        "Loaded datasets: "
        f"clean_train={clean_train.shape}, clean_test={clean_test.shape}, "
        f"engineered_train={engineered_train.shape}, engineered_test={engineered_test.shape}",
    )
