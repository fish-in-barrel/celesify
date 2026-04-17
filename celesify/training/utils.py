"""Utility functions for data loading, class mapping, and environment configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import pandas as pd

from celesify.core.constants import CLASS_ENCODING
from celesify.core.logging import log

SERVICE = "training"

DEFAULT_CLASS_MAP = CLASS_ENCODING


def load_split_variant(
    processed_dir: Path,
    candidates: list[tuple[str, str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, str, str, str]:
    """
    Load a train/test split variant from parquet files.

    Tries candidates in order until both files exist. Raises FileNotFoundError if none found.

    Args:
        processed_dir: Path to processed data directory
        candidates: List of (train_filename, test_filename, variant_name) tuples to try

    Returns:
        (train_df, test_df, variant_name, train_file_name, test_file_name)
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


def get_imbalance_recommendation(preprocess_report: dict) -> bool:
    """
    Extract imbalance recommendation from preprocessing report.

    Checks multiple possible locations in the report structure for a recommendation
    to use class_weight='balanced'.

    Args:
        preprocess_report: Dictionary loaded from preprocessing_report.json

    Returns:
        True if class balancing is recommended; False otherwise
    """
    recommendation = preprocess_report.get("imbalance_recommendation")
    if isinstance(recommendation, str) and "balanced" in recommendation.lower():
        return True

    nested_assessment = preprocess_report.get("imbalance_assessment")
    if isinstance(nested_assessment, dict):
        nested_recommendation = nested_assessment.get("recommendation")
        if isinstance(nested_recommendation, str) and "balanced" in nested_recommendation.lower():
            return True

    ratio = preprocess_report.get("majority_minority_ratio")
    if ratio is None:
        ratio = preprocess_report.get("imbalance_ratio")
    if ratio is None and isinstance(preprocess_report.get("class_balance"), dict):
        ratio = preprocess_report["class_balance"].get("majority_minority_ratio")
    if ratio is None and isinstance(nested_assessment, dict):
        ratio = nested_assessment.get("majority_to_minority_ratio")

    if ratio is not None:
        try:
            return float(ratio) > 2.0
        except (TypeError, ValueError):
            return False
    return False


def get_class_mapping(preprocess_report: dict) -> dict[str, int]:
    """
    Extract class encoding mapping from preprocessing report.

    Checks multiple possible report locations for the class-to-integer mapping.

    Args:
        preprocess_report: Dictionary loaded from preprocessing_report.json

    Returns:
        Dictionary mapping class names to integer IDs; falls back to DEFAULT_CLASS_MAP
    """
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
    return DEFAULT_CLASS_MAP


def get_int_env(name: str, default: int) -> int:
    """
    Read an integer environment variable.

    Logs a warning if the value is invalid and returns default.

    Args:
        name: Environment variable name
        default: Default value if not set or invalid

    Returns:
        Parsed integer value or default
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        log(SERVICE, f"Invalid {name}={raw!r}; using default {default}.")
        return default


def apply_max_train_rows(
    engineered_train_df: pd.DataFrame,
    clean_train_df: pd.DataFrame,
    engineered_feature_columns: list[str],
    clean_feature_columns: list[str],
    max_train_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply stratified row sampling limit to training data if specified.

    Maintains class proportions when subsampling.

    Args:
        engineered_train_df: Full engineered training DataFrame
        clean_train_df: Full clean training DataFrame
        engineered_feature_columns: Engineered feature names
        clean_feature_columns: Clean feature names
        max_train_rows: Maximum rows to keep (0 = no limit)

    Returns:
        Tuple of (engineered_train_sampled, x_train, clean_train_sampled, x_clean_train)
    """
    if max_train_rows <= 0 or len(engineered_train_df) <= max_train_rows:
        x_train = cast(pd.DataFrame, engineered_train_df[engineered_feature_columns])
        x_clean_train = cast(pd.DataFrame, clean_train_df[clean_feature_columns])
        return engineered_train_df, x_train, clean_train_df, x_clean_train

    class_counts = cast(pd.Series, engineered_train_df["class"].value_counts().sort_index())
    target_counts = cast(pd.Series, ((class_counts / len(engineered_train_df)) * max_train_rows).round().astype(int))
    target_counts[target_counts < 1] = 1

    while int(target_counts.sum()) > max_train_rows:
        reducible = target_counts[target_counts > 1]
        if reducible.empty:
            break
        reducible_index = reducible.index[0]
        target_counts.loc[reducible_index] = int(target_counts.loc[reducible_index]) - 1

    sampled_index_parts: list[int] = []
    for cls, count in target_counts.items():
        class_slice = engineered_train_df[engineered_train_df["class"] == cls]
        take_n = min(int(count), len(class_slice))
        sampled_index_parts.extend(class_slice.sample(n=take_n, random_state=42, replace=False).index.tolist())

    sampled_indices = pd.Index(sampled_index_parts)
    sampled_engineered = engineered_train_df.loc[sampled_indices].sample(frac=1.0, random_state=42).reset_index(drop=True)
    sampled_clean = clean_train_df.loc[sampled_indices].sample(frac=1.0, random_state=42).reset_index(drop=True)

    x_train = cast(pd.DataFrame, sampled_engineered[engineered_feature_columns])
    x_clean_train = cast(pd.DataFrame, sampled_clean[clean_feature_columns])

    log(SERVICE, f"Applied TRAINING_MAX_TRAIN_ROWS={max_train_rows}; sampled train rows={len(sampled_engineered)}")

    return sampled_engineered, x_train, sampled_clean, x_clean_train
