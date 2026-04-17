"""
Phase 2: Data Cleaning

Handles schema validation, missing value handling, malformed data detection,
target encoding, and skewness assessment.
"""

from __future__ import annotations
from typing import Hashable

import pandas as pd

from celesify.core.constants import (
    CLASS_ENCODING,
    SKEW_CHECK_COLUMNS,
    SKEW_THRESHOLD,
    TARGET_COLUMN,
    NON_INFORMATIVE_COLUMNS,
)
from celesify.core.logging import log


SERVICE = "preprocessing"


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that DataFrame contains all required columns."""
    required_columns = {
        TARGET_COLUMN,
        "alpha",
        "delta",
        "u",
        "g",
        "r",
        "i",
        "z",
        "redshift",
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def drop_non_informative_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove non-informative columns (IDs, plate numbers, etc.).

    Returns:
        Cleaned DataFrame and list of dropped column names
    """
    dropped_columns = [col for col in NON_INFORMATIVE_COLUMNS if col in df.columns]
    df_cleaned = df.drop(columns=dropped_columns)
    log(SERVICE, f"Dropped non-informative columns: {dropped_columns}")
    return df_cleaned, dropped_columns


def _numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get numeric columns to check for skew/malformation."""
    return [col for col in SKEW_CHECK_COLUMNS if col in df.columns]


def _coerce_numeric_features(df: pd.DataFrame, numeric_columns: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Coerce numeric columns to numeric type, tracking malformed values.

    Returns:
        Coerced DataFrame and count of malformed values per column
    """
    if not numeric_columns:
        return df.copy(), {}

    coerced = df.copy()
    malformed_by_column: dict[str, int] = {}
    for column in numeric_columns:
        original = coerced[column]
        numeric_series = pd.to_numeric(original, errors="coerce")
        malformed_mask = original.notna() & numeric_series.isna()
        malformed_by_column[column] = int(malformed_mask.sum())
        coerced[column] = numeric_series
    return coerced, malformed_by_column


def _clean_target(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, list[str]]:
    """
    Normalize and validate target column.

    Returns:
        Tuple of (normalized_series, missing_mask, unknown_mask, unknown_labels)
    """
    normalized = series.astype("string").str.strip().str.upper()
    missing_mask = normalized.isna()
    unknown_mask = (~missing_mask) & (~normalized.isin(CLASS_ENCODING.keys()))
    unknown_labels = sorted(normalized.loc[unknown_mask].dropna().unique().tolist())
    return normalized, missing_mask, unknown_mask, unknown_labels


def handle_missing_and_malformed_values(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int], int, int, int, float]:
    """
    Handle missing and malformed numeric values by dropping affected rows.
    Encode target labels to integers.

    Args:
        df: Input DataFrame with raw data

    Returns:
        Tuple of (cleaned_df, missing_by_col, malformed_by_col, rows_removed_missing,
                 rows_removed_malformed, rows_removed_total, removal_pct)
    """
    rows_before_cleaning = len(df)
    missing_by_col = {col: int(count) for col, count in df.isna().sum().items() if int(count) > 0}

    numeric_columns = _numeric_feature_columns(df)
    original_missing_row_mask = (
        df[numeric_columns + [TARGET_COLUMN]].isna().any(axis=1) if numeric_columns
        else df[TARGET_COLUMN].isna()
    )

    coerced_df, malformed_by_col = _coerce_numeric_features(df, numeric_columns)
    cleaned_target, missing_target_mask, unknown_target_mask, unknown_labels = _clean_target(
        coerced_df[TARGET_COLUMN]
    )

    malformed_numeric_row_mask = (
        coerced_df[numeric_columns].isna().any(axis=1) & (~original_missing_row_mask)
        if numeric_columns
        else pd.Series(False, index=coerced_df.index)
    )

    rows_to_drop = original_missing_row_mask | malformed_numeric_row_mask | missing_target_mask | unknown_target_mask

    rows_removed_missing = int(original_missing_row_mask.sum())
    rows_removed_malformed = int((malformed_numeric_row_mask | unknown_target_mask).sum())
    rows_removed_total = int(rows_to_drop.sum())
    removal_pct = float((rows_removed_total / rows_before_cleaning) * 100) if rows_before_cleaning else 0.0

    if missing_by_col:
        log(SERVICE, f"Missing values found by column: {missing_by_col}")
    else:
        log(SERVICE, "No missing values detected.")
    if malformed_by_col:
        log(SERVICE, f"Malformed numeric values found by column: {malformed_by_col}")
    if unknown_labels:
        log(SERVICE, f"Malformed target labels removed: {unknown_labels}")

    cleaned_df = coerced_df.loc[~rows_to_drop].copy()
    cleaned_df[TARGET_COLUMN] = cleaned_target.loc[~rows_to_drop].map(CLASS_ENCODING).astype("int64")

    # Ensure dict keys are strings for consistent typing
    missing_by_col = {str(col): count for col, count in missing_by_col.items()}
    malformed_by_col = {str(col): count for col, count in malformed_by_col.items()}

    return (
        cleaned_df,
        missing_by_col,
        malformed_by_col,
        rows_removed_missing,
        rows_removed_malformed,
        rows_removed_total,
        removal_pct,
    )

def assess_skew_and_log(df: pd.DataFrame) -> dict[Hashable, float]:
    """
    Assess and log skewness of numeric features. No transform is applied.

    Returns:
        Dictionary of column name → skew value
    """
    available_skew_cols = _numeric_feature_columns(df)
    skew_raw = df[available_skew_cols].skew(numeric_only=True)

    if isinstance(skew_raw, pd.Series):
        skew_values = {col: float(val) for col, val in skew_raw.items() if pd.notna(val)}
    else:
        skew_values = {}

    high_skew_columns = {col: val for col, val in skew_values.items() if abs(val) > SKEW_THRESHOLD}

    log(
        SERVICE,
        "Skew check complete; no log transform applied. "
        f"Columns above threshold {SKEW_THRESHOLD}: {high_skew_columns}",
    )

    return skew_values


def class_distribution(series: pd.Series) -> dict[str, int]:
    """Get class distribution as {label: count}."""
    counts = series.value_counts().sort_index()
    return {str(k): int(v) for k, v in counts.items()}


def class_proportions(series: pd.Series) -> dict[str, float]:
    """Get class proportions as {label: fraction}."""
    proportions = series.value_counts(normalize=True).sort_index()
    return {str(k): float(v) for k, v in proportions.items()}
