"""
Phase 3: Feature Engineering & Splitting

Handles stratified train/test splitting and feature engineering
(color features, band statistics, redshift interactions).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from celesify.core.constants import RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from celesify.core.logging import log


SERVICE = "preprocessing"

# Feature engineering constants
PHOTOMETRIC_BANDS = ["u", "g", "r", "i", "z"]
COLOR_FEATURES = [
    ("u", "g", "color_u_g"),
    ("g", "r", "color_g_r"),
    ("r", "i", "color_r_i"),
    ("i", "z", "color_i_z"),
    ("u", "r", "color_u_r"),
    ("g", "i", "color_g_i"),
    ("g", "z", "color_g_z"),
    ("r", "z", "color_r_z"),
    ("u", "z", "color_u_z"),
]
FEATURE_INTERACTIONS = [
    ("redshift", "color_u_g", "redshift_color_u_g"),
    ("redshift", "color_g_r", "redshift_color_g_r"),
    ("redshift", "color_r_i", "redshift_color_r_i"),
    ("redshift", "color_i_z", "redshift_color_i_z"),
    ("redshift", "color_g_z", "redshift_color_g_z"),
]


def stratified_split(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified train/test split preserving class proportions.

    Args:
        df: Input DataFrame
        target_column: Name of target column
        test_size: Fraction of data for test set (0 < test_size < 1)
        random_state: Seed for reproducibility

    Returns:
        Tuple of (train_df, test_df)
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1)")

    rng = np.random.default_rng(random_state)
    train_index: list[int] = []
    test_index: list[int] = []

    for _, group in df.groupby(target_column, sort=True):
        group_idx = group.index.to_numpy(copy=True)
        rng.shuffle(group_idx)

        split_point = int(np.floor(len(group_idx) * (1 - test_size)))
        if len(group_idx) > 1:
            split_point = max(1, min(split_point, len(group_idx) - 1))

        train_index.extend(group_idx[:split_point].tolist())
        test_index.extend(group_idx[split_point:].tolist())

    train_candidate = df.loc[train_index]
    test_candidate = df.loc[test_index]
    if not isinstance(train_candidate, pd.DataFrame) or not isinstance(test_candidate, pd.DataFrame):
        raise ValueError("Expected DataFrame outputs from stratified index selection.")

    train_df = train_candidate.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    test_df = test_candidate.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return train_df, test_df


def _engineer_colors(df: pd.DataFrame, added_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Engineer color difference features from photometric bands."""
    engineered = df.copy()
    for left, right, output_name in COLOR_FEATURES:
        if left in engineered.columns and right in engineered.columns:
            engineered[output_name] = engineered[left] - engineered[right]
            added_columns.append(output_name)
    return engineered, added_columns


def _engineer_band_statistics(df: pd.DataFrame, added_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Engineer band statistics from photometric bands."""
    engineered = df.copy()
    if all(column in engineered.columns for column in PHOTOMETRIC_BANDS):
        band_frame = engineered[PHOTOMETRIC_BANDS]
        engineered["band_mean"] = band_frame.mean(axis=1)
        engineered["band_std"] = band_frame.std(axis=1)
        engineered["band_min"] = band_frame.min(axis=1)
        engineered["band_max"] = band_frame.max(axis=1)
        engineered["band_range"] = engineered["band_max"] - engineered["band_min"]
        added_columns.extend(["band_mean", "band_std", "band_min", "band_max", "band_range"])
    return engineered, added_columns


def _engineer_interactions(df: pd.DataFrame, added_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Engineer redshift-color interaction features."""
    engineered = df.copy()
    for left, right, output_name in FEATURE_INTERACTIONS:
        if left in engineered.columns and right in engineered.columns:
            engineered[output_name] = engineered[left] * engineered[right]
            added_columns.append(output_name)
    return engineered, added_columns


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Engineer features: colors, band statistics, redshift interactions.

    Returns:
        Tuple of (engineered_df, list_of_added_column_names)
    """
    added_columns: list[str] = []
    engineered, added_columns = _engineer_colors(df, added_columns)
    engineered, added_columns = _engineer_band_statistics(engineered, added_columns)
    engineered, added_columns = _engineer_interactions(engineered, added_columns)
    return engineered, added_columns
