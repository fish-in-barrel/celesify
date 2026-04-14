from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from celesify.core.constants import (
    CLASS_ENCODING,
    KAGGLE_DATASET,
    KAGGLE_EXPECTED_FILE,
    NON_INFORMATIVE_COLUMNS,
    RANDOM_STATE,
    SKEW_CHECK_COLUMNS,
    SKEW_THRESHOLD,
    TARGET_COLUMN,
    TEST_SIZE,
)
from celesify.core.json_utils import write_json
from celesify.core.logging import log
from celesify.core.paths import resolve_preprocessing_paths


SERVICE = "preprocessing"

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


def validate_schema(df: pd.DataFrame) -> None:
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


def stratified_split(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def class_distribution(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts().sort_index()
    return {str(k): int(v) for k, v in counts.items()}


def class_proportions(series: pd.Series) -> dict[str, float]:
    proportions = series.value_counts(normalize=True).sort_index()
    return {str(k): float(v) for k, v in proportions.items()}


def _numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in SKEW_CHECK_COLUMNS if col in df.columns]


def _coerce_numeric_features(df: pd.DataFrame, numeric_columns: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
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
    normalized = series.astype("string").str.strip().str.upper()
    missing_mask = normalized.isna()
    unknown_mask = (~missing_mask) & (~normalized.isin(CLASS_ENCODING.keys()))
    unknown_labels = sorted(normalized.loc[unknown_mask].dropna().unique().tolist())
    return normalized, missing_mask, unknown_mask, unknown_labels


def _engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    engineered = df.copy()
    added_columns: list[str] = []

    for left, right, output_name in COLOR_FEATURES:
        if left in engineered.columns and right in engineered.columns:
            engineered[output_name] = engineered[left] - engineered[right]
            added_columns.append(output_name)

    if all(column in engineered.columns for column in PHOTOMETRIC_BANDS):
        band_frame = engineered[PHOTOMETRIC_BANDS]
        engineered["band_mean"] = band_frame.mean(axis=1)
        engineered["band_std"] = band_frame.std(axis=1)
        engineered["band_min"] = band_frame.min(axis=1)
        engineered["band_max"] = band_frame.max(axis=1)
        engineered["band_range"] = engineered["band_max"] - engineered["band_min"]
        added_columns.extend(["band_mean", "band_std", "band_min", "band_max", "band_range"])

    for left, right, output_name in FEATURE_INTERACTIONS:
        if left in engineered.columns and right in engineered.columns:
            engineered[output_name] = engineered[left] * engineered[right]
            added_columns.append(output_name)

    return engineered, added_columns


def has_kaggle_credentials() -> bool:
    return bool(os.getenv("KAGGLE_USERNAME")) and bool(os.getenv("KAGGLE_KEY"))


def download_from_kaggle(raw_dir: Path) -> list[Path]:
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


def run() -> None:
    raw_dir, processed_dir = resolve_preprocessing_paths()
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(raw_dir.glob("*.csv")) if raw_dir.exists() else []
    if not csv_files:
        csv_files = download_from_kaggle(raw_dir)

    selected_csv = csv_files[0]
    if len(csv_files) > 1:
        log(SERVICE, f"Found {len(csv_files)} CSV files; selecting {selected_csv.name} (alphabetical first).")
    else:
        log(SERVICE, f"Found 1 CSV file: {selected_csv.name}")

    df = pd.read_csv(selected_csv)
    log(SERVICE, f"Loaded {selected_csv.name} with shape {df.shape}")

    validate_schema(df)

    raw_target = df[TARGET_COLUMN].astype(str).str.strip().str.upper()
    log(SERVICE, f"Raw class distribution: {class_distribution(raw_target)}")

    dropped_columns = [col for col in NON_INFORMATIVE_COLUMNS if col in df.columns]
    df = df.drop(columns=dropped_columns)
    log(SERVICE, f"Dropped non-informative columns: {dropped_columns}")

    rows_before_cleaning = len(df)
    missing_by_col = {col: int(count) for col, count in df.isna().sum().items() if int(count) > 0}
    numeric_columns = _numeric_feature_columns(df)
    original_missing_row_mask = df[numeric_columns + [TARGET_COLUMN]].isna().any(axis=1) if numeric_columns else df[TARGET_COLUMN].isna()
    coerced_df, malformed_by_col = _coerce_numeric_features(df, numeric_columns)
    cleaned_target, missing_target_mask, unknown_target_mask, unknown_labels = _clean_target(coerced_df[TARGET_COLUMN])

    malformed_numeric_row_mask = (
        coerced_df[numeric_columns].isna().any(axis=1) & (~original_missing_row_mask)
        if numeric_columns
        else pd.Series(False, index=coerced_df.index)
    )
    rows_to_drop = original_missing_row_mask | malformed_numeric_row_mask | missing_target_mask | unknown_target_mask

    rows_removed_missing = int(original_missing_row_mask.sum())
    rows_removed_malformed = int((malformed_numeric_row_mask | unknown_target_mask).sum())
    rows_removed_total = int(rows_to_drop.sum())
    rows_after_cleaning = int(rows_before_cleaning - rows_removed_total)
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
    encoded_distribution = class_distribution(cleaned_df[TARGET_COLUMN])
    log(SERVICE, f"Encoded class distribution: {encoded_distribution}")

    available_skew_cols = _numeric_feature_columns(cleaned_df)
    skew_raw = cleaned_df[available_skew_cols].skew(numeric_only=True)
    if isinstance(skew_raw, pd.Series):
        skew_values = {
            col: float(val)
            for col, val in skew_raw.items()
            if pd.notna(val)
        }
    else:
        skew_values = {}
    high_skew_columns = {
        col: val
        for col, val in skew_values.items()
        if abs(val) > SKEW_THRESHOLD
    }
    log(
        SERVICE,
        "Skew check complete; no log transform applied. "
        f"Columns above threshold {SKEW_THRESHOLD}: {high_skew_columns}",
    )

    clean_train_df, clean_test_df = stratified_split(
        df=cleaned_df,
        target_column=TARGET_COLUMN,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    engineered_train_df, engineered_train_features = _engineer_features(clean_train_df)
    engineered_test_df, engineered_test_features = _engineer_features(clean_test_df)

    engineered_feature_columns = [
        str(col)
        for col in engineered_train_df.columns
        if col != TARGET_COLUMN
    ]
    clean_feature_columns = [str(col) for col in clean_train_df.columns if col != TARGET_COLUMN]

    clean_train_counts = class_distribution(clean_train_df[TARGET_COLUMN])
    clean_test_counts = class_distribution(clean_test_df[TARGET_COLUMN])
    clean_train_props = class_proportions(clean_train_df[TARGET_COLUMN])
    clean_test_props = class_proportions(clean_test_df[TARGET_COLUMN])

    engineered_train_counts = class_distribution(engineered_train_df[TARGET_COLUMN])
    engineered_test_counts = class_distribution(engineered_test_df[TARGET_COLUMN])
    engineered_train_props = class_proportions(engineered_train_df[TARGET_COLUMN])
    engineered_test_props = class_proportions(engineered_test_df[TARGET_COLUMN])

    log(SERVICE, f"Clean train split shape: {clean_train_df.shape}; class counts: {clean_train_counts}")
    log(SERVICE, f"Clean test split shape: {clean_test_df.shape}; class counts: {clean_test_counts}")
    log(
        SERVICE,
        "Feature engineering complete; "
        f"added {len(engineered_train_features)} engineered columns: {engineered_train_features}",
    )
    log(SERVICE, f"Engineered train split shape: {engineered_train_df.shape}; class counts: {engineered_train_counts}")
    log(SERVICE, f"Engineered test split shape: {engineered_test_df.shape}; class counts: {engineered_test_counts}")

    train_path = processed_dir / "train.parquet"
    test_path = processed_dir / "test.parquet"
    clean_train_path = processed_dir / "train_clean.parquet"
    clean_test_path = processed_dir / "test_clean.parquet"
    report_path = processed_dir / "preprocessing_report.json"

    clean_train_df.to_parquet(clean_train_path, index=False)
    clean_test_df.to_parquet(clean_test_path, index=False)
    engineered_train_df.to_parquet(train_path, index=False)
    engineered_test_df.to_parquet(test_path, index=False)

    qso_code = CLASS_ENCODING["QSO"]
    qso_share = float((cleaned_df[TARGET_COLUMN] == qso_code).mean())
    majority_ratio = 0.0
    if encoded_distribution:
        max_count = max(encoded_distribution.values())
        min_count = min(encoded_distribution.values())
        majority_ratio = float(max_count / min_count) if min_count else 0.0

    recommend_balanced = majority_ratio > 2.0
    report = {
        "selected_csv": selected_csv.name,
        "rows_initial": int(rows_before_cleaning),
        "rows_after_cleaning": int(rows_after_cleaning),
        "rows_after_missing_drop": int(rows_after_cleaning),
        "rows_removed_missing": int(rows_removed_missing),
        "rows_removed_malformed": int(rows_removed_malformed),
        "rows_removed_total": int(rows_removed_total),
        "rows_removed_missing_pct": removal_pct,
        "missing_value_strategy": "drop_rows_with_missing_values",
        "missing_values_by_column": missing_by_col,
        "malformed_values_by_column": malformed_by_col,
        "rows_removed_by_reason": {
            "missing": int(rows_removed_missing),
            "malformed": int(rows_removed_malformed),
            "total": int(rows_removed_total),
        },
        "dropped_columns": dropped_columns,
        "target_column": TARGET_COLUMN,
        "target_encoding": CLASS_ENCODING,
        "skew_check_columns": available_skew_cols,
        "skew_values": skew_values,
        "high_skew_columns": high_skew_columns,
        "transformations_applied": [],
        "clean_dataset": {
            "feature_columns": clean_feature_columns,
            "feature_count": int(len(clean_feature_columns)),
            "train_rows": int(len(clean_train_df)),
            "test_rows": int(len(clean_test_df)),
            "train_class_counts": clean_train_counts,
            "test_class_counts": clean_test_counts,
            "train_class_proportions": clean_train_props,
            "test_class_proportions": clean_test_props,
        },
        "feature_engineering": {
            "status": "completed",
            "engineered_columns": engineered_train_features,
            "engineered_feature_count": int(len(engineered_feature_columns)),
            "feature_count_before": int(len(clean_feature_columns)),
            "feature_count_after": int(len(engineered_feature_columns)),
            "feature_count_delta": int(len(engineered_feature_columns) - len(clean_feature_columns)),
            "feature_columns_before": clean_feature_columns,
            "feature_columns_after": engineered_feature_columns,
        },
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_rows": int(len(clean_train_df)),
            "test_rows": int(len(clean_test_df)),
            "train_class_counts": clean_train_counts,
            "test_class_counts": clean_test_counts,
            "train_class_proportions": clean_train_props,
            "test_class_proportions": clean_test_props,
            "engineered_train_rows": int(len(engineered_train_df)),
            "engineered_test_rows": int(len(engineered_test_df)),
        },
        "engineered_dataset": {
            "feature_columns": engineered_feature_columns,
            "feature_count": int(len(engineered_feature_columns)),
            "train_rows": int(len(engineered_train_df)),
            "test_rows": int(len(engineered_test_df)),
            "train_class_counts": engineered_train_counts,
            "test_class_counts": engineered_test_counts,
            "train_class_proportions": engineered_train_props,
            "test_class_proportions": engineered_test_props,
        },
        "dataset_comparison": {
            "rows_initial": int(rows_before_cleaning),
            "rows_after_cleaning": int(rows_after_cleaning),
            "feature_count_before": int(len(clean_feature_columns)),
            "feature_count_after": int(len(engineered_feature_columns)),
            "feature_count_delta": int(len(engineered_feature_columns) - len(clean_feature_columns)),
            "engineered_columns_added": engineered_train_features,
            "engineered_columns_removed": [],
        },
        "imbalance_assessment": {
            "qso_share": qso_share,
            "majority_to_minority_ratio": majority_ratio,
            "recommendation": (
                "use_class_weight_balanced"
                if recommend_balanced
                else "class_weight_not_required"
            ),
            "decision_basis": "majority_to_minority_ratio_gt_2.0",
        },
        "output_files": {
            "clean_train_parquet": clean_train_path.name,
            "clean_test_parquet": clean_test_path.name,
            "engineered_train_parquet": train_path.name,
            "engineered_test_parquet": test_path.name,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(report_path, report)

    log(SERVICE, f"Wrote {clean_train_path}")
    log(SERVICE, f"Wrote {clean_test_path}")
    log(SERVICE, f"Wrote {train_path}")
    log(SERVICE, f"Wrote {test_path}")
    log(SERVICE, f"Wrote {report_path}")
