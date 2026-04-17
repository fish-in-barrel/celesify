"""
Main preprocessing orchestrator.

Coordinates all phases:
1. Data loading (loading.py)
2. Data cleaning (cleaning.py)
3. Feature engineering & splitting (features.py)
4. Export & reporting (exports.py)
"""

from __future__ import annotations

from pathlib import Path

from celesify.core.constants import RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from celesify.core.logging import log
from celesify.core.paths import resolve_preprocessing_paths

from .cleaning import (
    assess_skew_and_log,
    class_distribution,
    class_proportions,
    drop_non_informative_columns,
    handle_missing_and_malformed_values,
    validate_schema,
)
from .exports import (
    build_preprocessing_report,
    export_parquet_files,
    save_report,
)
from .features import engineer_features, stratified_split
from .loading import load_raw_dataframe, select_csv_file


SERVICE = "preprocessing"




def run() -> None:
    """
    Main orchestration function: coordinates all preprocessing phases.

    Phases:
    1. Resolve paths and select CSV file
    2. Load raw data and validate schema
    3. Clean data (missing values, malformed data, target encoding)
    4. Assess and log feature skewness
    5. Perform stratified train/test split
    6. Engineer features (colors, statistics, interactions)
    7. Build and export report
    8. Export Parquet files
    """
    # ---- PHASE 1: Load data ----
    raw_dir, processed_dir = resolve_preprocessing_paths()
    processed_dir.mkdir(parents=True, exist_ok=True)

    selected_csv = select_csv_file(raw_dir)
    df = load_raw_dataframe(selected_csv)

    validate_schema(df)
    raw_target = df[TARGET_COLUMN].astype(str).str.strip().str.upper()
    log(SERVICE, f"Raw class distribution: {class_distribution(raw_target)}")

    # ---- PHASE 2: Clean data ----
    df, dropped_columns = drop_non_informative_columns(df)

    (
        cleaned_df,
        missing_by_col,
        malformed_by_col,
        rows_removed_missing,
        rows_removed_malformed,
        rows_removed_total,
        removal_pct,
    ) = handle_missing_and_malformed_values(df)

    rows_initial = len(df)
    rows_after_cleaning = len(cleaned_df)
    encoded_distribution = class_distribution(cleaned_df[TARGET_COLUMN])
    log(SERVICE, f"Encoded class distribution: {encoded_distribution}")

    # ---- PHASE 3: Assess skewness ----
    skew_values = assess_skew_and_log(cleaned_df)

    # ---- PHASE 4: Split data ----
    clean_train_df, clean_test_df = stratified_split(
        df=cleaned_df,
        target_column=TARGET_COLUMN,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # ---- PHASE 5: Engineer features ----
    engineered_train_df, engineered_train_features = engineer_features(clean_train_df)
    engineered_test_df, engineered_test_features = engineer_features(clean_test_df)

    engineered_feature_columns = [str(col) for col in engineered_train_df.columns if col != TARGET_COLUMN]
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

    # ---- PHASE 6: Build report metadata ----
    metadata = {
        "selected_csv": selected_csv.name,
        "rows_initial": rows_initial,
        "rows_after_cleaning": rows_after_cleaning,
        "rows_removed_missing": rows_removed_missing,
        "rows_removed_malformed": rows_removed_malformed,
        "rows_removed_total": rows_removed_total,
        "removal_pct": removal_pct,
        "missing_by_col": missing_by_col,
        "malformed_by_col": malformed_by_col,
        "dropped_columns": dropped_columns,
        "skew_values": skew_values,
        "encoded_distribution": encoded_distribution,
        "clean_feature_columns": clean_feature_columns,
        "clean_train_df": clean_train_df,
        "clean_test_df": clean_test_df,
        "clean_train_counts": clean_train_counts,
        "clean_test_counts": clean_test_counts,
        "clean_train_props": clean_train_props,
        "clean_test_props": clean_test_props,
        "engineered_feature_columns": engineered_feature_columns,
        "engineered_train_df": engineered_train_df,
        "engineered_test_df": engineered_test_df,
        "engineered_train_features": engineered_train_features,
        "engineered_train_counts": engineered_train_counts,
        "engineered_test_counts": engineered_test_counts,
        "engineered_train_props": engineered_train_props,
        "engineered_test_props": engineered_test_props,
        "clean_train_path": processed_dir / "train_clean.parquet",
        "clean_test_path": processed_dir / "test_clean.parquet",
        "engineered_train_path": processed_dir / "train.parquet",
        "engineered_test_path": processed_dir / "test.parquet",
    }

    report = build_preprocessing_report(metadata)

    # ---- PHASE 7: Export files ----
    report_path = processed_dir / "preprocessing_report.json"
    save_report(report, report_path)

    export_parquet_files(
        clean_train_df,
        clean_test_df,
        engineered_train_df,
        engineered_test_df,
        processed_dir,
    )
