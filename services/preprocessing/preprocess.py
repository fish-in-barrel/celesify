from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi


RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "class"
CLASS_ENCODING = {"STAR": 0, "GALAXY": 1, "QSO": 2}
NON_INFORMATIVE_COLUMNS = [
    "obj_ID",
    "run_ID",
    "rerun_ID",
    "cam_col",
    "field_ID",
    "spec_obj_ID",
    "plate",
    "MJD",
    "fiber_ID",
]
SKEW_CHECK_COLUMNS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
SKEW_THRESHOLD = 1.5
KAGGLE_DATASET = "fedesoriano/stellar-classification-dataset-sdss17"
KAGGLE_EXPECTED_FILE = "star_classification.csv"


def log(message: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] [preprocessing] {message}")


def resolve_paths() -> tuple[Path, Path]:
    workspace_raw = Path("/workspace/data/raw")
    workspace_processed = Path("/workspace/outputs/processed")

    repo_root = Path.cwd()
    local_raw = repo_root / "data" / "raw"
    local_processed = repo_root / "outputs" / "processed"

    if workspace_raw.exists() or workspace_processed.parent.exists():
        return workspace_raw, workspace_processed

    return local_raw, local_processed


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


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def has_kaggle_credentials() -> bool:
    return bool(os.getenv("KAGGLE_USERNAME")) and bool(os.getenv("KAGGLE_KEY"))


def download_from_kaggle(raw_dir: Path) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not has_kaggle_credentials():
        raise RuntimeError(
            "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY "
            "for the preprocessing container to enable auto-download."
        )

    log(f"No local CSV detected. Downloading dataset {KAGGLE_DATASET} via Kaggle API.")
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
        log(f"Kaggle download complete. Found expected file: {preferred_path.name}")
    else:
        log(
            "Kaggle download complete. Expected file name was not found; "
            f"available files: {[p.name for p in csv_files]}"
        )

    return csv_files


def main() -> None:
    raw_dir, processed_dir = resolve_paths()
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(raw_dir.glob("*.csv")) if raw_dir.exists() else []
    if not csv_files:
        csv_files = download_from_kaggle(raw_dir)

    selected_csv = csv_files[0]
    if len(csv_files) > 1:
        log(f"Found {len(csv_files)} CSV files; selecting {selected_csv.name} (alphabetical first).")
    else:
        log(f"Found 1 CSV file: {selected_csv.name}")

    df = pd.read_csv(selected_csv)
    log(f"Loaded {selected_csv.name} with shape {df.shape}")

    validate_schema(df)

    raw_target = df[TARGET_COLUMN].astype(str).str.strip().str.upper()
    log(f"Raw class distribution: {class_distribution(raw_target)}")

    dropped_columns = [col for col in NON_INFORMATIVE_COLUMNS if col in df.columns]
    df = df.drop(columns=dropped_columns)
    log(f"Dropped non-informative columns: {dropped_columns}")

    missing_by_col = {col: int(count) for col, count in df.isna().sum().items() if int(count) > 0}
    rows_before_dropna = len(df)
    df = df.dropna().copy()
    rows_after_dropna = len(df)
    rows_removed = rows_before_dropna - rows_after_dropna
    removal_pct = float((rows_removed / rows_before_dropna) * 100) if rows_before_dropna else 0.0
    if missing_by_col:
        log(f"Missing values found by column: {missing_by_col}")
    else:
        log("No missing values detected.")
    log(f"Missing-value strategy: drop rows -> removed {rows_removed} rows ({removal_pct:.4f}%).")

    cleaned_target = df[TARGET_COLUMN].astype(str).str.strip().str.upper()
    unknown_labels = sorted(set(cleaned_target.unique()) - set(CLASS_ENCODING.keys()))
    if unknown_labels:
        raise ValueError(f"Unknown labels in {TARGET_COLUMN}: {unknown_labels}")

    df[TARGET_COLUMN] = cleaned_target.map(CLASS_ENCODING).astype("int64")
    encoded_distribution = class_distribution(df[TARGET_COLUMN])
    log(f"Encoded class distribution: {encoded_distribution}")

    available_skew_cols = [c for c in SKEW_CHECK_COLUMNS if c in df.columns]
    skew_raw = df[available_skew_cols].skew(numeric_only=True)
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
        "Skew check complete; no log transform applied. "
        f"Columns above threshold {SKEW_THRESHOLD}: {high_skew_columns}"
    )

    train_df, test_df = stratified_split(
        df=df,
        target_column=TARGET_COLUMN,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_counts = class_distribution(train_df[TARGET_COLUMN])
    test_counts = class_distribution(test_df[TARGET_COLUMN])
    train_props = class_proportions(train_df[TARGET_COLUMN])
    test_props = class_proportions(test_df[TARGET_COLUMN])
    log(f"Train split shape: {train_df.shape}; class counts: {train_counts}")
    log(f"Test split shape: {test_df.shape}; class counts: {test_counts}")

    train_path = processed_dir / "train.parquet"
    test_path = processed_dir / "test.parquet"
    report_path = processed_dir / "preprocessing_report.json"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    qso_code = CLASS_ENCODING["QSO"]
    qso_share = float((df[TARGET_COLUMN] == qso_code).mean())
    majority_ratio = 0.0
    if encoded_distribution:
        max_count = max(encoded_distribution.values())
        min_count = min(encoded_distribution.values())
        majority_ratio = float(max_count / min_count) if min_count else 0.0

    recommend_balanced = majority_ratio > 2.0
    report = {
        "selected_csv": selected_csv.name,
        "rows_initial": int(rows_before_dropna),
        "rows_after_missing_drop": int(rows_after_dropna),
        "rows_removed_missing": int(rows_removed),
        "rows_removed_missing_pct": removal_pct,
        "missing_value_strategy": "drop_rows_with_missing_values",
        "missing_values_by_column": missing_by_col,
        "dropped_columns": dropped_columns,
        "target_column": TARGET_COLUMN,
        "target_encoding": CLASS_ENCODING,
        "skew_check_columns": available_skew_cols,
        "skew_values": skew_values,
        "high_skew_columns": high_skew_columns,
        "transformations_applied": [],
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_class_counts": train_counts,
            "test_class_counts": test_counts,
            "train_class_proportions": train_props,
            "test_class_proportions": test_props,
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
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(report_path, report)

    log(f"Wrote {train_path}")
    log(f"Wrote {test_path}")
    log(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
