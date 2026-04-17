"""
Phase 4: Export & Reporting

Handles Parquet file export, report generation, and metadata organization.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from celesify.core.constants import (
    CLASS_ENCODING,
    RANDOM_STATE,
    SKEW_THRESHOLD,
    TARGET_COLUMN,
    TEST_SIZE,
)
from celesify.core.json_utils import write_json
from celesify.core.logging import log


SERVICE = "preprocessing"


def compute_imbalance_metrics(encoded_distribution: dict[str, int]) -> tuple[float, float, str]:
    """
    Compute class imbalance metrics and recommendation.

    Returns:
        Tuple of (qso_share, majority_ratio, recommendation)
    """
    qso_code = CLASS_ENCODING["QSO"]
    # Compute QSO share from distribution counts
    total = sum(encoded_distribution.values())
    qso_count = encoded_distribution.get(str(qso_code), 0)
    qso_share = float(qso_count / total) if total else 0.0

    majority_ratio = 0.0
    if encoded_distribution:
        max_count = max(encoded_distribution.values())
        min_count = min(encoded_distribution.values())
        majority_ratio = float(max_count / min_count) if min_count else 0.0

    recommend_balanced = majority_ratio > 2.0
    recommendation = "use_class_weight_balanced" if recommend_balanced else "class_weight_not_required"

    return qso_share, majority_ratio, recommendation


def build_preprocessing_report(
    metadata: dict,
) -> dict:
    """
    Build comprehensive preprocessing report with all metadata and decisions.

    Args:
        metadata: Dictionary with preprocessing metadata

    Returns:
        Complete report dictionary
    """
    qso_share, majority_ratio, recommendation = compute_imbalance_metrics(
        metadata["encoded_distribution"]
    )

    report = {
        "selected_csv": metadata["selected_csv"],
        "rows_initial": metadata["rows_initial"],
        "rows_after_cleaning": metadata["rows_after_cleaning"],
        "rows_removed_missing": metadata["rows_removed_missing"],
        "rows_removed_malformed": metadata["rows_removed_malformed"],
        "rows_removed_total": metadata["rows_removed_total"],
        "rows_removed_missing_pct": metadata["removal_pct"],
        "missing_value_strategy": "drop_rows_with_missing_values",
        "missing_values_by_column": metadata["missing_by_col"],
        "malformed_values_by_column": metadata["malformed_by_col"],
        "rows_removed_by_reason": {
            "missing": metadata["rows_removed_missing"],
            "malformed": metadata["rows_removed_malformed"],
            "total": metadata["rows_removed_total"],
        },
        "dropped_columns": metadata["dropped_columns"],
        "target_column": TARGET_COLUMN,
        "target_encoding": CLASS_ENCODING,
        "skew_check_columns": list(metadata["skew_values"].keys()),
        "skew_values": metadata["skew_values"],
        "high_skew_columns": {
            col: val for col, val in metadata["skew_values"].items() if abs(val) > SKEW_THRESHOLD
        },
        "transformations_applied": [],
        "clean_dataset": {
            "feature_columns": metadata["clean_feature_columns"],
            "feature_count": len(metadata["clean_feature_columns"]),
            "train_rows": len(metadata["clean_train_df"]),
            "test_rows": len(metadata["clean_test_df"]),
            "train_class_counts": metadata["clean_train_counts"],
            "test_class_counts": metadata["clean_test_counts"],
            "train_class_proportions": metadata["clean_train_props"],
            "test_class_proportions": metadata["clean_test_props"],
        },
        "feature_engineering": {
            "status": "completed",
            "engineered_columns": metadata["engineered_train_features"],
            "engineered_feature_count": len(metadata["engineered_feature_columns"]),
            "feature_count_before": len(metadata["clean_feature_columns"]),
            "feature_count_after": len(metadata["engineered_feature_columns"]),
            "feature_count_delta": len(metadata["engineered_feature_columns"])
            - len(metadata["clean_feature_columns"]),
            "feature_columns_before": metadata["clean_feature_columns"],
            "feature_columns_after": metadata["engineered_feature_columns"],
        },
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_rows": len(metadata["clean_train_df"]),
            "test_rows": len(metadata["clean_test_df"]),
            "train_class_counts": metadata["clean_train_counts"],
            "test_class_counts": metadata["clean_test_counts"],
            "train_class_proportions": metadata["clean_train_props"],
            "test_class_proportions": metadata["clean_test_props"],
            "engineered_train_rows": len(metadata["engineered_train_df"]),
            "engineered_test_rows": len(metadata["engineered_test_df"]),
        },
        "engineered_dataset": {
            "feature_columns": metadata["engineered_feature_columns"],
            "feature_count": len(metadata["engineered_feature_columns"]),
            "train_rows": len(metadata["engineered_train_df"]),
            "test_rows": len(metadata["engineered_test_df"]),
            "train_class_counts": metadata["engineered_train_counts"],
            "test_class_counts": metadata["engineered_test_counts"],
            "train_class_proportions": metadata["engineered_train_props"],
            "test_class_proportions": metadata["engineered_test_props"],
        },
        "dataset_comparison": {
            "rows_initial": metadata["rows_initial"],
            "rows_after_cleaning": metadata["rows_after_cleaning"],
            "feature_count_before": len(metadata["clean_feature_columns"]),
            "feature_count_after": len(metadata["engineered_feature_columns"]),
            "feature_count_delta": len(metadata["engineered_feature_columns"])
            - len(metadata["clean_feature_columns"]),
            "engineered_columns_added": metadata["engineered_train_features"],
            "engineered_columns_removed": [],
        },
        "imbalance_assessment": {
            "qso_share": qso_share,
            "majority_to_minority_ratio": majority_ratio,
            "recommendation": recommendation,
            "decision_basis": "majority_to_minority_ratio_gt_2.0",
        },
        "output_files": {
            "clean_train_parquet": metadata["clean_train_path"].name,
            "clean_test_parquet": metadata["clean_test_path"].name,
            "engineered_train_parquet": metadata["engineered_train_path"].name,
            "engineered_test_parquet": metadata["engineered_test_path"].name,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    return report


def export_parquet_files(
    clean_train_df: pd.DataFrame,
    clean_test_df: pd.DataFrame,
    engineered_train_df: pd.DataFrame,
    engineered_test_df: pd.DataFrame,
    processed_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    """
    Export train/test sets to Parquet format.

    Returns:
        Tuple of (clean_train_path, clean_test_path, engineered_train_path, engineered_test_path)
    """
    clean_train_path = processed_dir / "train_clean.parquet"
    clean_test_path = processed_dir / "test_clean.parquet"
    engineered_train_path = processed_dir / "train.parquet"
    engineered_test_path = processed_dir / "test.parquet"

    clean_train_df.to_parquet(clean_train_path, index=False)
    clean_test_df.to_parquet(clean_test_path, index=False)
    engineered_train_df.to_parquet(engineered_train_path, index=False)
    engineered_test_df.to_parquet(engineered_test_path, index=False)

    log(SERVICE, f"Wrote {clean_train_path}")
    log(SERVICE, f"Wrote {clean_test_path}")
    log(SERVICE, f"Wrote {engineered_train_path}")
    log(SERVICE, f"Wrote {engineered_test_path}")

    return clean_train_path, clean_test_path, engineered_train_path, engineered_test_path


def save_report(report: dict, report_path: Path) -> None:
    """Write preprocessing report to JSON file."""
    write_json(report_path, report)
    log(SERVICE, f"Wrote {report_path}")
