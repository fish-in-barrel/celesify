"""Report generation and artifact writing for training pipeline."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from celesify.core.constants import CLASS_LABEL_ORDER, RANDOM_STATE
from celesify.core.json_utils import as_jsonable, write_json
from celesify.core.logging import log

SERVICE = "training"


@dataclass
class PreprocessingSummary:
    """Metadata extracted from preprocessing report."""

    rows_removed_missing: int | None
    rows_removed_malformed: int | None
    rows_removed_total: int | None
    feature_count_before: int | None
    feature_count_after: int | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values for cleaner JSON."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class DatasetInfo:
    """Dataset split information and feature metadata."""

    train_file: str
    test_file: str
    variant: str
    train_shape: tuple[int, int]
    test_shape: tuple[int, int]
    feature_columns: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for JSON serialization."""
        return {
            "split_files": {
                "train": self.train_file,
                "test": self.test_file,
            },
            "dataset_variant": self.variant,
            "dataset_shapes": {
                "train": list(self.train_shape),
                "test": list(self.test_shape),
            },
            "feature_columns": self.feature_columns,
            "n_features": len(self.feature_columns),
        }


@dataclass
class SearchMetadata:
    """Hyperparameter search configuration and backend info."""

    n_iter: int
    cv_splits: int
    n_jobs: int
    scoring: str = "f1_macro"
    backend: str = "sklearn_cpu"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for JSON serialization."""
        return {
            "n_iter_used": self.n_iter,
            "cv_splits": self.cv_splits,
            "n_jobs": self.n_jobs,
            "scoring": self.scoring,
            "search_backend": self.backend,
        }


@dataclass
class SearchResults:
    """Results from hyperparameter search."""

    best_params: dict[str, Any]
    best_cv_score: float
    top_5_results: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for JSON serialization."""
        return {
            "best_params": self.best_params,
            "best_cv_score": self.best_cv_score,
            "top_5_results": self.top_5_results,
        }


def build_preprocessing_summary(preprocess_report: dict) -> PreprocessingSummary:
    """
    Extract preprocessing summary from preprocessing report.

    Args:
        preprocess_report: Dictionary loaded from preprocessing_report.json

    Returns:
        PreprocessingSummary dataclass with extracted metadata
    """
    return PreprocessingSummary(
        rows_removed_missing=preprocess_report.get("rows_removed_missing"),
        rows_removed_malformed=preprocess_report.get("rows_removed_malformed"),
        rows_removed_total=preprocess_report.get("rows_removed_total"),
        feature_count_before=(preprocess_report.get("dataset_comparison") or {}).get("feature_count_before"),
        feature_count_after=(preprocess_report.get("dataset_comparison") or {}).get("feature_count_after"),
    )


def build_baseline_metrics_report(
    eval_results: dict[str, Any],
    dataset_info: DatasetInfo,
    preprocessing_summary: PreprocessingSummary,
    class_mapping: dict[str, int],
) -> dict[str, Any]:
    """
    Build baseline model metrics report.

    Args:
        eval_results: Output from evaluate_model()
        dataset_info: DatasetInfo with split and feature metadata
        preprocessing_summary: PreprocessingSummary from preprocessing
        class_mapping: Class encoding dictionary

    Returns:
        Baseline metrics payload ready for JSON serialization
    """
    return {
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "class_label_order": CLASS_LABEL_ORDER,
        "class_mapping": class_mapping,
        "preprocessing_summary": preprocessing_summary.to_dict(),
        **dataset_info.to_dict(),
        **eval_results,
    }


def build_tuned_metrics_report(
    eval_results: dict[str, Any],
    dataset_info: DatasetInfo,
    preprocessing_summary: PreprocessingSummary,
    search_metadata: SearchMetadata,
    search_results: SearchResults,
    class_mapping: dict[str, int],
    feature_engineering_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build tuned model metrics report.

    Args:
        eval_results: Output from evaluate_model()
        dataset_info: DatasetInfo with split and feature metadata
        preprocessing_summary: PreprocessingSummary from preprocessing
        search_metadata: SearchMetadata with tuning configuration
        search_results: SearchResults from hyperparameter search
        class_mapping: Class encoding dictionary
        feature_engineering_payload: Optional feature engineering metadata

    Returns:
        Tuned metrics payload ready for JSON serialization
    """
    return {
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "class_label_order": CLASS_LABEL_ORDER,
        "class_mapping": class_mapping,
        "preprocessing_summary": preprocessing_summary.to_dict(),
        "feature_engineering": feature_engineering_payload or {},
        **dataset_info.to_dict(),
        **search_metadata.to_dict(),
        **search_results.to_dict(),
        **eval_results,
    }


def build_best_params_report(
    search_metadata: SearchMetadata,
    search_results: SearchResults,
) -> dict[str, Any]:
    """
    Build best hyperparameters report.

    Args:
        search_metadata: SearchMetadata with tuning configuration
        search_results: SearchResults from hyperparameter search

    Returns:
        Best params payload ready for JSON serialization
    """
    return {
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **search_metadata.to_dict(),
        **search_results.to_dict(),
    }


def build_top_trials_report(
    search_metadata: SearchMetadata,
    search_results: SearchResults,
) -> dict[str, Any]:
    """
    Build top trials report.

    Args:
        search_metadata: SearchMetadata with tuning configuration
        search_results: SearchResults from hyperparameter search

    Returns:
        Top trials payload ready for JSON serialization
    """
    return {
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scoring": search_metadata.scoring,
        "n_iter_used": search_metadata.n_iter,
        "top_5_results": search_results.top_5_results,
    }


def write_baseline_artifacts(
    output_dir: Path,
    baseline_metrics_report: dict[str, Any],
) -> None:
    """
    Write baseline model artifacts to disk.

    Args:
        output_dir: Directory where metrics will be saved
        baseline_metrics_report: Baseline metrics payload from build_baseline_metrics_report()
    """
    write_json(output_dir / "baseline_metrics.json", as_jsonable(baseline_metrics_report))
    log(SERVICE, "Wrote baseline_metrics.json")


def write_tuned_artifacts(
    output_dir: Path,
    prefix: str,
    tuned_metrics_report: dict[str, Any],
    best_params_report: dict[str, Any],
    top_trials_report: dict[str, Any],
) -> None:
    """
    Write tuned model artifacts to disk.

    Args:
        output_dir: Directory where artifacts will be saved
        prefix: Prefix for output filenames (e.g., "clean_tuned" or "" for primary model)
        tuned_metrics_report: Tuned metrics payload from build_tuned_metrics_report()
        best_params_report: Best params payload from build_best_params_report()
        top_trials_report: Top trials payload from build_top_trials_report()
    """
    metrics_file = f"{prefix}_metrics.json" if prefix else "tuned_metrics.json"
    best_params_file = f"best_params_{prefix}.json" if prefix else "best_params.json"
    top_trials_file = f"top_trials_{prefix}.json" if prefix else "top_trials.json"

    write_json(output_dir / metrics_file, as_jsonable(tuned_metrics_report))
    write_json(output_dir / best_params_file, as_jsonable(best_params_report))
    write_json(output_dir / top_trials_file, as_jsonable(top_trials_report))

    log(SERVICE, f"Wrote {metrics_file}")
    log(SERVICE, f"Wrote {best_params_file}")
    log(SERVICE, f"Wrote {top_trials_file}")


def write_placeholder_artifacts(output_dir: Path) -> None:
    """
    Write placeholder artifacts for missing data case.

    Args:
        output_dir: Directory where placeholders will be saved
    """
    placeholder = {
        "status": "skipped_no_processed_data",
        "required_files": [
            str(output_dir.parent / "processed" / "train_clean.parquet"),
            str(output_dir.parent / "processed" / "test_clean.parquet"),
            str(output_dir.parent / "processed" / "train.parquet"),
            str(output_dir.parent / "processed" / "test.parquet"),
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "baseline_metrics.json", placeholder)
    write_json(output_dir / "tuned_metrics.json", placeholder)
    write_json(output_dir / "best_params.json", {"status": "not_run"})
    write_json(output_dir / "feature_importance.json", {"status": "not_run"})
