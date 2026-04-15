"""
Model evaluation and metrics computation.

This module handles:
- Computing accuracy, F1, and per-class metrics
- Generating confusion matrices
- Extracting feature importances
- Formatting evaluation results for export
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from celesify.core.constants import CLASS_LABEL_ORDER
from celesify.core.logging import log

SERVICE = "training"


def evaluate_model(
    model: RandomForestClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """
    Compute all evaluation metrics for a trained model.

    Calculates:
    - Overall accuracy
    - Macro-averaged F1 (primary metric under class imbalance)
    - Confusion matrix
    - Per-class precision, recall, F1

    Args:
        model: Trained RandomForestClassifier.
        x_test: Test features.
        y_test: Test target labels.

    Returns:
        Dictionary containing all evaluation metrics and confusion matrix.
    """
    y_pred = model.predict(x_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABEL_ORDER)

    # Per-class metrics
    report = classification_report(
        y_test,
        y_pred,
        labels=CLASS_LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )

    per_class_metrics = _extract_per_class_metrics(report)

    # Overall metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    log(SERVICE, f"Evaluation: accuracy={accuracy:.4f}, f1_macro={f1_macro:.4f}")

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics,
    }


def _extract_per_class_metrics(report: dict) -> dict[str, dict[str, float]]:
    """
    Extract per-class metrics from sklearn classification_report.

    Args:
        report: Output from sklearn.metrics.classification_report with output_dict=True.

    Returns:
        Dictionary mapping class label (str) to metric dict.
    """
    per_class_metrics = {}

    for label in CLASS_LABEL_ORDER:
        label_str = str(label)
        label_report = report.get(label_str, {}) if isinstance(report, dict) else {}

        if not isinstance(label_report, dict):
            continue

        per_class_metrics[label_str] = {
            "precision": float(label_report.get("precision", 0.0)),
            "recall": float(label_report.get("recall", 0.0)),
            "f1_score": float(label_report.get("f1-score", 0.0)),
            "support": int(label_report.get("support", 0)),
        }

    return per_class_metrics


def extract_feature_importance(
    model: RandomForestClassifier,
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    """
    Extract Mean Decrease in Impurity (MDI) feature importance scores.

    Ranks features by importance and returns sorted list.

    Args:
        model: Trained RandomForestClassifier.
        feature_columns: List of feature column names (in order).

    Returns:
        List of dicts with 'feature' and 'importance' (descending).
    """
    importances = np.asarray(
        getattr(model, "feature_importances_", np.zeros(len(feature_columns)))
    )

    if len(importances) != len(feature_columns):
        log(
            SERVICE,
            f"Warning: imported {len(importances)} features, expected {len(feature_columns)}",
        )

    # Pair features with importances and sort descending
    feature_importance_items = sorted(
        (
            {
                "feature": feature,
                "importance": float(importance),
            }
            for feature, importance in zip(feature_columns, importances)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )

    log(SERVICE, f"Extracted feature importance for {len(feature_importance_items)} features")
    return feature_importance_items


def format_baseline_metrics(
    evaluation: dict[str, Any],
    class_label_order: list[int],
    class_mapping: dict[str, int],
    feature_columns: list[str],
    clean_train_shape: tuple[int, int],
    clean_test_shape: tuple[int, int],
    clean_variant: str,
    clean_train_name: str,
    clean_test_name: str,
    preprocess_report: dict[str, Any],
    random_state: int,
) -> dict[str, Any]:
    """
    Package baseline model metrics into exportable format.

    Args:
        evaluation: Output from evaluate_model().
        class_label_order: List of class label integers.
        class_mapping: Dict mapping class names to integers.
        feature_columns: List of feature column names.
        clean_train_shape: Shape tuple of training data.
        clean_test_shape: Shape tuple of test data.
        clean_variant: Name of dataset variant used.
        clean_train_name: Filename of training split.
        clean_test_name: Filename of test split.
        preprocess_report: Metadata from preprocessing stage.
        random_state: Random seed used.

    Returns:
        Dictionary ready for JSON export.
    """
    from datetime import datetime, timezone

    return {
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": random_state,
        "dataset_variant": clean_variant,
        "split_files": {
            "train": clean_train_name,
            "test": clean_test_name,
        },
        "class_label_order": class_label_order,
        "class_mapping": class_mapping,
        "feature_columns": feature_columns,
        "dataset_shapes": {
            "train": list(clean_train_shape),
            "test": list(clean_test_shape),
        },
        "n_features": len(feature_columns),
        "preprocessing_summary": _extract_preprocessing_summary(preprocess_report),
        **evaluation,
    }


def format_tuned_metrics(
    evaluation: dict[str, Any],
    best_params: dict[str, Any],
    best_cv_score: float,
    top_5_results: list[dict[str, Any]],
    class_label_order: list[int],
    class_mapping: dict[str, int],
    feature_columns: list[str],
    dataset_shapes: dict[str, list[int]],
    dataset_variant: str,
    split_files: dict[str, str],
    preprocess_report: dict[str, Any],
    n_iter_used: int,
    cv_splits: int,
    n_jobs: int,
    random_state: int,
) -> dict[str, Any]:
    """
    Package tuned model metrics into exportable format.

    Args:
        evaluation: Output from evaluate_model().
        best_params: Best hyperparameters found by RandomizedSearchCV.
        best_cv_score: Best CV score achieved during search.
        top_5_results: Top 5 trial results from search.
        class_label_order: List of class label integers.
        class_mapping: Dict mapping class names to integers.
        feature_columns: List of feature column names.
        dataset_shapes: Dict with 'train' and 'test' shape lists.
        dataset_variant: Name of dataset variant used.
        split_files: Dict with 'train' and 'test' filenames.
        preprocess_report: Metadata from preprocessing stage.
        n_iter_used: Number of iterations used in RandomizedSearchCV.
        cv_splits: Number of CV folds used.
        n_jobs: Number of parallel jobs used.
        random_state: Random seed used.

    Returns:
        Dictionary ready for JSON export.
    """
    from datetime import datetime, timezone

    return {
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": random_state,
        "dataset_variant": dataset_variant,
        "split_files": split_files,
        "class_label_order": class_label_order,
        "class_mapping": class_mapping,
        "feature_columns": feature_columns,
        "dataset_shapes": dataset_shapes,
        "preprocessing_summary": _extract_preprocessing_summary(preprocess_report),
        "n_iter_used": n_iter_used,
        "cv_splits": cv_splits,
        "n_jobs": n_jobs,
        "search_backend": "sklearn_cpu",
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "top_5_results": top_5_results,
        **evaluation,
    }


def format_best_params(
    best_params: dict[str, Any],
    best_cv_score: float,
    top_5_results: list[dict[str, Any]],
    n_iter_used: int,
) -> dict[str, Any]:
    """
    Package best hyperparameters into exportable format.

    Args:
        best_params: Best hyperparameters found.
        best_cv_score: Best CV score achieved.
        top_5_results: Top 5 trial results.
        n_iter_used: Number of iterations used.

    Returns:
        Dictionary ready for JSON export.
    """
    from datetime import datetime, timezone

    return {
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_iter_used": n_iter_used,
        "scoring": "f1_macro",
        "best_cv_score": best_cv_score,
        "search_backend": "sklearn_cpu",
        "best_params": best_params,
        "top_5_results": top_5_results,
    }


def format_top_trials(
    top_results: list[dict[str, Any]],
    n_iter_used: int,
) -> dict[str, Any]:
    """
    Package top trial results into exportable format.

    Args:
        top_results: List of top trial dicts.
        n_iter_used: Number of iterations used.

    Returns:
        Dictionary ready for JSON export.
    """
    from datetime import datetime, timezone

    return {
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scoring": "f1_macro",
        "n_iter_used": n_iter_used,
        "top_5_results": top_results,
    }


def format_feature_importance(
    importance_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Package feature importance into exportable format.

    Args:
        importance_items: List of dicts with 'feature' and 'importance'.

    Returns:
        Dictionary ready for JSON export.
    """
    from datetime import datetime, timezone

    return {
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": "mdi",
        "feature_importance": importance_items,
    }


def _extract_preprocessing_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Extract preprocessing metrics from report for inclusion in training artifacts."""
    return {
        "rows_removed_missing": report.get("rows_removed_missing"),
        "rows_removed_malformed": report.get("rows_removed_malformed"),
        "rows_removed_total": report.get("rows_removed_total"),
        "feature_count_before": (report.get("dataset_comparison") or {}).get("feature_count_before"),
        "feature_count_after": (report.get("dataset_comparison") or {}).get("feature_count_after"),
    }
