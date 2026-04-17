"""Model evaluation and metrics computation for the training pipeline."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from celesify.core.constants import CLASS_LABEL_ORDER


def evaluate_model(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """
    Evaluate a trained model on test data.

    Args:
        model: Fitted scikit-learn classifier
        x_test: Test feature DataFrame
        y_test: Test target Series

    Returns:
        Dictionary with accuracy, f1_macro, confusion matrix, and per-class metrics
    """
    y_pred = model.predict(x_test)
    matrix = confusion_matrix(y_test, y_pred, labels=CLASS_LABEL_ORDER)
    report = classification_report(
        y_test,
        y_pred,
        labels=CLASS_LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    per_class_metrics = {}
    for label in CLASS_LABEL_ORDER:
        label_report = report.get(str(label), {}) if isinstance(report, dict) else {}
        if not isinstance(label_report, dict):
            continue
        per_class_metrics[str(label)] = {
            "precision": float(label_report.get("precision", 0.0)),
            "recall": float(label_report.get("recall", 0.0)),
            "f1_score": float(label_report.get("f1-score", 0.0)),
            "support": int(label_report.get("support", 0)),
        }

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "per_class_metrics": per_class_metrics,
    }
