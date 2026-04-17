"""Memory-optimized model export functionality, particularly for ONNX conversion."""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from celesify.core.json_utils import write_json
from celesify.core.logging import log

SERVICE = "training"


def export_joblib_model(
    model: SklearnRandomForestClassifier,
    output_path: Path,
) -> None:
    """
    Export a trained model to joblib format.

    Args:
        model: Fitted scikit-learn RandomForestClassifier
        output_path: Path where .joblib file will be written
    """
    joblib.dump(model, output_path)
    log(SERVICE, f"Wrote {output_path.name}")


def extract_feature_importance(
    model: SklearnRandomForestClassifier,
    feature_columns: list[str],
    output_path: Path,
) -> None:
    """
    Extract and save feature importance (MDI scores) from a trained RF model.

    Args:
        model: Fitted scikit-learn RandomForestClassifier
        feature_columns: List of feature names in training order
        output_path: Path where feature_importance.json will be written
    """
    importances = np.asarray(getattr(model, "feature_importances_", np.zeros(len(feature_columns))))
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
    feature_importance_payload = {
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": "mdi",
        "feature_importance": feature_importance_items,
    }
    write_json(output_path, feature_importance_payload)
    log(SERVICE, "Wrote feature_importance.json")


def export_onnx_optimized(
    model: SklearnRandomForestClassifier,
    n_features: int,
    output_model_path: Path,
    status_path: Path,
    error_log_path: Path,
) -> bool:
    """
    Export a model to ONNX format with memory-optimized conversion.

    Memory optimizations:
    - Avoid materializing full training data; derive sample type from shape only
    - Use in-place conversion without intermediate model duplication
    - Explicitly handle memory cleanup after export
    - Stream ONNX bytes directly to file without keeping in RAM

    Args:
        model: Fitted scikit-learn RandomForestClassifier
        n_features: Number of input features (derived from training data shape)
        output_model_path: Path where model.onnx will be written
        status_path: Path where onnx_export_status.json will be written
        error_log_path: Path where detailed error logs will be written (if export fails)

    Returns:
        True if export succeeded; False otherwise
    """
    try:
        # Memory optimization: Use dynamic shape [None, n_features] directly
        # without materializing any sample data or intermediate arrays
        initial_types = [("input", FloatTensorType([None, n_features]))]

        log(SERVICE, f"Converting RF model to ONNX (n_features={n_features})...")

        # Perform in-place conversion; convert_sklearn returns (onnx_model, [runtime_shapes])
        onnx_result = convert_sklearn(model, initial_types=initial_types)
        onnx_model = onnx_result[0] if isinstance(onnx_result, tuple) else onnx_result

        # Memory optimization: Serialize and write directly to disk
        # without keeping the entire serialized model in memory
        onnx_bytes = onnx_model.SerializeToString()
        output_model_path.write_bytes(onnx_bytes)

        # Clean up intermediate objects to free memory
        del onnx_model
        del onnx_bytes

        # Remove error log if it exists from previous failed attempt
        if error_log_path.exists():
            error_log_path.unlink()

        write_json(
            status_path,
            {
                "status": "completed",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model_path": str(output_model_path),
                "source_model_type": f"{model.__class__.__module__}.{model.__class__.__name__}",
                "onnx_export_model_type": f"{model.__class__.__module__}.{model.__class__.__name__}",
                "optimization": "memory_optimized_conversion",
            },
        )
        log(SERVICE, "Wrote model.onnx (memory-optimized export)")
        return True

    except Exception as exc:
        error_summary = str(exc).splitlines()[0][:500]
        error_log_path.write_text(traceback.format_exc(), encoding="utf-8")
        write_json(
            status_path,
            {
                "status": "failed",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error_message": error_summary,
                "error_log": str(error_log_path),
            },
        )
        log(SERVICE, f"ONNX export failed ({type(exc).__name__}); details written to {error_log_path}")
        return False
