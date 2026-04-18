from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from celesify.preprocessing.features import engineer_features
from celesify.streamlit_app.common import MODELS_DIR, inverse_class_mapping, load_onnx_session, safe_int


ORIGINAL_FEATURE_COLUMNS: list[str] = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]

FEATURE_LABELS: dict[str, str] = {
    "alpha": "alpha (right ascension, degrees)",
    "delta": "delta (declination, degrees)",
    "u": "u (ultraviolet magnitude, mag)",
    "g": "g (green magnitude, mag)",
    "r": "r (red magnitude, mag)",
    "i": "i (infrared magnitude, mag)",
    "z": "z (far-infrared magnitude, mag)",
    "redshift": "redshift (unitless)",
}


def _feature_columns_from_metrics(metrics: dict[str, Any], fallback: list[str]) -> list[str]:
    columns = metrics.get("feature_columns", [])
    if isinstance(columns, list) and columns:
        return [str(column) for column in columns]
    return fallback


def _clean_raw_input(raw_input: pd.DataFrame) -> pd.DataFrame:
    cleaned = raw_input.copy()
    for column in ORIGINAL_FEATURE_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned


def _prepare_model_input(raw_input: pd.DataFrame, feature_columns: list[str], feature_mode: str) -> pd.DataFrame:
    cleaned = _clean_raw_input(raw_input)

    if cleaned.isna().any(axis=1).any():
        raise ValueError("Some rows contain non-numeric or missing values in the original dataset inputs.")

    if feature_mode == "engineered":
        engineered_df, _ = engineer_features(cleaned)
        return engineered_df[feature_columns].copy()

    return cleaned[feature_columns].copy()


def _class_ids_from_inverse_map(inverse_map: dict[int, str]) -> list[int]:
    class_ids = sorted({safe_int(class_id) for class_id in inverse_map.keys()})
    return class_ids or [0, 1, 2]


def _prediction_probabilities(probability_output: Any, inverse_map: dict[int, str]) -> pd.DataFrame:
    class_ids = _class_ids_from_inverse_map(inverse_map)
    column_names = [f"prob_{inverse_map.get(class_id, str(class_id))}" for class_id in class_ids]

    if probability_output is None:
        return pd.DataFrame(columns=column_names)

    rows: list[list[float]] = []
    if isinstance(probability_output, np.ndarray):
        probability_array = probability_output
        if probability_array.ndim == 1:
            probability_array = probability_array.reshape(1, -1)
        for row in probability_array:
            rows.append([float(value) for value in row.tolist()])
    elif isinstance(probability_output, list):
        for row in probability_output:
            if isinstance(row, dict):
                rows.append([float(row.get(class_id, row.get(str(class_id), 0.0))) for class_id in class_ids])
            elif isinstance(row, (list, tuple, np.ndarray)):
                rows.append([float(value) for value in list(row)])
    else:
        return pd.DataFrame(columns=column_names)

    probability_frame = pd.DataFrame(rows, columns=column_names)
    return probability_frame


def _predict_with_onnx(session: Any, input_frame: pd.DataFrame, inverse_map: dict[int, str]) -> pd.DataFrame:
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_frame.to_numpy(dtype=np.float32, copy=False)})

    predicted_ids = np.asarray(outputs[0]).reshape(-1)
    predicted_labels = [inverse_map.get(safe_int(value), str(value)) for value in predicted_ids]

    output_frame = input_frame.copy()
    output_frame["predicted_class_id"] = [safe_int(value) for value in predicted_ids]
    output_frame["predicted_class"] = predicted_labels

    if len(outputs) > 1:
        probability_frame = _prediction_probabilities(outputs[1], inverse_map)
        if not probability_frame.empty:
            output_frame = pd.concat([output_frame, probability_frame], axis=1)

    return output_frame


def _model_options(
    baseline_metrics: dict[str, Any],
    clean_tuned_metrics: dict[str, Any],
    tuned_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "label": "Baseline RF",
            "filename": "model_baseline.onnx",
            "feature_mode": "clean",
            "metrics": baseline_metrics,
        },
        {
            "label": "Clean-tuned RF",
            "filename": "model_clean_tuned.onnx",
            "feature_mode": "clean",
            "metrics": clean_tuned_metrics,
        },
        {
            "label": "Engineered-tuned RF",
            "filename": "model.onnx",
            "feature_mode": "engineered",
            "metrics": tuned_metrics,
        },
    ]


def _render_input_form(feature_columns: list[str]) -> pd.DataFrame | None:
    with st.form("manual_infer_form"):
        values: dict[str, float] = {}
        columns = st.columns(2)
        for index, feature in enumerate(feature_columns):
            label = FEATURE_LABELS.get(feature, feature)
            values[feature] = columns[index % 2].number_input(label, value=0.0, format="%.6f")
        submitted = st.form_submit_button("Predict")

    if not submitted:
        return None

    return pd.DataFrame([values], columns=feature_columns)


def _render_upload_frame(feature_columns: list[str]) -> pd.DataFrame | None:
    upload = st.file_uploader("Upload CSV with original dataset columns", type=["csv"])
    if upload is None:
        st.info("Upload a CSV file to run batch inference.")
        return None

    uploaded_df = pd.read_csv(upload)
    missing_columns = [column for column in feature_columns if column not in uploaded_df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return None

    return uploaded_df[feature_columns].copy()


def render_upload_and_infer(
    baseline_or_model: Any,
    clean_tuned_or_tuned_metrics: dict[str, Any] | None,
    tuned_metrics: dict[str, Any] | None = None,
) -> None:
    if tuned_metrics is None:
        # Legacy call path: render_upload_and_infer(model, tuned_metrics)
        baseline_metrics: dict[str, Any] = {}
        clean_tuned_metrics: dict[str, Any] = {}
        tuned_metrics_normalized = (
            clean_tuned_or_tuned_metrics if isinstance(clean_tuned_or_tuned_metrics, dict) else {}
        )
    else:
        baseline_metrics = baseline_or_model if isinstance(baseline_or_model, dict) else {}
        clean_tuned_metrics = (
            clean_tuned_or_tuned_metrics if isinstance(clean_tuned_or_tuned_metrics, dict) else {}
        )
        tuned_metrics_normalized = tuned_metrics if isinstance(tuned_metrics, dict) else {}

    st.subheader("Upload and Infer")
    st.caption("Provide only the original SDSS17 inputs. The selected ONNX model determines any hidden feature engineering before prediction.")
    st.markdown(
        """
        <style>
        div[data-testid="stSelectbox"] {
            max-width: 400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    options = _model_options(baseline_metrics, clean_tuned_metrics, tuned_metrics_normalized)
    selected_label = st.selectbox("Prediction model", [option["label"] for option in options], index=2)
    selected_option = next(option for option in options if option["label"] == selected_label)

    metrics = (
        selected_option["metrics"]
        if isinstance(selected_option.get("metrics"), dict)
        else tuned_metrics_normalized
    )
    feature_columns = _feature_columns_from_metrics(metrics, ORIGINAL_FEATURE_COLUMNS)
    class_mapping = metrics.get("class_mapping")
    inverse_map = inverse_class_mapping(class_mapping if isinstance(class_mapping, dict) else None)

    onnx_path = MODELS_DIR / selected_option["filename"]
    try:
        session = load_onnx_session(str(onnx_path))
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # pragma: no cover - runtime safety path
        st.error(f"Failed to load {onnx_path.name}: {exc}")
        return

    mode = st.radio("Input mode", options=["Manual", "CSV Upload"], horizontal=True)

    if mode == "Manual":
        raw_input = _render_input_form(ORIGINAL_FEATURE_COLUMNS)
        if raw_input is None:
            return

        try:
            model_input = _prepare_model_input(raw_input, feature_columns, selected_option["feature_mode"])
        except ValueError as exc:
            st.error(str(exc))
            return

        pred_df = _predict_with_onnx(session, model_input, inverse_map)
        row = pred_df.iloc[0].to_dict()
        st.success(f"Predicted class: {row['predicted_class']}")

        prob_cols = [column for column in pred_df.columns if column.startswith("prob_")]
        if prob_cols:
            probs = pred_df[prob_cols].iloc[0].rename(lambda name: name.replace("prob_", ""))
            st.bar_chart(probs)

        display_df = pd.concat([raw_input.reset_index(drop=True), pred_df.drop(columns=feature_columns, errors="ignore")], axis=1)
        st.dataframe(display_df, use_container_width=True)
        return

    raw_upload = _render_upload_frame(ORIGINAL_FEATURE_COLUMNS)
    if raw_upload is None:
        return

    try:
        model_input = _prepare_model_input(raw_upload, feature_columns, selected_option["feature_mode"])
    except ValueError as exc:
        st.error(str(exc))
        return

    pred_df = _predict_with_onnx(session, model_input, inverse_map)
    display_df = pd.concat([raw_upload.reset_index(drop=True), pred_df.drop(columns=feature_columns, errors="ignore")], axis=1)

    st.success(f"Predictions generated for {len(pred_df)} row(s).")
    st.dataframe(display_df, use_container_width=True)
