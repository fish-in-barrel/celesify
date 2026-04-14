from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from celesify.streamlit_app.common import inverse_class_mapping, safe_int


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


def _infer_feature_columns(tuned_metrics: dict[str, Any]) -> list[str]:
    columns = tuned_metrics.get("feature_columns", [])
    if isinstance(columns, list) and columns:
        return [str(col) for col in columns]
    return ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]


def _predict_dataframe(model: Any, data: pd.DataFrame, inverse_map: dict[int, str]) -> pd.DataFrame:
    preds = model.predict(data)
    pred_labels = [inverse_map.get(safe_int(v), str(v)) for v in preds]
    output = data.copy()
    output["predicted_class_id"] = [safe_int(v) for v in preds]
    output["predicted_class"] = pred_labels

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(data)
        classes = getattr(model, "classes_", list(inverse_map.keys()))
        for idx, class_id in enumerate(classes):
            label = inverse_map.get(safe_int(class_id), str(class_id))
            output[f"prob_{label}"] = proba[:, idx]
    return output


def render_upload_and_infer(model: Any, tuned_metrics: dict[str, Any]) -> None:
    st.subheader("Upload and Infer")
    st.caption("Predict stellar class from manual feature values or uploaded CSV.")

    feature_columns = _infer_feature_columns(tuned_metrics)
    class_mapping = tuned_metrics.get("class_mapping")
    inverse_map = inverse_class_mapping(class_mapping if isinstance(class_mapping, dict) else None)

    mode = st.radio("Input mode", options=["Manual", "CSV Upload"], horizontal=True)

    if mode == "Manual":
        with st.form("manual_infer_form"):
            values: dict[str, float] = {}
            cols = st.columns(2)
            for idx, feature in enumerate(feature_columns):
                label = FEATURE_LABELS.get(feature, feature)
                values[feature] = cols[idx % 2].number_input(label, value=0.0, format="%.6f")
            submitted = st.form_submit_button("Predict")

        if submitted:
            input_df = pd.DataFrame([values], columns=feature_columns)
            pred_df = _predict_dataframe(model, input_df, inverse_map)
            row = pred_df.iloc[0].to_dict()
            st.success(f"Predicted class: {row['predicted_class']}")
            prob_cols = [col for col in pred_df.columns if col.startswith("prob_")]
            if prob_cols:
                probs = pred_df[prob_cols].iloc[0].rename(lambda name: name.replace("prob_", ""))
                st.bar_chart(probs)
            st.dataframe(pred_df, use_container_width=True)
        return

    upload = st.file_uploader("Upload CSV with feature columns", type=["csv"])
    if upload is None:
        st.info("Upload a CSV file to run batch inference.")
        return

    uploaded_df = pd.read_csv(upload)
    missing_columns = [col for col in feature_columns if col not in uploaded_df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return

    infer_df = uploaded_df[feature_columns].copy()
    for col in feature_columns:
        infer_df[col] = pd.to_numeric(infer_df[col], errors="coerce")

    nan_rows = infer_df[infer_df.isna().any(axis=1)]
    if not nan_rows.empty:
        st.error("Some rows contain non-numeric or empty feature values after coercion.")
        st.dataframe(nan_rows.head(20), use_container_width=True)
        return

    pred_df = _predict_dataframe(model, infer_df, inverse_map)
    st.success(f"Predictions generated for {len(pred_df)} row(s).")
    st.dataframe(pred_df, use_container_width=True)
