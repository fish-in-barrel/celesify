from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from celesify.streamlit_app.common import inverse_class_mapping, safe_int


# Original dataset parameters (user inputs only)
ORIGINAL_FEATURES: dict[str, str] = {
    "alpha": "alpha (right ascension, degrees)",
    "delta": "delta (declination, degrees)",
    "u": "u (ultraviolet magnitude, mag)",
    "g": "g (green magnitude, mag)",
    "r": "r (red magnitude, mag)",
    "i": "i (infrared magnitude, mag)",
    "z": "z (far-infrared magnitude, mag)",
    "redshift": "redshift (unitless)",
}

# Engineered features (derived from original parameters)
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
PHOTOMETRIC_BANDS = ["u", "g", "r", "i", "z"]
FEATURE_INTERACTIONS = [
    ("redshift", "color_u_g", "redshift_color_u_g"),
    ("redshift", "color_g_r", "redshift_color_g_r"),
    ("redshift", "color_r_i", "redshift_color_r_i"),
    ("redshift", "color_i_z", "redshift_color_i_z"),
    ("redshift", "color_g_z", "redshift_color_g_z"),
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate engineered features from original dataset parameters.

    Adds color features, band statistics, and redshift-color interactions.

    Args:
        df: DataFrame containing original feature columns.

    Returns:
        DataFrame with both original and engineered features.
    """
    engineered = df.copy()

    # Create color features (differences between photometric bands)
    for left, right, output_name in COLOR_FEATURES:
        if left in engineered.columns and right in engineered.columns:
            engineered[output_name] = engineered[left] - engineered[right]

    # Create band statistics
    if all(band in engineered.columns for band in PHOTOMETRIC_BANDS):
        band_frame = engineered[PHOTOMETRIC_BANDS]
        engineered["band_mean"] = band_frame.mean(axis=1)
        engineered["band_std"] = band_frame.std(axis=1)
        engineered["band_min"] = band_frame.min(axis=1)
        engineered["band_max"] = band_frame.max(axis=1)
        engineered["band_range"] = engineered["band_max"] - engineered["band_min"]

    # Create redshift-color interactions
    for left, right, output_name in FEATURE_INTERACTIONS:
        if left in engineered.columns and right in engineered.columns:
            engineered[output_name] = engineered[left] * engineered[right]

    return engineered


def _infer_model_features(tuned_metrics: dict[str, Any]) -> list[str]:
    """Get the feature columns expected by the trained model."""
    columns = tuned_metrics.get("feature_columns", [])
    if isinstance(columns, list) and columns:
        return [str(col) for col in columns]
    # Fallback: all original and engineered features
    return (
        list(ORIGINAL_FEATURES.keys())
        + [col for _, _, col in COLOR_FEATURES]
        + ["band_mean", "band_std", "band_min", "band_max", "band_range"]
        + [col for _, _, col in FEATURE_INTERACTIONS]
    )



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
    st.caption(
        "Predict stellar class from original dataset parameters (alpha, delta, magnitudes, redshift). "
        "Engineered features are calculated automatically."
    )

    model_features = _infer_model_features(tuned_metrics)
    class_mapping = tuned_metrics.get("class_mapping")
    inverse_map = inverse_class_mapping(class_mapping if isinstance(class_mapping, dict) else None)

    mode = st.radio("Input mode", options=["Manual", "CSV Upload"], horizontal=True)

    if mode == "Manual":
        with st.form("manual_infer_form"):
            values: dict[str, float] = {}
            cols = st.columns(2)
            # Collect only original parameters from the user
            for idx, (feature, label) in enumerate(ORIGINAL_FEATURES.items()):
                values[feature] = cols[idx % 2].number_input(label, value=0.0, format="%.6f")
            submitted = st.form_submit_button("Predict")

        if submitted:
            # Create dataframe with original parameters
            input_df = pd.DataFrame([values], columns=list(ORIGINAL_FEATURES.keys()))
            # Engineer features automatically
            engineered_df = engineer_features(input_df)
            # Extract only the features the model expects
            model_input = engineered_df[model_features].copy()
            # Predict
            pred_df = _predict_dataframe(model, model_input, inverse_map)
            row = pred_df.iloc[0].to_dict()
            st.success(f"Predicted class: {row['predicted_class']}")
            prob_cols = [col for col in pred_df.columns if col.startswith("prob_")]
            if prob_cols:
                probs = pred_df[prob_cols].iloc[0].rename(lambda name: name.replace("prob_", ""))
                st.bar_chart(probs)
            st.dataframe(pred_df, use_container_width=True)
        return

    upload = st.file_uploader("Upload CSV with original feature columns", type=["csv"])
    if upload is None:
        st.info("Upload a CSV file with columns: " + ", ".join(ORIGINAL_FEATURES.keys()))
        return

    uploaded_df = pd.read_csv(upload)
    missing_columns = [col for col in ORIGINAL_FEATURES.keys() if col not in uploaded_df.columns]
    if missing_columns:
        st.error(f"Missing required original columns: {', '.join(missing_columns)}")
        return

    # Extract only original feature columns
    infer_df = uploaded_df[list(ORIGINAL_FEATURES.keys())].copy()
    for col in ORIGINAL_FEATURES.keys():
        infer_df[col] = pd.to_numeric(infer_df[col], errors="coerce")

    nan_rows = infer_df[infer_df.isna().any(axis=1)]
    if not nan_rows.empty:
        st.error("Some rows contain non-numeric or empty feature values after coercion.")
        st.dataframe(nan_rows.head(20), use_container_width=True)
        return

    # Engineer features automatically
    engineered_df = engineer_features(infer_df)
    # Extract only the features the model expects
    model_input = engineered_df[model_features].copy()
    # Predict
    pred_df = _predict_dataframe(model, model_input, inverse_map)
    st.success(f"Predictions generated for {len(pred_df)} row(s).")
    st.dataframe(pred_df, use_container_width=True)

