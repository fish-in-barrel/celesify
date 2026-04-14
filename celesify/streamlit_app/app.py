from __future__ import annotations

import streamlit as st

from celesify.core.logging import log
from celesify.streamlit_app.common import (
    MODELS_DIR,
    PROCESSED_DIR,
    SERVICE,
    inverse_class_mapping,
    load_json,
    load_model,
    resolve_favicon,
    render_banner,
    render_startup_diagnostics,
)
from celesify.streamlit_app.page_data_explorer import render_data_explorer
from celesify.streamlit_app.page_performance_metrics import render_performance_metrics
from celesify.streamlit_app.page_upload_infer import render_upload_and_infer


def run() -> None:
    st.set_page_config(page_title="celesify", page_icon=resolve_favicon(), layout="wide")
    render_banner()

    baseline_metrics: dict = {}
    clean_tuned_metrics: dict = {}
    tuned_metrics: dict = {}
    best_params: dict = {}
    feature_importance: dict = {}
    preprocessing_report: dict = {}

    try:
        baseline_metrics = load_json(str(MODELS_DIR / "baseline_metrics.json"))
        clean_tuned_metrics = load_json(str(MODELS_DIR / "clean_tuned_metrics.json"))
        tuned_metrics = load_json(str(MODELS_DIR / "tuned_metrics.json"))
        best_params = load_json(str(MODELS_DIR / "best_params.json"))
        feature_importance = load_json(str(MODELS_DIR / "feature_importance.json"))
        preprocessing_report = load_json(str(PROCESSED_DIR / "preprocessing_report.json"))
    except FileNotFoundError as exc:
        st.warning(str(exc))

    class_mapping = tuned_metrics.get("class_mapping") if isinstance(tuned_metrics, dict) else None
    inverse_map = inverse_class_mapping(class_mapping if isinstance(class_mapping, dict) else None)

    model = None
    try:
        model = load_model(str(MODELS_DIR / "model.joblib"))
    except FileNotFoundError as exc:
        st.warning(str(exc))
    except Exception as exc:  # pragma: no cover - runtime safety path
        st.error(f"Failed to load model.joblib: {exc}")

    tab_explorer, tab_metrics, tab_infer = st.tabs(
        ["Data Exploration", "Model Evaluation", "Upload and Infer"]
    )

    with tab_explorer:
        render_data_explorer(inverse_map, preprocessing_report)

    with tab_metrics:
        render_performance_metrics(baseline_metrics, clean_tuned_metrics, tuned_metrics, best_params, feature_importance)

    with tab_infer:
        if model is None:
            st.info("Model artifact not loaded. Upload and inference are unavailable.")
        elif not tuned_metrics:
            st.info("Tuned metrics are missing, so feature schema could not be derived.")
        else:
            render_upload_and_infer(model, tuned_metrics)

    with st.expander("Utilities", expanded=False):
        if st.button("Refresh cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Caches cleared. Artifacts will be reloaded on next access.")
            log(SERVICE, "Streamlit caches cleared by user action.")

    with st.expander("Runtime Diagnostics", expanded=False):
        render_startup_diagnostics()
