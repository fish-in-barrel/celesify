from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from celesify.streamlit_app.common import (
    inverse_class_mapping,
    safe_float,
    safe_int,
    validate_results_artifacts,
)


def _metric_delta_pp(tuned: float, baseline: float) -> float:
    return (tuned - baseline) * 100.0


def _extract_top_tuning_rows(
    best_params: dict[str, Any],
    tuned_metrics: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    candidate_keys = [
        "top_5_results",
        "top_results",
        "top_hyperparameter_tunings",
        "cv_results_top5",
    ]
    source_rows: list[dict[str, Any]] = []

    for key in candidate_keys:
        maybe_rows = best_params.get(key)
        if isinstance(maybe_rows, list) and maybe_rows:
            source_rows = [row for row in maybe_rows if isinstance(row, dict)]
            if source_rows:
                break

    normalized_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(source_rows, start=1):
        params = row.get("params") if isinstance(row.get("params"), dict) else row.get("best_params")
        params = params if isinstance(params, dict) else {}
        normalized_rows.append(
            {
                "rank": safe_int(row.get("rank", idx)),
                "cv_macro_f1": safe_float(row.get("mean_test_score", row.get("cv_macro_f1", 0.0))),
                "cv_f1_std": safe_float(row.get("std_test_score", row.get("cv_f1_std", 0.0))),
                "mean_fit_time_s": safe_float(row.get("mean_fit_time", row.get("mean_fit_time_s", 0.0))),
                "mean_score_time_s": safe_float(row.get("mean_score_time", row.get("mean_score_time_s", 0.0))),
                "n_estimators": params.get("n_estimators", row.get("n_estimators")),
                "max_depth": params.get("max_depth", row.get("max_depth")),
                "min_samples_split": params.get("min_samples_split", row.get("min_samples_split")),
                "max_features": params.get("max_features", row.get("max_features")),
                "class_weight": params.get("class_weight", row.get("class_weight")),
            }
        )

    if normalized_rows:
        top_df = pd.DataFrame(normalized_rows).sort_values(["rank", "cv_macro_f1"], ascending=[True, False]).head(5)
        return top_df, "saved-top5"

    best_only_raw = best_params.get("best_params")
    if isinstance(best_only_raw, dict):
        best_only = best_only_raw
    else:
        tuned_best = tuned_metrics.get("best_params")
        best_only = tuned_best if isinstance(tuned_best, dict) else {}
    top_df = pd.DataFrame(
        [
            {
                "rank": 1,
                "cv_macro_f1": safe_float(best_params.get("best_cv_score", 0.0)),
                "cv_f1_std": 0.0,
                "mean_fit_time_s": 0.0,
                "mean_score_time_s": 0.0,
                "n_estimators": best_only.get("n_estimators"),
                "max_depth": best_only.get("max_depth"),
                "min_samples_split": best_only.get("min_samples_split"),
                "max_features": best_only.get("max_features"),
                "class_weight": best_only.get("class_weight"),
            }
        ]
    )
    return top_df, "best-only"


def _compute_loss_and_error_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    matrix = metrics.get("confusion_matrix", [])
    if not isinstance(matrix, list) or not matrix:
        return {
            "error_rate": 0.0,
            "zero_one_loss": 0.0,
            "hamming_loss": 0.0,
            "mean_classification_error": 0.0,
            "balanced_error_rate": 0.0,
        }

    total = 0
    correct = 0
    for row_idx, row in enumerate(matrix):
        if not isinstance(row, list):
            continue
        total += sum(safe_int(v) for v in row)
        if row_idx < len(row):
            correct += safe_int(row[row_idx])

    if total <= 0:
        return {
            "error_rate": 0.0,
            "zero_one_loss": 0.0,
            "hamming_loss": 0.0,
            "mean_classification_error": 0.0,
            "balanced_error_rate": 0.0,
        }

    accuracy = correct / total
    error_rate = 1.0 - accuracy

    per_class = metrics.get("per_class_metrics", {})
    recalls: list[float] = []
    if isinstance(per_class, dict):
        for label_metrics in per_class.values():
            if isinstance(label_metrics, dict):
                recalls.append(safe_float(label_metrics.get("recall", 0.0)))

    balanced_accuracy = (sum(recalls) / len(recalls)) if recalls else accuracy

    return {
        "error_rate": error_rate,
        "zero_one_loss": error_rate,
        "hamming_loss": error_rate,
        "mean_classification_error": error_rate,
        "balanced_error_rate": 1.0 - balanced_accuracy,
    }


def _render_fit_diagnostics(
    tuned_f1: float,
    tuned_acc: float,
    best_params: dict[str, Any],
) -> None:
    st.markdown("#### Fit Diagnostics (Overfitting and Underfitting)")

    cv_macro_f1 = safe_float(best_params.get("best_cv_score", 0.0))
    gap = cv_macro_f1 - tuned_f1 if cv_macro_f1 > 0 else 0.0

    diag_cols = st.columns(3)
    diag_cols[0].metric("CV Macro F1", f"{cv_macro_f1:.4f}" if cv_macro_f1 > 0 else "N/A")
    diag_cols[1].metric("Test Macro F1", f"{tuned_f1:.4f}")
    diag_cols[2].metric("Generalization Gap", f"{gap:+.4f}" if cv_macro_f1 > 0 else "N/A")

    if cv_macro_f1 <= 0:
        st.info("Cross-validation score history is limited; detailed overfitting diagnosis is partially unavailable.")
        return

    if gap > 0.03:
        st.warning("Potential overfitting: CV performance is noticeably higher than held-out test performance.")
    elif tuned_f1 < 0.90 or tuned_acc < 0.90:
        st.warning("Potential underfitting: absolute performance remains low, suggesting model capacity/feature limits.")
    else:
        st.success("Fit check looks healthy: CV and test performance are close with strong absolute metrics.")


def render_performance_metrics(
    baseline_metrics: dict[str, Any],
    tuned_metrics: dict[str, Any],
    best_params: dict[str, Any],
    feature_importance: dict[str, Any],
) -> None:
    st.subheader("Model Evaluation")
    st.caption("Model outcomes reproduced from saved Training artifacts. The purpose of this view is to evaluate model performance and metrics to help with evaluating tuning and compare different approaches.")

    issues = validate_results_artifacts(baseline_metrics, tuned_metrics, feature_importance)
    for issue in issues:
        st.warning(issue)

    baseline_acc = safe_float(baseline_metrics.get("accuracy"))
    tuned_acc = safe_float(tuned_metrics.get("accuracy"))
    baseline_f1 = safe_float(baseline_metrics.get("f1_macro"))
    tuned_f1 = safe_float(tuned_metrics.get("f1_macro"))

    st.markdown("#### Overall Model Performance")
    st.caption("High-level quality: accuracy is overall correctness, while macro F1 balances precision/recall equally across classes.")
    card_cols = st.columns(4)
    card_cols[0].metric("Baseline Accuracy", f"{baseline_acc:.4f}", help="Fraction of all predictions that are correct using the baseline model.")
    card_cols[1].metric(
        "Tuned Accuracy",
        f"{tuned_acc:.4f}",
        delta=f"{_metric_delta_pp(tuned_acc, baseline_acc):+.2f} pp",
        help="Fraction of all predictions that are correct using the selected tuned (rank-1) model.",
    )
    card_cols[2].metric("Baseline Macro F1", f"{baseline_f1:.4f}", help="Average F1 over classes with equal class weight for the baseline model.")
    card_cols[3].metric(
        "Tuned Macro F1",
        f"{tuned_f1:.4f}",
        delta=f"{_metric_delta_pp(tuned_f1, baseline_f1):+.2f} pp",
        help="Average F1 over classes with equal class weight for the selected tuned (rank-1) model.",
    )

    class_mapping = tuned_metrics.get("class_mapping")
    inverse_map = inverse_class_mapping(class_mapping if isinstance(class_mapping, dict) else None)
    class_order = [safe_int(v) for v in tuned_metrics.get("class_label_order", list(inverse_map.keys()))]
    axis_labels = [inverse_map.get(idx, str(idx)) for idx in class_order]

    st.markdown("#### Class-Level Performance")
    st.caption("How the selected tuned (rank-1) model behaves per class, including error concentration and class-wise precision/recall tradeoffs.")

    confusion_matrix = tuned_metrics.get("confusion_matrix", [])
    if isinstance(confusion_matrix, list) and confusion_matrix:
        cm_df = pd.DataFrame(confusion_matrix, index=axis_labels, columns=axis_labels)
        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Reds",
            labels={"x": "Predicted", "y": "Actual", "color": "Count"},
            aspect="auto",
        )
        fig_cm.update_layout(title="Tuned Model Confusion Matrix", height=320, margin={"l": 36, "r": 18, "t": 46, "b": 36})
        st.plotly_chart(fig_cm, use_container_width=True, key="confusion_matrix")
        st.caption("Cells on the diagonal are correct predictions; larger off-diagonal values indicate where classes are confused.")

    rows: list[dict[str, Any]] = []
    baseline_per_class = baseline_metrics.get("per_class_metrics", {})
    tuned_per_class = tuned_metrics.get("per_class_metrics", {})
    for class_id in class_order:
        key = str(class_id)
        b_row = baseline_per_class.get(key, {}) if isinstance(baseline_per_class, dict) else {}
        t_row = tuned_per_class.get(key, {}) if isinstance(tuned_per_class, dict) else {}
        rows.append(
            {
                "class": inverse_map.get(class_id, key),
                "baseline_precision": safe_float(b_row.get("precision")),
                "baseline_recall": safe_float(b_row.get("recall")),
                "baseline_f1": safe_float(b_row.get("f1_score")),
                "tuned_precision": safe_float(t_row.get("precision")),
                "tuned_recall": safe_float(t_row.get("recall")),
                "tuned_f1": safe_float(t_row.get("f1_score")),
                "support": safe_int(t_row.get("support", b_row.get("support", 0))),
            }
        )

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption("Precision: purity of predicted class. Recall: coverage of true class. F1: precision/recall balance. Support: sample count.")

    st.markdown("#### Feature Signal")
    st.caption("Relative contribution of each feature in the tuned Random Forest based on mean decrease in impurity (MDI).")
    importance_items = feature_importance.get("feature_importance", [])
    if isinstance(importance_items, list) and importance_items:
        imp_df = pd.DataFrame(importance_items)
        if {"feature", "importance"}.issubset(imp_df.columns):
            fig_imp = px.bar(
                imp_df,
                x="importance",
                y="feature",
                orientation="h",
                color_discrete_sequence=["#e76f51"],
            )
            fig_imp.update_layout(title="Feature Importance (MDI)", height=320, margin={"l": 36, "r": 18, "t": 46, "b": 36})
            st.plotly_chart(fig_imp, use_container_width=True, key="feature_importance")
            st.caption("Higher importance means the feature contributed more often to useful class-separating splits.")

    st.markdown("#### Error and Reliability")
    st.caption("Error-focused view of the selected tuned (rank-1) model, including class-balance sensitivity and overfit/underfit checks.")

    st.markdown("##### Loss and Error Metrics")
    loss_metrics = _compute_loss_and_error_metrics(tuned_metrics)
    loss_cols = st.columns(2)
    loss_cols[0].metric(
        "Classification Error",
        f"{loss_metrics['error_rate']:.4f}",
        help="Equivalent here to error rate, zero-one loss, hamming loss, and mean classification error for single-label multiclass classification.",
    )
    loss_cols[1].metric(
        "Balanced Error",
        f"{loss_metrics['balanced_error_rate']:.4f}",
        help="One minus balanced accuracy; penalizes poor recall in minority classes.",
    )

    st.caption("Note: probabilistic losses (for example log loss) are unavailable in saved artifacts because class-probability outputs were not persisted.")

    _render_fit_diagnostics(tuned_f1=tuned_f1, tuned_acc=tuned_acc, best_params=best_params)

    st.markdown("#### Hyperparameter Search Comparison")
    st.caption("Top CV-ranked trial configurations from tuning. Useful differences often come from fit time, score stability, and score-time overhead, not just mean CV F1.")
    top_tuning_df, tuning_source = _extract_top_tuning_rows(best_params, tuned_metrics)
    if tuning_source == "best-only":
        st.info("Top-5 trial history is not present in saved artifacts; showing best available tuning snapshot.")
    if "mean_fit_time_s" in top_tuning_df.columns:
        top_tuning_df["f1_per_fit_second"] = top_tuning_df.apply(
            lambda row: (safe_float(row.get("cv_macro_f1")) / safe_float(row.get("mean_fit_time_s")))
            if safe_float(row.get("mean_fit_time_s")) > 0
            else 0.0,
            axis=1,
        )
    st.dataframe(top_tuning_df, use_container_width=True)

    if {"cv_macro_f1", "mean_fit_time_s"}.issubset(top_tuning_df.columns):
        tradeoff_df = top_tuning_df.copy()
        tradeoff_df["rank_label"] = tradeoff_df["rank"].astype(str)
        fig_tradeoff = px.scatter(
            tradeoff_df,
            x="mean_fit_time_s",
            y="cv_macro_f1",
            color="rank_label",
            size="n_estimators" if "n_estimators" in tradeoff_df.columns else None,
            hover_data=["cv_f1_std", "mean_score_time_s", "max_depth", "class_weight"],
            labels={
                "mean_fit_time_s": "Mean Fit Time (s)",
                "cv_macro_f1": "CV Macro F1",
                "rank_label": "Rank",
            },
            title="CV Macro F1 vs Mean Fit Time",
        )
        fig_tradeoff.update_layout(template="plotly_white", height=320, margin={"l": 36, "r": 18, "t": 46, "b": 36})
        st.plotly_chart(fig_tradeoff, use_container_width=True, key="top_tuning_tradeoff")

    if {"rank", "cv_f1_std"}.issubset(top_tuning_df.columns):
        stability_df = top_tuning_df.copy()
        stability_df["rank_label"] = stability_df["rank"].astype(str)
        fig_stability = px.bar(
            stability_df,
            x="rank_label",
            y="cv_f1_std",
            title="CV Stability by Hyperparameter Rank (Std of CV Macro F1)",
            labels={"rank_label": "Rank", "cv_f1_std": "CV Macro F1 Std"},
            color_discrete_sequence=["#f4a261"],
        )
        fig_stability.update_layout(template="plotly_white", height=300, margin={"l": 36, "r": 18, "t": 46, "b": 36})
        st.plotly_chart(fig_stability, use_container_width=True, key="top_tuning_stability")

    if "cv_macro_f1" in top_tuning_df.columns and "mean_fit_time_s" not in top_tuning_df.columns:
        cv_plot_df = top_tuning_df.copy()
        cv_plot_df["rank_label"] = cv_plot_df["rank"].astype(str)
        fig_top = px.bar(
            cv_plot_df,
            x="rank_label",
            y="cv_macro_f1",
            title="Cross-Validation Macro F1 by Hyperparameter Rank",
            labels={"rank_label": "Rank", "cv_macro_f1": "CV Macro F1"},
            color_discrete_sequence=["#4e79a7"],
        )
        fig_top.update_layout(template="plotly_white", height=300, margin={"l": 36, "r": 18, "t": 46, "b": 36})
        st.plotly_chart(fig_top, use_container_width=True, key="top_tuning_bar")

    with st.expander("Best Hyperparameters"):
        st.json(best_params)
