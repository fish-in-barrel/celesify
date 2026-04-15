from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from celesify.core.constants import TARGET_COLUMN
from celesify.streamlit_app.common import PHOTOMETRIC_BANDS, PROCESSED_DIR, load_parquet, render_plot_grid, safe_int


def _select_explorer_frame(dataset_variant: str) -> pd.DataFrame:
    clean_train_path = PROCESSED_DIR / "train_clean.parquet"
    clean_test_path = PROCESSED_DIR / "test_clean.parquet"
    train_path = PROCESSED_DIR / "train.parquet"
    test_path = PROCESSED_DIR / "test.parquet"

    if dataset_variant == "cleaned":
        if clean_train_path.exists():
            return load_parquet(str(clean_train_path))
        if clean_test_path.exists():
            return load_parquet(str(clean_test_path))
        if train_path.exists():
            return load_parquet(str(train_path))
        if test_path.exists():
            return load_parquet(str(test_path))
    else:
        if train_path.exists():
            return load_parquet(str(train_path))
        if test_path.exists():
            return load_parquet(str(test_path))
        if clean_train_path.exists():
            return load_parquet(str(clean_train_path))
        if clean_test_path.exists():
            return load_parquet(str(clean_test_path))

    raise FileNotFoundError("No parquet files found for Data Explorer.")


def render_data_explorer(inverse_map: dict[int, str], preprocess_report: dict | None = None) -> None:
    st.subheader("Data Exploration")
    st.caption("Class distribution, univariate/multivariate analysis, and correlations from processed parquet. The intent of this view is to provide a view of the data to help make decisions on refining the model and identifying opportunities for feature engineering.")

    comparison = preprocess_report.get("dataset_comparison", {}) if isinstance(preprocess_report, dict) and isinstance(preprocess_report.get("dataset_comparison", {}), dict) else {}
    rows_removed = preprocess_report.get("rows_removed_by_reason", {}) if isinstance(preprocess_report, dict) and isinstance(preprocess_report.get("rows_removed_by_reason", {}), dict) else {}
    clean_dataset = preprocess_report.get("clean_dataset", {}) if isinstance(preprocess_report, dict) and isinstance(preprocess_report.get("clean_dataset", {}), dict) else {}
    engineered_dataset = preprocess_report.get("engineered_dataset", {}) if isinstance(preprocess_report, dict) and isinstance(preprocess_report.get("engineered_dataset", {}), dict) else {}

    st.markdown("#### Preprocessing Comparison")
    st.caption("Cleaning is applied before feature engineering. Use the toggle to analyze either the cleaned baseline dataset or engineered dataset.")
    prep_cols = st.columns(4)
    prep_cols[0].metric("Rows removed", f"{safe_int(rows_removed.get('total', 0)):,}")
    prep_cols[1].metric("Dropped missing", f"{safe_int(rows_removed.get('missing', 0)):,}")
    prep_cols[2].metric("Dropped malformed", f"{safe_int(rows_removed.get('malformed', 0)):,}")
    prep_cols[3].metric("Feature delta", f"{safe_int(comparison.get('feature_count_delta', 0)):+d}")

    prep_table = pd.DataFrame(
        [
            {
                "stage": "cleaned baseline",
                "feature_count": safe_int(clean_dataset.get("feature_count", 0)),
                "train_rows": safe_int(clean_dataset.get("train_rows", 0)),
                "test_rows": safe_int(clean_dataset.get("test_rows", 0)),
            },
            {
                "stage": "engineered tuned",
                "feature_count": safe_int(engineered_dataset.get("feature_count", 0)),
                "train_rows": safe_int(engineered_dataset.get("train_rows", 0)),
                "test_rows": safe_int(engineered_dataset.get("test_rows", 0)),
            },
        ]
    )
    st.dataframe(prep_table, use_container_width=True)

    added_features = comparison.get("engineered_columns_added", []) if isinstance(comparison.get("engineered_columns_added", []), list) else []
    if added_features:
        with st.expander("Engineered features"):
            st.write(", ".join(str(feature) for feature in added_features))

    dataset_label = st.radio(
        "Dataset for analysis",
        options=["Cleaned baseline", "Engineered"],
        horizontal=True,
        key="explorer_dataset_variant",
    )
    dataset_variant = "cleaned" if dataset_label == "Cleaned baseline" else "engineered"

    try:
        df = _select_explorer_frame(dataset_variant)
    except FileNotFoundError as exc:
        st.warning(str(exc))
        return

    st.caption(f"Active dataset: {dataset_label}")

    if TARGET_COLUMN not in df.columns:
        st.error(f"Expected target column '{TARGET_COLUMN}' in parquet data.")
        return

    sample_size = st.slider(
        "Rows used for plotting",
        min_value=1000,
        max_value=min(len(df), 30000),
        value=min(len(df), 12000),
        step=1000,
        key="explorer_sample_size",
    )
    sampled = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df

    class_counts = sampled[TARGET_COLUMN].astype(int).value_counts().sort_index()
    class_labels = [inverse_map.get(int(idx), str(idx)) for idx in class_counts.index]
    class_df = pd.DataFrame({"Class": class_labels, "Count": class_counts.values})

    distribution_plot_specs: list[tuple[str, Callable[[], go.Figure], str]] = []

    st.markdown("#### Distributions")
    st.caption("These plots show how often each class and feature value appears. Look for class imbalance, skewed feature ranges, and unusual tails or outliers.")

    def class_distribution_figure() -> go.Figure:
        fig = px.bar(class_df, x="Class", y="Count", color_discrete_sequence=["#3e8e7e"])
        fig.update_layout(title="Class Distribution")
        return fig

    distribution_plot_specs.append(("Class Distribution", class_distribution_figure, "class_distribution"))

    feature_candidates = [col for col in sampled.columns if col != TARGET_COLUMN]
    
    # Set defaults based on dataset variant
    original_features = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
    if dataset_variant == "cleaned":
        # For cleaned data, show only original features
        default_features = [col for col in original_features if col in feature_candidates][:4]
    else:
        # For engineered data, show original features + engineered ones
        engineered_features = [
            "color_u_g", "color_g_r", "color_r_i", "color_i_z", "color_u_r",
            "color_g_i", "color_g_z", "color_r_z", "color_u_z",
            "band_mean", "band_std", "band_min", "band_max", "band_range"
        ]
        default_features = [col for col in original_features + engineered_features if col in feature_candidates][:6]
    
    selected_features = st.multiselect(
        "Histogram features",
        options=feature_candidates,
        default=default_features if default_features else feature_candidates[:4],
        key="explorer_hist_features",
    )

    if selected_features:
        for feature in selected_features:
            series = sampled[feature]

            def feature_histogram_figure(
                feature_name: str = feature,
                feature_series: pd.Series = series,
            ) -> go.Figure:
                fig = px.histogram(
                    x=feature_series,
                    nbins=40,
                    color_discrete_sequence=["#2a9d8f"],
                    labels={"x": feature_name, "y": "count"},
                )
                fig.update_layout(title=f"Distribution: {feature_name}")
                return fig

            distribution_plot_specs.append((f"Distribution: {feature}", feature_histogram_figure, f"hist_{feature}"))

    if distribution_plot_specs:
        render_plot_grid(distribution_plot_specs, columns=3, chart_height=245)

    st.markdown("#### Correlation")
    st.caption("This heatmap shows which photometric features move together. Strong positive or negative correlations can indicate redundant information or useful feature relationships.")
    include_redshift = st.checkbox("Include redshift in correlation heatmap", value=False, key="explorer_include_redshift")
    corr_features = [col for col in PHOTOMETRIC_BANDS if col in sampled.columns]
    if include_redshift and "redshift" in sampled.columns:
        corr_features = corr_features + ["redshift"]

    if len(corr_features) >= 2:
        corr = sampled[corr_features].corr(numeric_only=True)
        corr_display = corr.rename(index={"redshift": "rs"}, columns={"redshift": "rs"})
        annot_font_size = 7 if include_redshift else 9

        def correlation_figure() -> go.Figure:
            fig = px.imshow(
                corr_display,
                text_auto=True,
                color_continuous_scale="Viridis",
                aspect="equal",
            )
            fig.update_traces(textfont={"size": annot_font_size})
            fig.update_layout(title="Correlation Heatmap", width=500)
            return fig

        left, center, right = st.columns([1, 2, 1])
        with center:
            with st.spinner("Loading Correlation Heatmap..."):
                corr_fig = correlation_figure()
                corr_fig.update_layout(
                    template="plotly_white",
                    height=320,
                    margin={"l": 36, "r": 18, "t": 46, "b": 36},
                )
                st.plotly_chart(corr_fig, use_container_width=False, key="correlation_plot")
    else:
        st.info("Not enough numeric photometric features available for correlation heatmap.")

    numeric_features = [
        col
        for col in sampled.columns
        if col != TARGET_COLUMN and pd.api.types.is_numeric_dtype(sampled[col])
    ]

    st.markdown("#### Univariate Exploration")
    st.caption("These boxplots compare the spread of each feature across classes. They help reveal which variables separate classes well and where class distributions overlap.")
    
    # Use context-aware defaults for univariate features
    original_features_list = ["u", "g", "r", "i", "z", "redshift"]
    if dataset_variant == "cleaned":
        uni_defaults = [col for col in original_features_list if col in numeric_features][:4]
    else:
        engineered_features_list = ["color_u_g", "color_g_r", "color_r_i", "color_i_z", "band_mean", "band_std"]
        uni_defaults = [col for col in original_features_list + engineered_features_list if col in numeric_features][:4]
    
    univariate_features = st.multiselect(
        "Features for univariate analysis",
        options=numeric_features,
        default=uni_defaults if uni_defaults else numeric_features[:4],
        key="univariate_features",
    )

    if univariate_features:
        boxplot_specs: list[tuple[str, Callable[[], go.Figure], str]] = []

        for feature in univariate_features:
            series = sampled[feature].dropna()

            def single_boxplot(
                feature_name: str = feature,
                feature_frame: pd.DataFrame = sampled[[feature, TARGET_COLUMN]].dropna(),
            ) -> go.Figure:
                frame = feature_frame.copy()
                frame["class_label"] = frame[TARGET_COLUMN].astype(int).map(
                    lambda idx: inverse_map.get(safe_int(idx), str(idx))
                )
                fig = px.box(
                    frame,
                    x="class_label",
                    y=feature_name,
                    color="class_label",
                    points="outliers",
                    labels={"class_label": "Class", feature_name: feature_name},
                    title=f"Boxplot by Class: {feature_name}",
                )
                fig.update_layout(showlegend=False)
                return fig

            boxplot_specs.append((f"Boxplot: {feature}", single_boxplot, f"uni_box_{feature}"))

        st.markdown("##### Univariate Boxplots")
        render_plot_grid(boxplot_specs, columns=3, chart_height=260)
    else:
        st.info("Select at least one feature for univariate exploration.")

    st.markdown("#### Multivariate Analysis")
    st.caption("The pair matrix shows feature relationships two at a time. Off-diagonal scatter plots highlight class separation and interactions, while diagonal density curves show each feature’s distribution by class.")
    
    # Use context-aware defaults for pair plot features
    if dataset_variant == "cleaned":
        pair_defaults = [col for col in ["u", "g", "r", "i"] if col in numeric_features]
    else:
        pair_defaults = [col for col in ["u", "g", "color_u_g", "redshift"] if col in numeric_features]
    
    pair_features = st.multiselect(
        "Features for pair plot",
        options=numeric_features,
        default=pair_defaults if pair_defaults else numeric_features[:3],
        key="pairplot_features",
    )

    if len(pair_features) >= 2:
        max_pair_rows = min(len(sampled), 5000)
        pair_rows = st.slider(
            "Rows used for pair plot",
            min_value=500,
            max_value=max_pair_rows,
            value=min(1500, max_pair_rows),
            step=250,
            key="pairplot_rows",
        )
        pair_df = sampled[pair_features + [TARGET_COLUMN]].dropna().copy()
        if len(pair_df) > pair_rows:
            pair_df = pair_df.sample(n=pair_rows, random_state=42)
        pair_df["class_label"] = pair_df[TARGET_COLUMN].astype(int).map(
            lambda idx: inverse_map.get(safe_int(idx), str(idx))
        )

        class_labels = sorted(pair_df["class_label"].dropna().unique().tolist())
        palette = px.colors.qualitative.Set2 + px.colors.qualitative.Plotly
        class_colors = {label: palette[idx % len(palette)] for idx, label in enumerate(class_labels)}

        def _kde_curve(values: np.ndarray, points: int = 90) -> tuple[np.ndarray, np.ndarray] | None:
            if len(values) < 2:
                return None
            std = float(np.std(values, ddof=1))
            if std <= 1e-12:
                return None
            bandwidth = 1.06 * std * (len(values) ** (-1 / 5))
            if bandwidth <= 1e-12:
                return None
            xs = np.linspace(float(values.min()), float(values.max()), points)
            diffs = (xs[:, None] - values[None, :]) / bandwidth
            ys = np.exp(-0.5 * diffs**2).sum(axis=1) / (len(values) * bandwidth * np.sqrt(2.0 * np.pi))
            return xs, ys

        with st.spinner("Loading Pair Plot Matrix..."):
            n_features = len(pair_features)
            fig_pair = make_subplots(
                rows=n_features,
                cols=n_features,
                shared_xaxes=True,
                shared_yaxes=False,
                horizontal_spacing=0.02,
                vertical_spacing=0.02,
            )

            for row_idx, row_feature in enumerate(pair_features, start=1):
                for col_idx, col_feature in enumerate(pair_features, start=1):
                    if row_idx == col_idx:
                        for class_label in class_labels:
                            class_series = pd.Series(
                                pd.to_numeric(
                                    pair_df.loc[pair_df["class_label"] == class_label, row_feature],
                                    errors="coerce",
                                )
                            )
                            class_vals = class_series.dropna().to_numpy()
                            kde_vals = _kde_curve(class_vals)
                            if kde_vals is None:
                                continue
                            xs, ys = kde_vals
                            fig_pair.add_trace(
                                go.Scatter(
                                    x=xs,
                                    y=ys,
                                    mode="lines",
                                    line={"width": 1.6, "color": class_colors[class_label]},
                                    name=class_label,
                                    legendgroup=class_label,
                                    showlegend=(row_idx == 1 and col_idx == 1),
                                    hovertemplate=(
                                        f"feature={row_feature}<br>class={class_label}<br>"
                                        "value=%{x:.4f}<br>density=%{y:.4f}<extra></extra>"
                                    ),
                                ),
                                row=row_idx,
                                col=col_idx,
                            )
                    else:
                        for class_label in class_labels:
                            class_frame = pair_df[pair_df["class_label"] == class_label]
                            fig_pair.add_trace(
                                go.Scattergl(
                                    x=class_frame[col_feature],
                                    y=class_frame[row_feature],
                                    mode="markers",
                                    marker={
                                        "size": 4,
                                        "opacity": 0.42,
                                        "color": class_colors[class_label],
                                    },
                                    name=class_label,
                                    legendgroup=class_label,
                                    showlegend=False,
                                    hovertemplate=(
                                        f"class={class_label}<br>{col_feature}=%{{x:.4f}}<br>"
                                        f"{row_feature}=%{{y:.4f}}<extra></extra>"
                                    ),
                                ),
                                row=row_idx,
                                col=col_idx,
                            )

            for col_idx, feature in enumerate(pair_features, start=1):
                fig_pair.update_xaxes(title_text=feature, row=n_features, col=col_idx)
            for row_idx, feature in enumerate(pair_features, start=1):
                fig_pair.update_yaxes(title_text=feature, row=row_idx, col=1)

            fig_pair.update_layout(
                title="Pair Plot Matrix (Density on Diagonal)",
                template="plotly_white",
                height=max(560, 170 * len(pair_features)),
                margin={"l": 36, "r": 18, "t": 46, "b": 36},
                legend_title_text="Class",
            )
            st.plotly_chart(fig_pair, use_container_width=True, key="pair_plot_matrix")
    else:
        st.info("Select at least two features to render the pair plot matrix.")
