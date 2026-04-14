from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from celesify.core.constants import CLASS_ENCODING

SERVICE = "streamlit"
FAVICON_PATH = Path(__file__).resolve().parent / "assets" / "favicon_galaxy.svg"
EXPECTED_MODEL_FILES = [
    "model.joblib",
    "baseline_metrics.json",
    "tuned_metrics.json",
    "best_params.json",
    "feature_importance.json",
]
PHOTOMETRIC_BANDS = ["u", "g", "r", "i", "z"]

MILKY_WAY_BANNER_URL = "https://upload.wikimedia.org/wikipedia/commons/0/00/Center_of_the_Milky_Way_Galaxy_IV_%E2%80%93_Composite.jpg"
MILKY_WAY_ATTRIBUTION = "Image: NASA/JPL-Caltech/ESA/CXC/STScI (Public Domain), via Wikimedia Commons"
GITHUB_REPO_URL = "https://github.com/fish-in-barrel/celesify"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_streamlit_paths() -> tuple[Path, Path]:
    local_outputs = _repo_root() / "outputs"
    workspace_outputs = Path("/workspace/outputs")
    cwd_outputs = Path.cwd().resolve() / "outputs"

    candidates = [workspace_outputs, local_outputs, cwd_outputs]
    for outputs_root in candidates:
        models_dir = outputs_root / "models"
        processed_dir = outputs_root / "processed"
        if models_dir.exists() or processed_dir.exists():
            return models_dir, processed_dir

    return local_outputs / "models", local_outputs / "processed"


MODELS_DIR, PROCESSED_DIR = resolve_streamlit_paths()


def resolve_favicon() -> str:
    return str(FAVICON_PATH) if FAVICON_PATH.exists() else "🌌"


def render_banner() -> None:
    st.markdown(
        f"""
        <style>
        [class*="StyledLinkIconContainer"],
        [data-testid*="stHeaderActionElements"],
        [class*="stHeaderActionElements"] {{
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            height: 0 !important;
            overflow: hidden !important;
        }}
        svg[aria-label="Link to heading"],
        a[href^="#"]:has(svg[aria-label="Link to heading"]) {{
            display: none !important;
        }}
        .celesify-banner {{
            position: relative;
            isolation: isolate;
            overflow: hidden;
            border-radius: 14px;
            min-height: 220px;
            background-color: #0b1424;
            display: flex;
            align-items: center;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
            box-shadow: 0 14px 26px rgba(0, 0, 0, 0.28);
        }}
        .celesify-banner-image {{
            position: absolute;
            inset: -1px;
            width: 100%;
            height: 100%;
            display: block;
            object-fit: cover;
            object-position: center 42%;
            transform: scale(1.02);
        }}
        .celesify-banner-overlay {{
            position: absolute;
            inset: -1px;
            background: linear-gradient(180deg, rgba(5, 9, 18, 0.18) 0%, rgba(5, 9, 18, 0.72) 100%);
            pointer-events: none;
        }}
        .celesify-banner-content {{
            position: relative;
            z-index: 1;
        }}
        .celesify-banner-title {{
            margin: 0;
            color: #f8fbff;
            font-size: clamp(2.4rem, 4vw, 3.2rem);
            line-height: 1.2;
            letter-spacing: 0.02em;
            font-weight: 700;
            user-select: none;
            text-decoration: none;
            cursor: pointer;
        }}
        .celesify-banner-title:visited,
        .celesify-banner-title:hover,
        .celesify-banner-title:active {{
            color: #f8fbff;
            text-decoration: none;
        }}
        .celesify-banner-subtitle {{
            margin: 0.2rem 0 0;
            color: #d7deea;
            font-size: 0.98rem;
            letter-spacing: 0.01em;
        }}
        .celesify-banner-attribution {{
            position: absolute;
            right: 0.9rem;
            bottom: 0.55rem;
            margin: 0;
            color: #aab7cf;
            font-size: 0.66rem;
            letter-spacing: 0.01em;
            z-index: 1;
        }}
        .celesify-banner-github {{
            position: absolute;
            top: 0.7rem;
            right: 0.7rem;
            width: 2rem;
            height: 2rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            color: #ecf2ff;
            background: rgba(8, 15, 30, 0.58);
            border: 1px solid rgba(203, 217, 246, 0.28);
            text-decoration: none;
            z-index: 2;
            backdrop-filter: blur(3px);
        }}
        .celesify-banner-github:hover,
        .celesify-banner-github:focus-visible {{
            color: #ffffff;
            background: rgba(8, 15, 30, 0.82);
            border-color: rgba(203, 217, 246, 0.5);
            text-decoration: none;
            outline: none;
        }}
        .celesify-banner-github svg {{
            width: 1.12rem;
            height: 1.12rem;
            fill: currentColor;
        }}
        </style>
        <section class="celesify-banner">
            <img class="celesify-banner-image" src="{MILKY_WAY_BANNER_URL}" alt="Milky Way">
            <div class="celesify-banner-overlay"></div>
            <a class="celesify-banner-github" href="{GITHUB_REPO_URL}" target="_blank" rel="noopener noreferrer" aria-label="Open GitHub repository" title="GitHub Repository">
                <svg viewBox="0 0 16 16" aria-hidden="true">
                    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.5-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.64 7.64 0 0 1 4 0c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
                </svg>
            </a>
            <div class="celesify-banner-content">
                <a class="celesify-banner-title" href="/" target="_self" onclick="window.location.assign('/'); return false;">celesify</a>
                <p class="celesify-banner-subtitle">Machine Learning stellar classification model dashboard: explorer, results, and inference</p>
            </div>
            <p class="celesify-banner-attribution">{MILKY_WAY_ATTRIBUTION}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def inverse_class_mapping(class_mapping: dict[str, Any] | None = None) -> dict[int, str]:
    mapping = class_mapping if class_mapping else CLASS_ENCODING
    inverse: dict[int, str] = {}
    for label, encoded in mapping.items():
        try:
            inverse[int(encoded)] = str(label)
        except (TypeError, ValueError):
            continue

    if not inverse:
        for label, encoded in CLASS_ENCODING.items():
            inverse[int(encoded)] = str(label)
    return inverse


@st.cache_data(show_spinner=False)
def load_json(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_parquet(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet file: {path}")
    return pd.read_parquet(path)


@st.cache_resource(show_spinner=False)
def load_model(path_str: str) -> Any:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    return joblib.load(path)


def validate_results_artifacts(
    baseline_metrics: dict[str, Any],
    tuned_metrics: dict[str, Any],
    feature_importance: dict[str, Any],
) -> list[str]:
    issues: list[str] = []

    if baseline_metrics.get("status") != "completed":
        issues.append("baseline_metrics.json status is not completed.")
    if tuned_metrics.get("status") != "completed":
        issues.append("tuned_metrics.json status is not completed.")
    if feature_importance.get("status") != "completed":
        issues.append("feature_importance.json status is not completed.")

    class_order = tuned_metrics.get("class_label_order", [])
    confusion = tuned_metrics.get("confusion_matrix", [])
    if not isinstance(class_order, list) or not class_order:
        issues.append("Missing class_label_order in tuned metrics.")
    if not isinstance(confusion, list) or len(confusion) != len(class_order):
        issues.append("Confusion matrix shape does not match class_label_order.")

    baseline_features = baseline_metrics.get("feature_columns", [])
    tuned_features = tuned_metrics.get("feature_columns", [])
    if baseline_features != tuned_features:
        issues.append("Feature column order differs between baseline and tuned metrics.")

    return issues


def render_startup_diagnostics() -> None:
    st.subheader("Runtime Diagnostics")
    st.caption("Checks artifact and data paths used by this dashboard.")
    st.write({"models_dir": str(MODELS_DIR), "processed_dir": str(PROCESSED_DIR)})

    missing = [name for name in EXPECTED_MODEL_FILES if not (MODELS_DIR / name).exists()]
    if missing:
        st.warning(f"Missing model artifacts: {', '.join(missing)}")
    else:
        st.success("Required model artifacts are present.")

    train_ok = (PROCESSED_DIR / "train.parquet").exists()
    test_ok = (PROCESSED_DIR / "test.parquet").exists()
    if train_ok or test_ok:
        st.info("At least one processed parquet file is available for Data Explorer.")
    else:
        st.warning("No processed parquet files found. Run preprocessing first.")


def render_plot_grid(
    plot_specs: list[tuple[str, Callable[[], go.Figure], str]],
    columns: int = 2,
    chart_height: int = 280,
) -> None:
    grid_cols = st.columns(columns)
    for idx, (title, figure_builder, key_prefix) in enumerate(plot_specs):
        with grid_cols[idx % columns]:
            with st.spinner(f"Loading {title}..."):
                fig = figure_builder()
                fig.update_layout(
                    template="plotly_white",
                    height=chart_height,
                    margin={"l": 36, "r": 18, "t": 46, "b": 36},
                    dragmode="zoom",
                )
                st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_{idx}")
