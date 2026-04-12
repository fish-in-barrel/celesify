from __future__ import annotations

from pathlib import Path

import streamlit as st

MODELS_DIR = Path("/workspace/outputs/models")


def run() -> None:
    st.set_page_config(page_title="celesify", layout="wide")
    st.title("celesify")
    st.caption("Phase 1 scaffold: Streamlit service is wired and running.")

    st.subheader("Model Artifacts Path")
    st.code(str(MODELS_DIR), language="text")

    if MODELS_DIR.exists():
        entries = sorted(p.name for p in MODELS_DIR.iterdir())
        st.success("Models directory is mounted.")
        st.write(entries if entries else "No artifacts yet.")
    else:
        st.warning("Models directory is not available yet. Run preprocessing and training first.")
