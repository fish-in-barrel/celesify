"""
Entry point for Streamlit Community Cloud deployment.

This script downloads prebuilt artifacts from GitHub releases and runs the celesify dashboard.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Set up artifact downloads before importing the app
import setup_artifacts
setup_artifacts.download_artifacts()
setup_artifacts.verify_artifacts()

# Add celesify to path so we can import it
sys.path.insert(0, str(Path(__file__).parent))

from celesify.streamlit_app.app import run

if __name__ == "__main__":
    run()
