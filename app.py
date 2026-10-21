"""Unified Application Entry Point.

Compatible with:
1. Streamlit Community Cloud (runs `streamlit_app.py` directly).
2. Standalone Python: `python app.py` launches the Streamlit interface.
3. Flask execution: `python app.py --flask` or `python backend/app.py` launches Flask.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root and backend to python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "backend"))


def is_running_under_streamlit() -> bool:
    """Detect if the current execution is driven by Streamlit's script runner."""
    try:
        import streamlit as st
        # In Streamlit runtime, get_script_run_ctx() returns an active context object
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__" or is_running_under_streamlit():
    if "--flask" in sys.argv:
        from backend.app import app
        app.run(host="127.0.0.1", port=5000, debug=True)
    elif is_running_under_streamlit() or "streamlit" in sys.modules and any("streamlit" in a for a in sys.argv):
        import streamlit_app
        streamlit_app.main()
    else:
        # User executed `python app.py` directly from terminal:
        import subprocess
        print("[App Launcher] Launching Streamlit web application on http://localhost:8501 ...")
        cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT_DIR / "streamlit_app.py")]
        subprocess.run(cmd)
