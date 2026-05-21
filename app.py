"""HuggingFace Spaces entrypoint.

HF Spaces with the Streamlit SDK looks for `app.py` at the repo root.
This file re-execs the real Streamlit entrypoint at `app/frontend/streamlit_app.py`.
"""
import runpy
import sys

sys.argv = ["streamlit", "run", "app/frontend/streamlit_app.py"]
runpy.run_module("streamlit.web.cli", run_name="__main__")
