"""Streamlit front end.

    uv run streamlit run app/streamlit_app.py

Streamlit Community Cloud installs requirements.txt but not this project as a
package, so the repo root and src/ are put on sys.path before anything from
the project is imported.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Freight Rate Prediction", page_icon=":truck:", layout="wide")

from app.common import backend_badge  # noqa: E402

VIEWS = ROOT / "app" / "views"
navigation = st.navigation(
    {
        "Project": [
            st.Page(str(VIEWS / "overview.py"), title="Overview", icon=":material/home:", default=True),
            st.Page(str(VIEWS / "eda.py"), title="Data exploration", icon=":material/query_stats:"),
            st.Page(str(VIEWS / "models.py"), title="Model comparison", icon=":material/leaderboard:"),
        ],
        "Pricing": [
            st.Page(str(VIEWS / "predict.py"), title="Price a load", icon=":material/local_shipping:"),
            st.Page(str(VIEWS / "batch.py"), title="Batch pricing", icon=":material/table_view:"),
            st.Page(str(VIEWS / "december.py"), title="December chart", icon=":material/calendar_month:"),
        ],
    }
)
backend_badge()
navigation.run()
