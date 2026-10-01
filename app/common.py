"""Shared helpers for the Streamlit pages: loading artefacts and the pricing client.

The app runs in two modes:

* **Remote** - `FREIGHT_API_URL` is set (docker compose): every price comes
  from the FastAPI service, so the UI exercises the real production API.
* **Embedded** - no URL (Streamlit Community Cloud, which cannot host a second
  process): the same `FreightRatePredictor` the API uses runs in-process.

Either way the pages call `price_loads` and never know the difference.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from freight_rate import config
from freight_rate.inference import FreightRatePredictor

# Categorical slots 1-3 of the validated reference palette (CVD-safe as a set).
EQUIPMENT_COLORS = {"Dry Van": "#2a78d6", "Reefer": "#eb6834", "Flatbed": "#1baf7a"}
PRIMARY = "#2a78d6"
NEUTRAL = "#8a8984"
CRITICAL = "#e34948"


def api_url() -> str | None:
    url = os.getenv("FREIGHT_API_URL")
    if not url:
        try:
            url = st.secrets.get("FREIGHT_API_URL")  # type: ignore[attr-defined]
        except Exception:  # no secrets file is the normal case
            url = None
    return url.rstrip("/") if url else None


@st.cache_resource(show_spinner="Loading model ...")
def load_predictor() -> FreightRatePredictor:
    return FreightRatePredictor.load()


@st.cache_data
def load_model_card() -> dict:
    path = config.MODEL_CARD_PATH
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


@st.cache_data
def load_eda_summary() -> dict:
    path = config.EDA_SUMMARY_PATH
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


@st.cache_data
def load_experiments() -> pd.DataFrame:
    path = config.EXPERIMENTS_PATH
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def figure_path(name: str) -> Path:
    return config.FIGURES_DIR / name


def _api_down(exc: Exception) -> None:
    # Same model, same code path: degrade to in-process pricing rather than
    # showing the user an error because the API container is restarting.
    st.toast(f"API unreachable ({type(exc).__name__}); pricing with the embedded model.")


def price_loads(loads: list[dict]) -> list[float]:
    """Price loads through the API when configured, otherwise in-process."""
    url = api_url()
    if url:
        clean = [{k: v for k, v in load.items() if v is not None and not (isinstance(v, float) and pd.isna(v))}
                 for load in loads]
        try:
            response = requests.post(f"{url}/api/predict/batch", json={"loads": clean}, timeout=30)
        except requests.RequestException as exc:
            _api_down(exc)
        else:
            if response.status_code == 422:
                raise ValueError(response.json().get("detail"))
            response.raise_for_status()
            return response.json()["predictions"]
    return [round(float(r), 2) for r in load_predictor().predict_frame(pd.DataFrame(loads))]


def price_one(load: dict) -> dict:
    url = api_url()
    if url:
        clean = {k: v for k, v in load.items() if v is not None}
        try:
            response = requests.post(f"{url}/api/predict", json=clean, timeout=15)
        except requests.RequestException as exc:
            _api_down(exc)
        else:
            if response.status_code == 422:
                raise ValueError(response.json().get("detail"))
            response.raise_for_status()
            return response.json()
    return load_predictor().predict_one(load)


def backend_badge() -> None:
    url = api_url()
    if url:
        try:
            ok = requests.get(f"{url}/api/health", timeout=3).json().get("model_loaded", False)
        except requests.RequestException:
            ok = False
        status = "healthy" if ok else "unreachable, using the embedded model"
        st.sidebar.caption(f"Backend: FastAPI at `{url}` ({status})")
    else:
        st.sidebar.caption("Backend: embedded model (same code path as the API)")


def kpi(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)
