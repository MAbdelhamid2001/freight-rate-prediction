"""Upload a CSV of loads, get it back priced."""

import io

import pandas as pd
import streamlit as st

from app.common import price_loads
from freight_rate import config

st.title("Batch pricing")
st.write(
    "Upload a CSV with at least `pickup`, `delivery`, `equipment` and `date`. Optional columns "
    "(`distance`, `weight`, coordinates, `market_index`, `quote_signal`) are used when present and "
    "filled by the model otherwise. Up to 5,000 rows per file."
)

template = pd.DataFrame([
    {"pickup": "Lexington", "delivery": "Fort Wayne", "equipment": "Dry Van", "date": "2025-12-01",
     "distance": 360, "weight": 32000},
    {"pickup": "Atlanta", "delivery": "Nashville", "equipment": "Reefer", "date": "2025-12-02",
     "distance": 250, "weight": 38000},
])
st.download_button("Download a template", template.to_csv(index=False), "loads_template.csv", "text/csv")

upload = st.file_uploader("Loads CSV", type="csv")
if upload is None:
    st.stop()

frame = pd.read_csv(upload)
missing = {"pickup", "delivery", "equipment", "date"} - set(frame.columns)
if missing:
    st.error(f"Missing required columns: {sorted(missing)}")
    st.stop()
if len(frame) > 5_000:
    st.error("Please upload at most 5,000 rows.")
    st.stop()

keep = [c for c in config.RAW_FEATURES if c in frame.columns]
records = frame[keep].astype(object).where(frame[keep].notna(), None).to_dict(orient="records")
for record in records:
    record["date"] = pd.to_datetime(record["date"]).strftime("%Y-%m-%d")

try:
    with st.spinner(f"Pricing {len(records):,} loads ..."):
        frame["predicted_rate"] = price_loads(records)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Loads priced", f"{len(frame):,}")
c2.metric("Median rate", f"${frame.predicted_rate.median():,.0f}")
c3.metric("Total", f"${frame.predicted_rate.sum():,.0f}")
st.dataframe(frame, use_container_width=True, hide_index=True)
buffer = io.StringIO()
frame.to_csv(buffer, index=False)
st.download_button("Download priced CSV", buffer.getvalue(), "priced_loads.csv", "text/csv", type="primary")
