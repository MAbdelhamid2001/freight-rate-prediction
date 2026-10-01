"""Price one load, then see how its price moves across a date range."""

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.common import PRIMARY, load_predictor, price_loads, price_one
from freight_rate import config

st.title("Price a load")
try:
    predictor = load_predictor()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

cities = sorted(predictor.cities)
calendar = predictor.market_calendar
first, last = calendar.index.min().date(), calendar.index.max().date()

with st.form("load"):
    c1, c2, c3 = st.columns(3)
    pickup = c1.selectbox("Pickup city", cities, index=cities.index("Lexington") if "Lexington" in cities else 0)
    delivery = c2.selectbox("Delivery city", cities,
                            index=cities.index("Fort Wayne") if "Fort Wayne" in cities else 1)
    equipment = c3.selectbox("Equipment", config.EQUIPMENT_TYPES)
    c4, c5, c6 = st.columns(3)
    load_date = c4.date_input("Pickup date", value=date(2025, 12, 15), min_value=first, max_value=last + timedelta(days=60),
                              help=f"Market calendar known from {first} to {last}; later dates reuse the last week")
    weight = c5.number_input("Weight (lb)", min_value=1_000, max_value=47_500, value=32_000, step=500)
    auto_distance = c6.checkbox("Estimate distance from coordinates", value=False)
    distance = c6.number_input("Distance (miles)", min_value=10.0, max_value=4_000.0, value=360.0, step=5.0,
                               disabled=auto_distance)
    submitted = st.form_submit_button("Price this load", type="primary")

if pickup == delivery:
    st.warning("Pickup and delivery must differ.")
    st.stop()

load = {"pickup": pickup, "delivery": delivery, "equipment": equipment, "date": load_date.isoformat(),
        "weight": float(weight), "distance": None if auto_distance else float(distance)}

try:
    result = price_one(load)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Predicted posted rate", f"${result['predicted_rate']:,.2f}")
c2.metric("Rate per mile", f"${result['rate_per_mile']:.2f}" if result["rate_per_mile"] else "n/a")
c3.metric("Distance used", f"{result['distance_used']:,.0f} mi")
if result["imputed_fields"]:
    st.caption("Filled in by the model: " + ", ".join(result["imputed_fields"]))

# ---------------------------------------------------------------------------
st.subheader("Same load across a date range")
range_start, range_end = st.slider("Dates", min_value=first, max_value=last + timedelta(days=30),
                                   value=(date(2025, 11, 1), date(2025, 12, 31)), format="YYYY-MM-DD")
dates = pd.date_range(range_start, range_end, freq="D")
rows = [{**load, "date": d.strftime("%Y-%m-%d")} for d in dates]
try:
    rates = price_loads(rows)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
curve = pd.DataFrame({"date": dates, "predicted_rate": rates})
fig = go.Figure(go.Scatter(x=curve.date, y=curve.predicted_rate, mode="lines+markers",
                           line=dict(color=PRIMARY, width=2), marker=dict(size=6),
                           hovertemplate="%{x|%a %d %b %Y}<br>$%{y:,.2f}<extra></extra>"))
fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="predicted rate ($)",
                  hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)
st.caption(f"Range ${curve.predicted_rate.min():,.0f} to ${curve.predicted_rate.max():,.0f}. "
           "Only the date changes; the model's date sensitivity comes from the day-of-week pattern.")
