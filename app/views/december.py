"""The fixed December chart from the assessment, computed live from the model."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.common import NEUTRAL, PRIMARY, load_model_card, price_loads

features = set(load_model_card().get("features", []))
uses_market = any(f.startswith("market") for f in features)
st.title("December 2025 chart")
st.write(
    "Fixed load: **Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb**. Only the date changes. "
    "The input file carries no coordinates, market or quote columns. The model fills coordinates from "
    "its city lookup"
    + (" and the market index from the daily December market observed in the validation loads."
       if uses_market else
       ". The selected model does not use the market index (it lost in forward validation), so the "
       "date enters only through the day-of-week pattern.")
)

rows = [{"pickup": "Lexington", "delivery": "Fort Wayne", "distance": 360.0, "equipment": "Dry Van",
         "weight": 32_000.0, "date": d.strftime("%Y-%m-%d")}
        for d in pd.date_range("2025-12-01", "2025-12-31", freq="D")]
try:
    rates = price_loads(rows)
except Exception as exc:  # surfaced to the user, not swallowed
    st.error(str(exc))
    st.stop()

chart = pd.DataFrame({"date": pd.to_datetime([r["date"] for r in rows]), "predicted_rate": rates})
chart["weekday"] = chart.date.dt.day_name()
mean_rate = chart.predicted_rate.mean()

fig = go.Figure()
fig.add_hline(y=mean_rate, line=dict(color=NEUTRAL, width=1, dash="dot"),
              annotation_text=f"mean ${mean_rate:,.0f}", annotation_position="top left")
fig.add_trace(go.Scatter(x=chart.date, y=chart.predicted_rate, mode="lines+markers",
                         line=dict(color=PRIMARY, width=2.5), marker=dict(size=7),
                         customdata=chart.weekday,
                         hovertemplate="%{customdata} %{x|%d %b}<br>$%{y:,.2f}<extra></extra>"))
fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="predicted rate ($)",
                  showlegend=False)
st.plotly_chart(fig, width="stretch")

c1, c2, c3 = st.columns(3)
c1.metric("Mean", f"${mean_rate:,.2f}")
c2.metric("Range", f"${chart.predicted_rate.min():,.0f} to ${chart.predicted_rate.max():,.0f}")
c3.metric("Rate per mile", f"${mean_rate / 360:.2f}")

st.caption("The weekly ripple is learned from the data (rates peak mid-week). There is no trend across the "
           "month because the December market level sits flat, close to Sep-Oct.")
with st.expander("Table"):
    st.dataframe(chart, hide_index=True, width="stretch")
