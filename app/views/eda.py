"""Interactive EDA rendered from reports/eda_summary.json (aggregates only)."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.common import CRITICAL, EQUIPMENT_COLORS, NEUTRAL, PRIMARY, load_eda_summary

summary = load_eda_summary()
st.title("Data exploration")
if not summary:
    st.warning("reports/eda_summary.json is missing. Run `uv run freight-eda`.")
    st.stop()

st.caption("Every chart on this page is built from pre-aggregated statistics, so no raw load is published.")

rows, periods = summary["rows"], summary["periods"]
cols = st.columns(4)
cols[0].metric("Labelled loads", f"{rows['train']:,}", help=f"{periods['train'][0]} to {periods['train'][1]}")
cols[1].metric("Loads to price", f"{rows['validation']:,}",
               help=f"{periods['validation'][0]} to {periods['validation'][1]}")
cols[2].metric("Loads on unseen cities", f"{summary['unseen_city_rows_in_validation']:,}",
               help="Validation loads whose pickup or delivery city never appears in training")
shift = summary["market_shift"]
cols[3].metric("Market index, train to validation", f"{shift['validation_mean']:.3f}",
               delta=f"{shift['validation_mean'] - shift['train_mean']:+.3f}")


def _hist_figure(hist: dict, color: str, x_title: str, log_y: bool = False) -> go.Figure:
    edges = np.asarray(hist["edges"])
    fig = go.Figure(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=hist["counts"], marker_color=color,
                           marker_line_width=0, hovertemplate="%{x:.2f}: %{y:,} loads<extra></extra>"))
    fig.update_layout(bargap=0.05, xaxis_title=x_title, yaxis_title="loads", height=330,
                      margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    if log_y:
        fig.update_yaxes(type="log")
    return fig


tab_target, tab_market, tab_quality, tab_geo, tab_corr = st.tabs(
    ["Target", "Market & time", "Data quality", "Geography", "Correlations"])

with tab_target:
    left, right = st.columns(2)
    left.markdown("**Posted rate (log10 $)**: right-skewed, hence a log-scale target")
    left.plotly_chart(_hist_figure(summary["target"]["log10_hist"], PRIMARY, "log10(posted rate $)"),
                      use_container_width=True)
    right.markdown("**Rate per mile**: a tight core and two corrupted tails (log scale)")
    right.plotly_chart(_hist_figure(summary["target"]["rpm_hist"], PRIMARY, "$ per mile", log_y=True),
                       use_container_width=True)

    bands = pd.DataFrame(summary["distance_bands"])
    st.markdown("**Median $/mile by distance band**: falls with distance; Reefer > Flatbed > Dry Van")
    fig = px.line(bands, x="distance_band", y="p50", color="equipment", markers=True,
                  color_discrete_map=EQUIPMENT_COLORS,
                  labels={"p50": "median $ per mile", "distance_band": "distance band (miles)"},
                  hover_data={"p10": ":.2f", "p90": ":.2f", "p50": ":.2f"})
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

with tab_market:
    daily = pd.DataFrame(summary["daily"])
    daily["date"] = pd.to_datetime(daily["date"])
    daily["trend"] = daily["market_index"].rolling(28, min_periods=1).mean()
    st.markdown("**Daily market index** with its 28-day trailing mean. Validation (Nov-Dec) is shaded.")
    fig = go.Figure()
    fig.add_vrect(x0="2025-11-01", x1=daily.date.max(), fillcolor=NEUTRAL, opacity=0.12, line_width=0)
    fig.add_trace(go.Scatter(x=daily.date, y=daily.market_index, name="daily mean",
                             line=dict(color=PRIMARY, width=1.2)))
    fig.add_trace(go.Scatter(x=daily.date, y=daily.trend, name="28-day mean", line=dict(color=NEUTRAL, width=2.5)))
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      yaxis_title="market index", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Daily median $/mile** (labelled period only): smooth, no weekly cycle")
    rpm = daily.dropna(subset=["median_rpm"])
    fig = go.Figure(go.Scatter(x=rpm.date, y=rpm.median_rpm, line=dict(color=PRIMARY, width=1.2), name="daily"))
    fig.add_trace(go.Scatter(x=rpm.date, y=rpm.median_rpm.rolling(28, min_periods=1).mean(), name="28-day mean",
                             line=dict(color=NEUTRAL, width=2.5)))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      yaxis_title="median $ per mile", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        f"Within a day the market index varies by only {shift['within_day_std']:.3f}, across days by "
        f"{shift['across_day_std']:.3f}: it is a daily market value plus noise. Rates track its slow trend, "
        "but Sep-Oct rates are ~5% above Jan-Feb at the same market level, so the mapping drifts. "
        "That is why market-based features lose in forward cross-validation."
    )

with tab_quality:
    quality = pd.DataFrame(summary["quality"]).rename_axis("check").reset_index()
    labels = {
        "missing_weight": "Missing weight", "missing_market_index": "Missing market index",
        "negative_weight": "Negative weight (sign error)", "weight_at_cap": "Weight at 47,500 lb cap",
        "distance_floor_rows": "Distance at 70-mile floor", "circuity_outliers": "Road/crow distance > 1.6",
        "suspect_rate_per_mile": "Rate per mile outside $1-5", "duplicate_ids": "Duplicate load IDs",
        "duplicate_rows": "Duplicate rows", "non_positive_target": "Non-positive rate", "rows": "Rows",
    }
    quality["check"] = quality["check"].map(labels).fillna(quality["check"])
    left, right = st.columns([1, 1])
    left.dataframe(quality, hide_index=True, use_container_width=True)
    right.markdown(
        """
**How each issue is handled**

* **Negative weight**: sign error; its absolute values match the positive distribution, so `abs()`.
* **Missing weight / quote signal**: median of the same equipment type, learned inside the pipeline.
* **Missing market index**: that day's market mean (error ~0.025 instead of ~0.17).
* **Corrupted rates** (~1.4%): flagged out-of-fold, dropped from training, excluded from clean metrics.
* **Impossible circuity**: the ratio feature is clipped to [0.8, 2.5]; road distance itself is kept.
"""
    )
    right.plotly_chart(_hist_figure(summary["weight_hist"], PRIMARY, "weight (lb)"), use_container_width=True)

with tab_geo:
    cities = pd.DataFrame(summary["cities"])
    cities["status"] = np.where(cities.seen_in_training, "in training", "validation only")
    fig = px.scatter_geo(cities, lat="lat", lon="lon", size="loads", color="status", hover_name="city",
                         color_discrete_map={"in training": PRIMARY, "validation only": CRITICAL},
                         scope="usa", size_max=18)
    fig.update_layout(height=480, margin=dict(l=0, r=0, t=0, b=0), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
    unseen = ", ".join(sorted(cities.loc[~cities.seen_in_training, "city"]))
    st.caption(f"Cities that never appear in training: {unseen}. The model uses coordinates, so it can still "
               "price them; the training script rehearses this by hiding 8 cities from a fold.")

with tab_corr:
    corr = pd.DataFrame(summary["correlations"])
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Linear correlation understates quote_signal: its effect is V-shaped around each equipment's "
               "typical value (see the static figure in the report), which the model captures directly.")
