"""Model comparison: every MLflow-tracked run, the winner, and how it was evaluated."""

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from app.common import EQUIPMENT_COLORS, NEUTRAL, PRIMARY, figure_path, load_experiments, load_model_card
from freight_rate import config

st.title("Model comparison")
experiments = load_experiments()
card = load_model_card()
if experiments.empty or not card:
    st.warning("No experiment results yet. Run `uv run freight-train`.")
    st.stop()

st.write(
    f"**{len(experiments)} runs**, all scored on the same three expanding-window folds "
    "(predict Jul-Aug, Aug-Sep, Sep-Oct from everything before). Selection metric: mean MAPE on "
    "clean labels."
)

# ---------------------------------------------------------------------------
st.subheader("Best configuration per model family")
pool = experiments[experiments.stage.isin(["features", "recency", "models", "tuning"])]
best = (pool.sort_values(config.SELECTION_METRIC).drop_duplicates("model")
        .sort_values(config.SELECTION_METRIC))
winner = card["estimator"]
best["selected"] = best.model.eq(winner).map({True: "selected", False: "other"})
fig = px.bar(best, x=config.SELECTION_METRIC, y="model", orientation="h", color="selected",
             color_discrete_map={"selected": PRIMARY, "other": NEUTRAL},
             hover_data={"holdout_clean_mape": ":.3f", "cv_all_mae": ":.1f", "feature_set": True,
                         "selected": False},
             labels={config.SELECTION_METRIC: "forward-CV MAPE, clean labels (%)", "model": ""})
fig.update_layout(height=60 * len(best) + 80, showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                  yaxis=dict(categoryorder="total descending"))
st.plotly_chart(fig, width="stretch")

table = best[["model", "feature_set", "half_life_days", "cv_clean_mape", "cv_clean_mape_std",
              "holdout_clean_mape", "cv_clean_mae", "cv_all_mae", "cv_all_mape", "cv_clean_r2", "fit_seconds"]]
st.dataframe(
    table.rename(columns={
        "cv_clean_mape": "CV MAPE %", "cv_clean_mape_std": "CV MAPE std", "holdout_clean_mape": "Sep-Oct MAPE %",
        "cv_clean_mae": "CV MAE $ (clean)", "cv_all_mae": "CV MAE $ (all)", "cv_all_mape": "CV MAPE % (all)",
        "cv_clean_r2": "CV R²", "fit_seconds": "CV seconds", "half_life_days": "half-life (d)",
    }).round(3),
    hide_index=True, width="stretch",
)

# ---------------------------------------------------------------------------
st.subheader("What each study stage decided")
stage_tabs = st.tabs(["Feature sets", "Recency weighting", "Hyper-parameter tuning", "All runs"])
with stage_tabs[0]:
    features = experiments[experiments.stage == "features"]
    fig = px.bar(features, x="feature_set", y=config.SELECTION_METRIC, color="model", barmode="group",
                 color_discrete_sequence=[PRIMARY, EQUIPMENT_COLORS["Reefer"]],
                 labels={config.SELECTION_METRIC: "forward-CV MAPE %", "feature_set": "feature set"})
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="")
    st.plotly_chart(fig, width="stretch")
    st.caption("Market-index features and the trend feature fit the past but hurt the next two months. "
               "Dropping quote_signal, whose effect flips sign month to month, helps both families.")
with stage_tabs[1]:
    recency = experiments[experiments.stage.isin(["recency"])]
    st.dataframe(recency[["candidate", "half_life_days", "cv_clean_mape", "holdout_clean_mape"]].round(3),
                 hide_index=True, width="stretch")
with stage_tabs[2]:
    tuning = experiments[experiments.stage == "tuning"].copy()
    if tuning.empty:
        st.info("No tuning runs recorded.")
    else:
        tuning["trial"] = tuning.groupby("model").cumcount()
        tuning["best so far"] = tuning.groupby("model")[config.SELECTION_METRIC].cummin()
        fig = px.line(tuning, x="trial", y="best so far", color="model", markers=True,
                      labels={"best so far": "best forward-CV MAPE so far %", "trial": "Optuna trial"})
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="")
        st.plotly_chart(fig, width="stretch")
        hp_cols = [c for c in tuning.columns if c.startswith("hp_")]
        model_pick = st.selectbox("Trials of", sorted(tuning.model.unique()))
        part = tuning[tuning.model == model_pick].dropna(axis=1, how="all")
        st.dataframe(part[["trial", config.SELECTION_METRIC, "holdout_clean_mape"]
                          + [c for c in hp_cols if c in part]].sort_values(config.SELECTION_METRIC).round(4),
                     hide_index=True, width="stretch")
with stage_tabs[3]:
    st.dataframe(experiments.drop(columns=[c for c in experiments.columns if c.startswith("hp_")]).round(3),
                 hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
st.subheader("Selected model")
cols = st.columns(4)
cols[0].metric("Estimator", card["estimator"])
cols[1].metric("Feature set", card["feature_set"])
cols[2].metric("Training loads", f"{card['training_rows']:,}")
cols[3].metric("Dropped corrupted labels", f"{card['dropped_noisy_labels']:,}")
with st.expander("Hyper-parameters and features"):
    st.json({"hyperparameters": card["hyperparameters"], "features": card["features"],
             "half_life_days": card["half_life_days"]})

st.markdown("**Per-fold results**")
folds = pd.DataFrame(card["cv_per_fold"])
st.dataframe(folds[["fold", "n_train", "n_test", "clean_mape", "clean_mae", "clean_r2", "clean_bias",
                    "all_mape", "all_mae", "noisy_rows"]].round(3), hide_index=True, width="stretch")

checks = card.get("robustness_checks", {})
if checks:
    st.markdown("**Robustness checks**")
    c = st.columns(3)
    c[0].metric("Time split vs random split", f"{checks['time_split_cv_clean_mape']:.2f}%",
                delta=f"random: {checks['random_split_cv_clean_mape']:.2f}%", delta_color="off")
    c[1].metric("With vs without noise filter", f"{checks['time_split_cv_clean_mape']:.2f}%",
                delta=f"without: {checks['no_noise_filter_cv_clean_mape']:.2f}%", delta_color="off")
    c[2].metric("Unseen vs seen cities", f"{checks['unseen_city_clean_mape']:.2f}%",
                delta=f"seen: {checks['seen_city_clean_mape']:.2f}%", delta_color="off",
                help=f"Hidden from training: {checks['hidden_cities']}")

# ---------------------------------------------------------------------------
evaluation_path = config.EVALUATION_PATH
if evaluation_path.is_file():
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    holdout = evaluation["holdout"]
    st.subheader(f"Evaluation on the {holdout['fold']} holdout")
    c = st.columns(4)
    c[0].metric("MAPE (clean)", f"{holdout['overall_clean']['mape']:.2f}%")
    c[1].metric("MAE (clean)", f"${holdout['overall_clean']['mae']:.0f}")
    c[2].metric("Within ±5%", f"{holdout['within_5pct_clean']:.1f}%")
    c[3].metric("R² (clean)", f"{holdout['overall_clean']['r2']:.4f}")
    seg_tabs = st.tabs(["By equipment", "By distance", "By month"])
    for tab, key in zip(seg_tabs, ("by_equipment", "by_distance_band", "by_month")):
        tab.dataframe(pd.DataFrame(holdout[key]), hide_index=True, width="stretch")
    if figure_path("07_holdout_diagnostics.png").is_file():
        st.image(str(figure_path("07_holdout_diagnostics.png")), width="stretch")

    st.subheader("Inference")
    inference = evaluation["inference"]
    sanity = evaluation["validation_predictions"]
    c = st.columns(4)
    c[0].metric("Single load, p50", f"{inference['single_load_ms_p50']:.1f} ms")
    c[1].metric("Single load, p95", f"{inference['single_load_ms_p95']:.1f} ms")
    c[2].metric("Batch throughput", f"{inference['batch_rows_per_second']:,} loads/s")
    c[3].metric("Nov-Dec predictions", f"{sanity['rows']:,}", help=f"${sanity['min']:,.0f} to ${sanity['max']:,.0f}")
    st.dataframe(pd.DataFrame(sanity["by_equipment"]).T.rename(columns={
        "predicted_median_rpm": "Nov-Dec predicted median $/mile",
        "sep_oct_median_rpm": "Sep-Oct actual median $/mile"}), width="stretch")

if figure_path("09_feature_importance.png").is_file():
    with st.expander("Feature importance"):
        st.image(str(figure_path("09_feature_importance.png")))
