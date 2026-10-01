"""Landing page: the problem, the headline numbers and the key decisions."""

import streamlit as st

from app.common import load_experiments, load_model_card

card = load_model_card()
experiments = load_experiments()

st.title("Freight Rate Prediction")
st.write(
    "Predicts the **posted rate** of a truckload from its lane, equipment, weight and date. "
    "The model is trained on Jan-Oct 2025 loads and prices Nov-Dec 2025, so everything here, "
    "from validation to model choice, is built around forecasting forward in time."
)

if not card:
    st.warning("No trained model found. Run `uv run freight-train` and `uv run freight-predict` first.")
    st.stop()

cv = card["cv_metrics"]
baseline = experiments.loc[experiments.model == "baseline_rpm", "cv_clean_mape"]
cols = st.columns(4)
cols[0].metric("Forward-CV MAPE", f"{cv['cv_clean_mape']:.2f}%",
               help="Mean over 3 expanding-window folds with a 2-month horizon, corrupted labels excluded")
cols[1].metric("Latest fold (Sep-Oct) MAPE", f"{cv['holdout_clean_mape']:.2f}%",
               help="Train Jan-Aug, predict Sep-Oct: the closest rehearsal of the real task")
cols[2].metric("Forward-CV MAE", f"${cv['cv_all_mae']:.0f}", help="All rows, including corrupted labels")
if not baseline.empty:
    cols[3].metric("vs. $/mile rule of thumb", f"{cv['cv_clean_mape']:.2f}%",
                   delta=f"{cv['cv_clean_mape'] - float(baseline.iloc[0]):.2f} pts", delta_color="inverse",
                   help="Baseline: median $/mile by equipment and distance band")

st.subheader("What drives the approach")
left, right = st.columns(2)
with left:
    st.markdown(
        """
**Time split, not a random split.** Labels cover Jan-Oct, predictions are for Nov-Dec.
Every candidate is scored on three expanding-window folds that each predict the *next
two months*, which is the question the task actually asks. Relationships in this data
drift month to month, and only a forward split measures how a model copes with that.

**Corrupted labels.** About 1.4% of posted rates are 2-5x too high or too low for
their lane. They are detected out-of-fold, dropped from training, and reported
separately in every metric.

**Unseen cities.** 12% of validation loads touch one of 8 cities absent from
training, so the model uses coordinates and distance, never city IDs.
"""
    )
with right:
    st.markdown(
        f"""
**Market index.** One value per day plus noise, with a strong weekly cycle. Rates
follow the market's slow trend but not its daily swings, and that relationship
drifts over the year, so every market-based feature set loses in forward CV and the
selected model does not use it. quote_signal is dropped for the same reason: its
effect flips sign from month to month.

**Model.** `{card['estimator']}` (a spline GAM when it is `ridge`) on the `{card['feature_set']}`
feature set, trained on log(rate), selected by forward-CV MAPE from {len(experiments)}
MLflow-tracked runs across 8 model families, including tuned XGBoost, LightGBM, Random
Forest and a neural network.

**Data quality fixes.** Negative weights (sign errors) are made positive, missing
weight and quote signal are imputed per equipment type, a missing market index
is filled from that day's market mean, and impossible circuity ratios are clipped.
"""
        )

st.caption(f"Model trained {card['trained_at_utc']} on {card['training_rows']:,} clean loads "
           f"({card['training_period'][0]} to {card['training_period'][1]}).")
