"""Evaluation and inference report for the selected model.

Called at the end of `freight-train`. Produces

  reports/evaluation.json            holdout metrics by segment, prediction
                                     sanity checks, inference latency
  reports/figures/07_*.png ...       diagnostics and the model comparison chart

The holdout is the last time fold (train Jan-Aug, predict Sep-Oct): the same
shape as the real task, scored by a model that never saw those two months.
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from freight_rate import config
from freight_rate.eda import EQUIPMENT_COLORS, INK_2, MUTED, PRIMARY, _save, _style
from freight_rate.validation import regression_metrics

DISTANCE_BANDS = [0, 250, 500, 1000, 1500, 2000, 3500]


def _segment_table(frame: pd.DataFrame, pred: np.ndarray, noisy: np.ndarray, by: pd.Series) -> list[dict]:
    y = frame[config.TARGET].to_numpy(dtype=float)
    rows = []
    for key, idx in by.groupby(by, observed=True).groups.items():
        pos = frame.index.get_indexer(idx)
        clean = pos[~noisy[pos]]
        metrics = regression_metrics(y[clean], pred[clean])
        rows.append({"segment": str(key), "loads": int(len(clean)),
                     **{k: round(v, 3) for k, v in metrics.items()}})
    return rows


def holdout_evaluation(pipeline_factory, frame: pd.DataFrame, noisy: np.ndarray, fold) -> tuple[dict, pd.DataFrame]:
    """Fit on the fold's clean training rows, score its test rows by segment."""
    train_idx = fold.train_idx[~noisy[fold.train_idx]]
    model = pipeline_factory(frame.iloc[train_idx])
    test = frame.iloc[fold.test_idx].copy()
    test_noisy = noisy[fold.test_idx]
    pred = model.predict(test)
    test["predicted"] = pred
    test["noisy"] = test_noisy
    test = test.reset_index(drop=True)

    y = test[config.TARGET].to_numpy(dtype=float)
    result = {
        "fold": fold.name,
        "overall_clean": {k: round(v, 3) for k, v in regression_metrics(y[~test_noisy], pred[~test_noisy]).items()},
        "overall_all": {k: round(v, 3) for k, v in regression_metrics(y, pred).items()},
        "within_5pct_clean": round(float((np.abs(pred - y) / y < 0.05)[~test_noisy].mean() * 100), 2),
        "within_10pct_clean": round(float((np.abs(pred - y) / y < 0.10)[~test_noisy].mean() * 100), 2),
        "by_equipment": _segment_table(test, pred, test_noisy, test["equipment"]),
        "by_distance_band": _segment_table(test, pred, test_noisy,
                                           pd.cut(test["distance"], DISTANCE_BANDS).astype(str)),
        "by_month": _segment_table(test, pred, test_noisy, test[config.DATE_COL].dt.strftime("%Y-%m")),
    }
    return result, test


def prediction_sanity(final_pipeline, validation: pd.DataFrame, train: pd.DataFrame, noisy: np.ndarray) -> dict:
    """Do the Nov-Dec predictions look like the labelled data they came from?"""
    pred = final_pipeline.predict(validation)
    rpm_pred = pred / validation["distance"].to_numpy()
    clean = train.loc[~noisy]
    rpm_train = (clean[config.TARGET] / clean["distance"]).to_numpy()
    recent = clean[config.DATE_COL] >= "2025-09-01"
    by_equipment = {
        eq: {
            "predicted_median_rpm": round(float(np.median(rpm_pred[validation.equipment.to_numpy() == eq])), 4),
            "sep_oct_median_rpm": round(float(np.median(
                rpm_train[(clean.equipment == eq).to_numpy() & recent.to_numpy()])), 4),
        }
        for eq in config.EQUIPMENT_TYPES
    }
    return {
        "rows": int(len(pred)),
        "min": round(float(pred.min()), 2),
        "median": round(float(np.median(pred)), 2),
        "max": round(float(pred.max()), 2),
        "non_positive": int((pred <= 0).sum()),
        "predicted_rpm_p01_p99": [round(float(np.quantile(rpm_pred, q)), 3) for q in (0.01, 0.99)],
        "by_equipment": by_equipment,
    }


def inference_benchmark(predictor_pipeline, validation: pd.DataFrame, n_single: int = 200) -> dict:
    """Latency of one load (the API path) and throughput of a full batch."""
    one = validation.iloc[[0]]
    predictor_pipeline.predict(one)  # warm-up
    timings = []
    for i in range(n_single):
        row = validation.iloc[[i % len(validation)]]
        start = time.perf_counter()
        predictor_pipeline.predict(row)
        timings.append((time.perf_counter() - start) * 1000)
    start = time.perf_counter()
    predictor_pipeline.predict(validation)
    batch_seconds = time.perf_counter() - start
    return {
        "single_load_ms_p50": round(float(np.percentile(timings, 50)), 2),
        "single_load_ms_p95": round(float(np.percentile(timings, 95)), 2),
        "batch_rows": int(len(validation)),
        "batch_seconds": round(batch_seconds, 3),
        "batch_rows_per_second": int(len(validation) / batch_seconds),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_diagnostics(test: pd.DataFrame, title_suffix: str) -> None:
    _style()
    clean = test[~test.noisy]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for equipment, color in EQUIPMENT_COLORS.items():
        part = clean[clean.equipment == equipment]
        axes[0].scatter(part[config.TARGET], part.predicted, s=3, alpha=0.3, color=color, label=equipment,
                        rasterized=True)
    lims = [clean[config.TARGET].min(), clean[config.TARGET].max()]
    axes[0].plot(lims, lims, color=INK_2, linewidth=1)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title(f"Predicted vs actual ({title_suffix})")
    axes[0].set_xlabel("actual rate $")
    axes[0].set_ylabel("predicted rate $")
    axes[0].legend(markerscale=4, loc="upper left")

    pct = (clean.predicted - clean[config.TARGET]) / clean[config.TARGET] * 100
    axes[1].hist(pct.clip(-15, 15), bins=90, color=PRIMARY, edgecolor="none")
    axes[1].axvline(0, color=INK_2, linewidth=1)
    axes[1].set_title("Percentage error (clean labels)")
    axes[1].set_xlabel("(predicted - actual) / actual, %")
    axes[1].set_ylabel("loads")

    bands = pd.cut(clean.distance, DISTANCE_BANDS)
    ape = (pct.abs()).groupby([bands, clean.equipment], observed=True).mean().unstack()
    x = np.arange(len(ape))
    width = 0.27
    for i, (equipment, color) in enumerate(EQUIPMENT_COLORS.items()):
        axes[2].bar(x + (i - 1) * width, ape[equipment], width=width - 0.03, color=color, label=equipment)
    axes[2].set_xticks(x, [f"{int(b.left)}-{int(b.right)}" for b in ape.index], rotation=30)
    axes[2].set_title("MAPE by distance band and equipment")
    axes[2].set_ylabel("MAPE %")
    axes[2].legend()
    _save(fig, "07_holdout_diagnostics")


def fig_model_comparison(experiments: pd.DataFrame) -> None:
    """Best run per model family (models + tuning stages), forward-CV MAPE."""
    _style()
    pool = experiments[experiments.stage.isin(["models", "tuning", "recency", "features"])]
    best = (pool.sort_values(config.SELECTION_METRIC).drop_duplicates("model")
            .sort_values(config.SELECTION_METRIC, ascending=False))
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(best) + 1.2))
    colors = [PRIMARY if i == len(best) - 1 else MUTED for i in range(len(best))]
    ax.barh(best.model, best[config.SELECTION_METRIC], color=colors, height=0.6)
    for y, (value, hold) in enumerate(zip(best[config.SELECTION_METRIC], best["holdout_clean_mape"])):
        ax.text(value + 0.03, y, f"{value:.2f}%  (Sep-Oct {hold:.2f}%)", va="center", color=INK_2, fontsize=9)
    ax.set_xlim(0, best[config.SELECTION_METRIC].max() * 1.45)
    ax.set_xlabel("forward-CV MAPE on clean labels, % (lower is better)")
    ax.set_title("Best configuration per model family")
    ax.grid(axis="y", visible=False)
    _save(fig, "08_model_comparison")


def fig_feature_importance(importance: pd.Series | None) -> None:
    if importance is None:
        return
    _style()
    top = importance.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index, top.values * 100, color=PRIMARY, height=0.6)
    ax.set_xlabel("share of split importance, %")
    ax.set_title("Feature importance (tree component)")
    ax.grid(axis="y", visible=False)
    _save(fig, "09_feature_importance")
