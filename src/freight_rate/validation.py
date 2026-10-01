"""Time-aware validation: folds, metrics and label-noise detection.

The split is the most consequential decision in this task. The labelled data
is Jan-Oct 2025 and the loads to price are Nov-Dec 2025, so the question the
validation must answer is "how well does a model trained on the past price the
next two months?". A random K-fold answers a different, easier question: it
lets the model see the same days (and the same daily market level) in train
and test. `compare_random_vs_time_split` in train.py measures exactly how
optimistic that is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from freight_rate import config


@dataclass(frozen=True)
class Fold:
    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray


def time_folds(frame: pd.DataFrame, folds=config.CV_FOLDS) -> Iterator[Fold]:
    """Expanding-window folds with a 2-month test horizon (see config.CV_FOLDS)."""
    dates = pd.to_datetime(frame[config.DATE_COL])
    for train_end, test_start, test_end in folds:
        train_idx = np.flatnonzero(dates < train_end)
        test_idx = np.flatnonzero((dates >= test_start) & (dates < test_end))
        name = f"{pd.Timestamp(test_start):%b}-{pd.Timestamp(test_end) - pd.Timedelta(days=1):%b}"
        yield Fold(name=name, train_idx=train_idx, test_idx=test_idx)


def random_folds(frame: pd.DataFrame, n_splits: int = 3) -> Iterator[Fold]:
    """Shuffled K-fold, kept only to show how much it overstates accuracy."""
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    for i, (train_idx, test_idx) in enumerate(splitter.split(frame)):
        yield Fold(name=f"random-{i + 1}", train_idx=train_idx, test_idx=test_idx)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred, prefix: str = "") -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    ape = np.abs(err) / np.abs(y_true)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        f"{prefix}mae": float(np.mean(np.abs(err))),
        f"{prefix}rmse": float(np.sqrt(np.mean(err**2))),
        f"{prefix}mape": float(np.mean(ape) * 100),
        f"{prefix}medape": float(np.median(ape) * 100),
        f"{prefix}r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        f"{prefix}bias": float(np.mean(err)),
    }


def evaluate(y_true, y_pred, noisy_mask) -> dict[str, float]:
    """Metrics on every test row, plus on the rows whose label is not corrupted.

    The `all_` numbers are what an external scorer will see (it cannot drop the
    corrupted labels either). The `clean_` numbers are the better signal for
    choosing between models, because ~1.4% of labels are off by 2-5x and their
    huge errors are noise that no model can, or should, fit.
    """
    noisy_mask = np.asarray(noisy_mask, dtype=bool)
    metrics = regression_metrics(y_true, y_pred, prefix="all_")
    metrics |= regression_metrics(np.asarray(y_true)[~noisy_mask], np.asarray(y_pred)[~noisy_mask], prefix="clean_")
    metrics["noisy_rows"] = float(noisy_mask.sum())
    return metrics


# ---------------------------------------------------------------------------
# Label noise
# ---------------------------------------------------------------------------
def flag_label_noise(frame: pd.DataFrame, reference, n_splits: int = 5,
                     ratio: float = config.LABEL_NOISE_RATIO) -> np.ndarray:
    """Flag rows whose posted rate is implausible given everything else.

    A rule on raw $/mile cannot do this: a $4/mile rate is normal for a short
    reefer haul and absurd for a 2,500-mile dry van. Instead, a robust model is
    fitted out-of-fold, and a row is flagged when its label is more than `ratio`
    times away from the prediction made *without seeing that row*.
    Used to (a) drop corrupted labels from training and (b) report clean metrics.
    """
    # Local import: models.py imports this module's neighbours, keep the graph acyclic.
    from freight_rate.models import build_pipeline

    y = frame[config.TARGET].to_numpy(dtype=float)
    oof = np.empty_like(y)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    for train_idx, test_idx in splitter.split(frame):
        scout = build_pipeline(
            "lightgbm", feature_set="full", reference=reference,
            params={"n_estimators": 600, "learning_rate": 0.06},
        )
        scout.fit(frame.iloc[train_idx], y[train_idx])
        oof[test_idx] = scout.predict(frame.iloc[test_idx])
    log_ratio = np.abs(np.log(y / oof))
    return log_ratio > np.log(ratio)
