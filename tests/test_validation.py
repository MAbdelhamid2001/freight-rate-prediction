"""Time folds and metrics: the parts that decide which model wins."""

import numpy as np
import pandas as pd
import pytest

from freight_rate.validation import evaluate, random_folds, regression_metrics, time_folds


def test_time_folds_never_train_on_the_future(train_frame):
    dates = pd.to_datetime(train_frame["date"])
    for fold in time_folds(train_frame):
        assert dates.iloc[fold.train_idx].max() < dates.iloc[fold.test_idx].min()


def test_time_folds_test_two_month_blocks(train_frame):
    dates = pd.to_datetime(train_frame["date"])
    folds = list(time_folds(train_frame))
    assert [f.name for f in folds] == ["Jul-Aug", "Aug-Sep", "Sep-Oct"]
    for fold in folds:
        assert dates.iloc[fold.test_idx].dt.to_period("M").nunique() == 2


def test_time_folds_expand(train_frame):
    sizes = [len(f.train_idx) for f in time_folds(train_frame)]
    assert sizes == sorted(sizes)


def test_random_folds_cover_every_row_once(train_frame):
    seen = np.concatenate([f.test_idx for f in random_folds(train_frame)])
    assert sorted(seen) == list(range(len(train_frame)))


def test_metrics_on_a_known_example():
    metrics = regression_metrics([100, 200], [110, 180])
    assert metrics["mae"] == pytest.approx(15)
    assert metrics["mape"] == pytest.approx(10)
    assert metrics["bias"] == pytest.approx(-5)


def test_evaluate_separates_noisy_rows():
    y = np.array([100.0, 100.0, 100.0, 1000.0])
    pred = np.array([100.0, 100.0, 100.0, 100.0])
    result = evaluate(y, pred, noisy_mask=[False, False, False, True])
    assert result["clean_mape"] == 0
    assert result["all_mae"] == pytest.approx(225)
    assert result["noisy_rows"] == 1
