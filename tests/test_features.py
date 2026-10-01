"""Feature engineering: market calendar, gap filling and feature sets."""

import numpy as np
import pandas as pd
import pytest

from freight_rate.data import clean_frame
from freight_rate.features import FEATURE_SETS, FreightFeatures, MarketCalendar


@pytest.fixture(scope="module")
def features(train_frame, reference) -> FreightFeatures:
    return FreightFeatures(feature_set="full", reference=reference).fit(clean_frame(train_frame))


def test_calendar_is_continuous_and_daily(reference):
    daily = reference.calendar.daily
    assert daily.index.is_monotonic_increasing
    assert (daily.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()
    assert not daily.isna().any()


def test_calendar_rolling_means_only_look_backwards():
    frame = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=10).repeat(2),
                          "market_index": np.repeat(np.arange(10, dtype=float), 2)})
    calendar = MarketCalendar.from_frames(frame)
    # Day 0 has no history: its 7-day mean is itself, not pulled up by later days.
    assert calendar.roll7.iloc[0] == 0.0
    assert calendar.roll7.iloc[6] == pytest.approx(np.mean(range(7)))


def test_dates_after_the_calendar_reuse_the_same_weekday(reference):
    calendar = reference.calendar
    future = calendar.last_date + pd.Timedelta(days=7)
    expected = calendar.daily.loc[calendar.last_date]
    assert calendar.daily_for(pd.Series([future]))[0] == pytest.approx(expected)


def test_transform_has_no_missing_values(features, train_frame):
    matrix = features.transform(clean_frame(train_frame))
    assert list(matrix.columns) == list(FEATURE_SETS["full"])
    assert not matrix.isna().any().any()


def test_missing_market_index_uses_that_days_market(features, reference):
    row = clean_frame(pd.DataFrame([{
        "pickup": "Lexington", "delivery": "Fort Wayne", "pickup_lat": 38.04, "pickup_lon": -84.5,
        "delivery_lat": 41.08, "delivery_lon": -85.14, "distance": 360.0, "equipment": "Dry Van",
        "weight": 30_000.0, "date": "2025-06-10", "market_index": np.nan, "quote_signal": 2.0,
    }]))
    value = features.transform(row)["market_index"].iloc[0]
    assert value == pytest.approx(reference.calendar.daily.loc["2025-06-10"])


def test_coordinates_are_filled_from_the_city_lookup(features, reference):
    row = clean_frame(pd.DataFrame([{
        "pickup": "Lexington", "delivery": "Fort Wayne", "pickup_lat": np.nan, "pickup_lon": np.nan,
        "delivery_lat": np.nan, "delivery_lon": np.nan, "distance": 360.0, "equipment": "Dry Van",
        "weight": 32_000.0, "date": "2025-12-01", "market_index": np.nan, "quote_signal": np.nan,
    }]))
    matrix = features.transform(row)
    assert matrix["pickup_lat"].iloc[0] == pytest.approx(reference.cities["Lexington"][0])
    assert not matrix.isna().any().any()


def test_missing_distance_is_estimated_from_coordinates(features, train_frame):
    row = clean_frame(train_frame.head(1).assign(distance=np.nan))
    assert features.transform(row)["distance"].iloc[0] > 0


@pytest.mark.parametrize("feature_set", sorted(FEATURE_SETS))
def test_every_feature_set_transforms(feature_set, train_frame, reference):
    cleaned = clean_frame(train_frame)
    matrix = FreightFeatures(feature_set=feature_set, reference=reference).fit(cleaned).transform(cleaned)
    assert matrix.shape == (len(train_frame), len(FEATURE_SETS[feature_set]))


def test_no_quote_set_really_drops_quote_features():
    assert not any(name.startswith("quote") for name in FEATURE_SETS["no_quote"])


def test_unknown_feature_set_is_rejected(train_frame):
    with pytest.raises(ValueError, match="unknown feature_set"):
        FreightFeatures(feature_set="nope").fit(clean_frame(train_frame))
