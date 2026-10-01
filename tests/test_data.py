"""Row-level cleaning rules and the data-quality report."""

import numpy as np
import pandas as pd
import pytest

from freight_rate import config
from freight_rate.data import SchemaError, clean_frame, haversine_miles, load_csv, quality_report


def _row(**overrides) -> pd.DataFrame:
    base = {"load_id": "X-1", "pickup": " Lexington ", "delivery": "Fort Wayne", "pickup_lat": 38.0,
            "pickup_lon": -84.5, "delivery_lat": 41.1, "delivery_lon": -85.1, "distance": 360.0,
            "equipment": "Dry Van", "weight": 30_000.0, "date": "2025-12-01", "market_index": 1.0,
            "quote_signal": 2.0}
    return pd.DataFrame([{**base, **overrides}])


def test_negative_weight_is_treated_as_a_sign_error():
    assert clean_frame(_row(weight=-25_000))["weight"].iloc[0] == 25_000


def test_zero_weight_becomes_missing():
    assert np.isnan(clean_frame(_row(weight=0))["weight"].iloc[0])


def test_weight_is_capped_at_the_legal_maximum():
    assert clean_frame(_row(weight=60_000))["weight"].iloc[0] == config.MAX_WEIGHT_LB


def test_strings_are_stripped_and_dates_parsed():
    cleaned = clean_frame(_row())
    assert cleaned["pickup"].iloc[0] == "Lexington"
    assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])


def test_clean_frame_does_not_mutate_its_input():
    frame = _row(weight=-1_000)
    clean_frame(frame)
    assert frame["weight"].iloc[0] == -1_000


def test_haversine_known_distance():
    # New York to Los Angeles is ~2,445 miles great-circle.
    assert haversine_miles(40.7128, -74.0060, 34.0522, -118.2437) == pytest.approx(2445, rel=0.01)


def test_quality_report_counts_defects(train_frame):
    report = quality_report(train_frame)
    assert report.rows == len(train_frame)
    assert report.missing_weight == train_frame["weight"].isna().sum()
    assert report.negative_weight == (train_frame["weight"] < 0).sum() > 0
    assert report.missing_market_index > 0
    assert report.non_positive_target == 0


def test_load_csv_rejects_missing_columns(tmp_path):
    path = tmp_path / "broken.csv"
    _row().drop(columns=["distance"]).to_csv(path, index=False)
    with pytest.raises(SchemaError, match="distance"):
        load_csv(path)


def test_load_csv_requires_target_when_asked(tmp_path):
    path = tmp_path / "no_target.csv"
    _row().to_csv(path, index=False)
    with pytest.raises(SchemaError, match="posted_rate"):
        load_csv(path, require_target=True)
