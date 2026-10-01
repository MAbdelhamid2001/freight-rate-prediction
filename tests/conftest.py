"""Shared fixtures.

The real dataset is not in the repository, so the suite runs on a small
synthetic dataset with the same schema and the same kinds of defects
(negative and missing weights, missing market index). A small model is
trained on it once per session and injected into the API, so the tests never
depend on files that only exist on the author's machine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from freight_rate.features import ReferenceData
from freight_rate.inference import FreightRatePredictor
from freight_rate.models import build_pipeline

CITIES = {
    "Lexington": (38.04, -84.50), "Fort Wayne": (41.08, -85.14), "Atlanta": (33.75, -84.39),
    "Nashville": (36.16, -86.78), "Dallas": (32.78, -96.80), "Phoenix": (33.45, -112.07),
    "Denver": (39.74, -104.99), "Richmond": (37.54, -77.44), "Memphis": (35.15, -90.05),
    "Tulsa": (36.15, -95.99),
}
RPM = {"Dry Van": 2.05, "Reefer": 2.30, "Flatbed": 2.20}


def make_loads(n: int, start: str, end: str, seed: int, with_target: bool = True, prefix: str = "TR") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(CITIES)
    pickup = rng.choice(names, n)
    delivery = np.array([rng.choice([c for c in names if c != p]) for p in pickup])
    plat, plon = np.array([CITIES[c] for c in pickup]).T
    dlat, dlon = np.array([CITIES[c] for c in delivery]).T
    crow = 69.0 * np.hypot(dlat - plat, (dlon - plon) * np.cos(np.radians((plat + dlat) / 2)))
    distance = np.round(np.maximum(crow * rng.uniform(1.1, 1.3, n), 70), 1)
    equipment = rng.choice(list(RPM), n, p=[0.55, 0.25, 0.20])
    dates = pd.to_datetime(rng.choice(pd.date_range(start, end, freq="D"), n))
    weight = rng.normal(31_000, 8_000, n).clip(5_000, 47_500).round()
    market = 1.0 + 0.1 * np.sin(2 * np.pi * dates.dayofweek / 7) + rng.normal(0, 0.02, n)

    frame = pd.DataFrame({
        "load_id": [f"{prefix}-{i:06d}" for i in range(1, n + 1)],
        "pickup": pickup, "delivery": delivery,
        "pickup_lat": plat, "pickup_lon": plon, "delivery_lat": dlat, "delivery_lon": dlon,
        "distance": distance, "equipment": equipment, "weight": weight,
        "date": dates.strftime("%Y-%m-%d"), "market_index": market.round(5),
        "quote_signal": rng.normal(2.05, 0.25, n).round(5),
    })
    if with_target:
        rpm = np.array([RPM[e] for e in equipment]) * (distance / 500) ** -0.15
        frame["posted_rate"] = (distance * rpm * (1 + weight / 1e6) * rng.normal(1, 0.02, n)).round(2)
    # Defects of the kind found in the real data.
    frame.loc[frame.sample(frac=0.01, random_state=seed).index, "weight"] = np.nan
    frame.loc[frame.sample(frac=0.01, random_state=seed + 1).index, "market_index"] = np.nan
    negative = frame.sample(frac=0.01, random_state=seed + 2).index
    frame.loc[negative, "weight"] = -frame.loc[negative, "weight"].abs()
    return frame


@pytest.fixture(scope="session")
def train_frame() -> pd.DataFrame:
    frame = make_loads(1_500, "2025-01-01", "2025-10-31", seed=1)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


@pytest.fixture(scope="session")
def validation_frame() -> pd.DataFrame:
    frame = make_loads(300, "2025-11-01", "2025-12-31", seed=2, with_target=False, prefix="TE")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


@pytest.fixture(scope="session")
def reference(train_frame, validation_frame) -> ReferenceData:
    return ReferenceData.from_frames(train_frame, validation_frame)


@pytest.fixture(scope="session")
def fitted_pipeline(train_frame, reference):
    pipeline = build_pipeline("lightgbm", feature_set="base", reference=reference,
                              params={"n_estimators": 60, "num_leaves": 15})
    pipeline.fit(train_frame, train_frame["posted_rate"].to_numpy())
    return pipeline


@pytest.fixture(scope="session")
def predictor(fitted_pipeline) -> FreightRatePredictor:
    card = {"trained_at_utc": "test", "estimator": "lightgbm", "feature_set": "base",
            "training_period": ["2025-01-01", "2025-10-31"], "market_calendar_period": ["2025-01-01", "2025-12-31"],
            "cv_metrics": {"cv_clean_mape": 2.0}}
    return FreightRatePredictor(pipeline=fitted_pipeline, card=card)


@pytest.fixture(scope="session")
def client(predictor) -> TestClient:
    """The real app, with the session's small model injected in place of models/."""
    from api.main import app
    from api.routers.predictions import predictor_dependency

    app.dependency_overrides[predictor_dependency] = lambda: predictor
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def valid_load() -> dict:
    return {"pickup": "Lexington", "delivery": "Fort Wayne", "equipment": "Dry Van",
            "date": "2025-12-15", "distance": 360, "weight": 32000}
