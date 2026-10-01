"""Loading, schema checks and row-level cleaning.

Cleaning here is deliberately *stateless*: every rule depends only on the row
itself, so it behaves identically on the training data, on validation.csv and
on a single request hitting the API. Anything that has to be learned from data
(medians, the market calendar) lives in features.py and is fitted inside the
pipeline, which is what keeps the cross-validation honest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from freight_rate import config


class SchemaError(ValueError):
    """Raised when an input frame is missing required columns or is unusable."""


def load_csv(path: Path, *, require_target: bool = False) -> pd.DataFrame:
    """Read one of the task CSVs and verify it carries the expected columns."""
    frame = pd.read_csv(path)
    required = set(config.RAW_FEATURES) | ({config.TARGET} if require_target else set())
    missing = required - set(frame.columns)
    if missing:
        raise SchemaError(f"{path.name} is missing columns: {sorted(missing)}")
    frame[config.DATE_COL] = pd.to_datetime(frame[config.DATE_COL], errors="raise")
    return frame


def load_training_data() -> pd.DataFrame:
    return load_csv(config.TRAIN_PATH, require_target=True)


def load_validation_data() -> pd.DataFrame:
    return load_csv(config.VALIDATION_PATH)


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the row-level data-quality rules. Returns a new frame."""
    out = frame.copy()

    for col in ("pickup", "delivery", "equipment"):
        out[col] = out[col].astype("string").str.strip()

    out[config.DATE_COL] = pd.to_datetime(out[config.DATE_COL])

    # Negative weights are sign errors, not a different population: their
    # absolute values follow the same distribution as the positive ones
    # (median 31.8k vs 31.4k lb), so the sign is dropped rather than the row.
    weight = pd.to_numeric(out["weight"], errors="coerce").abs()
    # Zero is not a real load; let the fitted imputer handle it like a missing value.
    weight = weight.where(weight > 0)
    out["weight"] = weight.clip(upper=config.MAX_WEIGHT_LB)

    for col in ("distance", "market_index", "quote_signal",
                "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


class FrameCleaner(BaseEstimator, TransformerMixin):
    """`clean_frame` as a pipeline step, so the saved model cleans its own input."""

    def fit(self, X: pd.DataFrame, y=None) -> "FrameCleaner":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return clean_frame(X)


# ---------------------------------------------------------------------------
# Data-quality report
# ---------------------------------------------------------------------------
@dataclass
class QualityReport:
    rows: int
    duplicate_ids: int
    duplicate_rows: int
    missing_weight: int
    missing_market_index: int
    negative_weight: int
    weight_at_cap: int
    distance_floor_rows: int
    circuity_outliers: int
    non_positive_target: int | None = None
    suspect_rate_per_mile: int | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in miles; vectorised over arrays or Series."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 3958.8 * np.arcsin(np.sqrt(a))


def quality_report(frame: pd.DataFrame) -> QualityReport:
    """Count every data-quality issue the cleaning rules address."""
    crow = haversine_miles(frame.pickup_lat, frame.pickup_lon, frame.delivery_lat, frame.delivery_lon)
    circuity = frame["distance"] / np.maximum(crow, 1.0)
    has_target = config.TARGET in frame.columns

    report = QualityReport(
        rows=len(frame),
        duplicate_ids=int(frame[config.ID_COL].duplicated().sum()) if config.ID_COL in frame else 0,
        duplicate_rows=int(frame.drop(columns=[config.ID_COL], errors="ignore").duplicated().sum()),
        missing_weight=int(frame["weight"].isna().sum()),
        missing_market_index=int(frame["market_index"].isna().sum()),
        negative_weight=int((frame["weight"] < 0).sum()),
        weight_at_cap=int((frame["weight"].abs() >= config.MAX_WEIGHT_LB).sum()),
        distance_floor_rows=int((frame["distance"] <= 70).sum()),
        circuity_outliers=int((circuity > 1.6).sum()),
    )
    if has_target:
        rpm = frame[config.TARGET] / frame["distance"]
        report.non_positive_target = int((frame[config.TARGET] <= 0).sum())
        # Clean lanes sit between ~1.6 and ~3.2 $/mile; outside 1-5 is suspicious.
        report.suspect_rate_per_mile = int(((rpm > 5) | (rpm < 1)).sum())
    return report
