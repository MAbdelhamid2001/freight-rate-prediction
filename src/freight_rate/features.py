"""Feature engineering as a fitted scikit-learn transformer.

Design choices, each backed by the EDA:

* **No city identifiers.** 12% of validation rows touch one of 8 cities that
  never appear in training (Chicago, Charlotte, Laredo, ...). Coordinates,
  distance and bearing generalise to an unseen city; a city ID cannot.
* **No month / day-of-year.** Training stops in October and we predict Nov-Dec,
  so any calendar-position feature would be extrapolated. Seasonality reaches
  the model through the market index, which *is* observed for those months.
* **Market calendar.** `market_index` is one market value per day plus small
  per-load noise (within-day std ~0.025 vs ~0.17 across days). Averaging it
  per day denoises it, gives a principled way to fill the 1.4% missing values,
  and gives trailing means, since rates react to the market with some lag.
  The December chart file has no market columns at all, and this calendar is
  what lets those rows still be priced with the real December market.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from freight_rate import config
from freight_rate.data import haversine_miles

FEATURE_SETS: dict[str, tuple[str, ...]] = {}

# Origin of the `trend_days` feature. Any fixed date works; trees only use it
# to order loads in time.
TREND_EPOCH = pd.Timestamp("2025-01-01")

_BASE = (
    "distance",
    "log_distance",
    "crow_miles",
    "circuity",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "delta_lat",
    "delta_lon",
    "bearing_sin",
    "bearing_cos",
    "is_reefer",
    "is_flatbed",
    "weight",
    "quote_signal",
    "quote_dev",
    "quote_abs_dev",
    "day_of_week",
    "is_weekend",
)
_QUOTE = ("quote_signal", "quote_dev", "quote_abs_dev")
_MARKET = ("market_daily", "market_roll7", "market_roll28")

FEATURE_SETS["base"] = _BASE
# quote_signal's effect changes sign from month to month (see the EDA), so
# whether it helps forward in time is decided by cross-validation.
FEATURE_SETS["no_quote"] = tuple(c for c in _BASE if c not in _QUOTE)
FEATURE_SETS["market_raw"] = _BASE + ("market_index",)
FEATURE_SETS["market_smooth"] = _BASE + _MARKET
FEATURE_SETS["full"] = _BASE + ("market_index",) + _MARKET
FEATURE_SETS["base_trend"] = _BASE + ("trend_days",)


# ---------------------------------------------------------------------------
# Reference data: things learned from *feature* columns only (never the target)
# ---------------------------------------------------------------------------
@dataclass
class MarketCalendar:
    """Daily market index with trailing means, indexed by calendar date."""

    daily: pd.Series = field(default_factory=pd.Series)
    roll7: pd.Series = field(default_factory=pd.Series)
    roll28: pd.Series = field(default_factory=pd.Series)

    @classmethod
    def from_frames(cls, *frames: pd.DataFrame) -> "MarketCalendar":
        stacked = pd.concat([f[[config.DATE_COL, "market_index"]] for f in frames], ignore_index=True)
        stacked[config.DATE_COL] = pd.to_datetime(stacked[config.DATE_COL]).dt.normalize()
        daily = stacked.groupby(config.DATE_COL)["market_index"].mean().sort_index()
        # Reindex onto a continuous calendar so gaps are visible and interpolated.
        full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        daily = daily.reindex(full_range).interpolate(limit_direction="both")
        # Trailing windows only: a load is priced on the market up to its own date.
        return cls(
            daily=daily,
            roll7=daily.rolling(7, min_periods=1).mean(),
            roll28=daily.rolling(28, min_periods=1).mean(),
        )

    def _lookup(self, series: pd.Series, dates: pd.Series) -> np.ndarray:
        dates = pd.to_datetime(dates).dt.normalize()
        values = series.reindex(dates.values).to_numpy(dtype=float, copy=True)
        missing = np.isnan(values)
        if missing.any():
            # Outside the known calendar: reuse the same weekday from the last
            # observed week, because the index has a strong weekly cycle.
            last_week = series.iloc[-7:]
            by_weekday = dict(zip(last_week.index.dayofweek, last_week.to_numpy()))
            fallback = dates[missing].dt.dayofweek.map(by_weekday).to_numpy(dtype=float)
            values[missing] = np.where(np.isnan(fallback), float(series.median()), fallback)
        return values

    def daily_for(self, dates: pd.Series) -> np.ndarray:
        return self._lookup(self.daily, dates)

    def roll7_for(self, dates: pd.Series) -> np.ndarray:
        return self._lookup(self.roll7, dates)

    def roll28_for(self, dates: pd.Series) -> np.ndarray:
        return self._lookup(self.roll28, dates)

    @property
    def first_date(self) -> pd.Timestamp:
        return self.daily.index.min()

    @property
    def last_date(self) -> pd.Timestamp:
        return self.daily.index.max()


@dataclass
class ReferenceData:
    """Lookups learned from unlabeled feature columns of every known load."""

    calendar: MarketCalendar
    cities: dict[str, tuple[float, float]]

    @classmethod
    def from_frames(cls, *frames: pd.DataFrame) -> "ReferenceData":
        cities: dict[str, tuple[float, float]] = {}
        for frame in frames:
            for side in ("pickup", "delivery"):
                coords = frame.groupby(side)[[f"{side}_lat", f"{side}_lon"]].first()
                for city, (lat, lon) in coords.iterrows():
                    cities.setdefault(str(city), (float(lat), float(lon)))
        return cls(calendar=MarketCalendar.from_frames(*frames), cities=dict(sorted(cities.items())))


# ---------------------------------------------------------------------------
# The transformer
# ---------------------------------------------------------------------------
class FreightFeatures(BaseEstimator, TransformerMixin):
    """Turns a cleaned load frame into the numeric model matrix.

    Parameters
    ----------
    feature_set:
        Key of FEATURE_SETS. Compared in MLflow; the winner is what ships.
    reference:
        Market calendar + city coordinates. If None it is built from the frame
        passed to `fit`, which is what a plain refit on new data would do.
    """

    def __init__(self, feature_set: str = "full", reference: ReferenceData | None = None):
        self.feature_set = feature_set
        self.reference = reference

    # -- fitting -----------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None) -> "FreightFeatures":
        if self.feature_set not in FEATURE_SETS:
            raise ValueError(f"unknown feature_set {self.feature_set!r}; choose from {list(FEATURE_SETS)}")
        self.reference_ = self.reference or ReferenceData.from_frames(X)
        self.weight_median_ = X.groupby("equipment")["weight"].median().to_dict()
        self.weight_global_median_ = float(X["weight"].median())
        self.quote_median_ = X.groupby("equipment")["quote_signal"].median().to_dict()
        self.quote_global_median_ = float(X["quote_signal"].median())
        crow = haversine_miles(X.pickup_lat, X.pickup_lon, X.delivery_lat, X.delivery_lon)
        self.circuity_median_ = float(np.nanmedian(X["distance"] / np.maximum(crow, 1.0)))
        self.feature_names_ = list(FEATURE_SETS[self.feature_set])
        return self

    # -- helpers -----------------------------------------------------------
    def _fill_coordinates(self, out: pd.DataFrame) -> None:
        """Fill missing lat/lon from the city lookup (the December file has none)."""
        for side in ("pickup", "delivery"):
            for i, axis in enumerate(("lat", "lon")):
                col = f"{side}_{axis}"
                if col not in out:
                    out[col] = np.nan
                gap = out[col].isna()
                if gap.any():
                    out.loc[gap, col] = out.loc[gap, side].map(
                        lambda c, i=i: self.reference_.cities.get(str(c), (np.nan, np.nan))[i]
                    )

    def _by_equipment(self, out: pd.DataFrame, col: str, medians: dict, fallback: float) -> pd.Series:
        fill = out["equipment"].map(medians).astype(float).fillna(fallback)
        return out[col].fillna(fill)

    # -- transform ---------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in ("weight", "market_index", "quote_signal", "distance"):
            if col not in out:
                out[col] = np.nan
        self._fill_coordinates(out)

        cal = self.reference_.calendar
        dates = pd.to_datetime(out[config.DATE_COL])

        crow = haversine_miles(out.pickup_lat, out.pickup_lon, out.delivery_lat, out.delivery_lon)
        distance = out["distance"].fillna(pd.Series(crow * self.circuity_median_, index=out.index))
        bearing = np.arctan2(out.delivery_lon - out.pickup_lon, out.delivery_lat - out.pickup_lat)
        market_daily = cal.daily_for(dates)
        quote = self._by_equipment(out, "quote_signal", self.quote_median_, self.quote_global_median_)

        features = pd.DataFrame(
            {
                "distance": distance,
                "log_distance": np.log1p(distance),
                "crow_miles": crow,
                # Clipped: a handful of coordinates are wrong (e.g. New Orleans-
                # Shreveport at 7 crow miles), and an unbounded ratio would let
                # those few rows dominate the split points.
                "circuity": np.clip(distance / np.maximum(crow, 1.0), 0.8, 2.5),
                "pickup_lat": out.pickup_lat,
                "pickup_lon": out.pickup_lon,
                "delivery_lat": out.delivery_lat,
                "delivery_lon": out.delivery_lon,
                "delta_lat": out.delivery_lat - out.pickup_lat,
                "delta_lon": out.delivery_lon - out.pickup_lon,
                "bearing_sin": np.sin(bearing),
                "bearing_cos": np.cos(bearing),
                "is_reefer": (out["equipment"] == "Reefer").astype(float),
                "is_flatbed": (out["equipment"] == "Flatbed").astype(float),
                "weight": self._by_equipment(out, "weight", self.weight_median_, self.weight_global_median_),
                "quote_signal": quote,
                # quote_signal acts through its distance from the equipment's
                # typical value (a V shape in the EDA), so expose that directly.
                "quote_dev": quote - out["equipment"].map(self.quote_median_).astype(float).fillna(
                    self.quote_global_median_),
                "quote_abs_dev": (quote - out["equipment"].map(self.quote_median_).astype(float).fillna(
                    self.quote_global_median_)).abs(),
                "day_of_week": dates.dt.dayofweek.astype(float),
                "is_weekend": (dates.dt.dayofweek >= 5).astype(float),
                # A missing per-load index is replaced by that day's market mean,
                # which is within ~0.025 of the truth instead of ~0.17 for a global median.
                "market_index": out["market_index"].fillna(pd.Series(market_daily, index=out.index)),
                "market_daily": market_daily,
                "market_roll7": cal.roll7_for(dates),
                "market_roll28": cal.roll28_for(dates),
                "trend_days": (dates - TREND_EPOCH).dt.days.astype(float),
            },
            index=out.index,
        )
        return features[self.feature_names_].astype(float)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.feature_names_, dtype=object)
