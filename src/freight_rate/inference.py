"""The one inference entry point shared by the CLI, the API and the Streamlit app.

It loads the pipeline and its model card once, completes partial inputs
(a request may give a city instead of coordinates, or no market index), and
refuses inputs the model cannot price rather than returning a confident guess.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from freight_rate import config


class PredictionInputError(ValueError):
    """The request cannot be priced (unknown city without coordinates, ...)."""


@dataclass
class FreightRatePredictor:
    pipeline: object
    card: dict

    # -- construction --------------------------------------------------------
    @classmethod
    def load(cls, model_path: Path = config.MODEL_PATH, card_path: Path | None = None) -> "FreightRatePredictor":
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"model not found at {model_path}; run `uv run freight-train` first")
        card_path = card_path or Path(model_path).with_name(config.MODEL_CARD_PATH.name)
        card = json.loads(Path(card_path).read_text(encoding="utf-8")) if Path(card_path).is_file() else {}
        return cls(pipeline=joblib.load(model_path), card=card)

    # -- reference data exposed to the UIs -------------------------------------
    @property
    def _features(self):
        return self.pipeline.named_steps["features"]

    @property
    def cities(self) -> dict[str, tuple[float, float]]:
        return self._features.reference_.cities

    @property
    def market_calendar(self) -> pd.DataFrame:
        cal = self._features.reference_.calendar
        return pd.DataFrame({"market_daily": cal.daily, "market_roll7": cal.roll7, "market_roll28": cal.roll28})

    @property
    def feature_names(self) -> list[str]:
        return list(self._features.feature_names_)

    # -- prediction ------------------------------------------------------------
    def _complete(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for col in config.RAW_FEATURES:
            if col not in out:
                out[col] = np.nan
        out[config.DATE_COL] = pd.to_datetime(out[config.DATE_COL], errors="coerce")
        if out[config.DATE_COL].isna().any():
            raise PredictionInputError("every load needs a valid date")
        bad_equipment = set(out["equipment"].dropna()) - set(config.EQUIPMENT_TYPES)
        if bad_equipment or out["equipment"].isna().any():
            raise PredictionInputError(f"equipment must be one of {config.EQUIPMENT_TYPES}")
        for side in ("pickup", "delivery"):
            no_coords = out[f"{side}_lat"].isna() | out[f"{side}_lon"].isna()
            unknown = no_coords & ~out[side].astype(str).isin(self.cities)
            if unknown.any():
                names = sorted(out.loc[unknown, side].astype(str).unique())[:5]
                raise PredictionInputError(
                    f"{side} city {names} is not in the reference list; supply {side}_lat/{side}_lon"
                )
        return out

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """Predicted posted rate in dollars, one per row."""
        if frame.empty:
            return np.empty(0)
        return np.asarray(self.pipeline.predict(self._complete(frame)), dtype=float)

    def predict_one(self, load: dict) -> dict:
        """Price one load and report which inputs had to be filled in."""
        frame = self._complete(pd.DataFrame([load]))
        rate = float(self.pipeline.predict(frame)[0])
        features = self._features.transform(self.pipeline.named_steps["clean"].transform(frame)).iloc[0]
        distance = float(features["distance"]) if "distance" in features else float(frame["distance"].iloc[0])
        imputed = [col for col in ("distance", "weight", "market_index", "quote_signal",
                                   "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon")
                   if load.get(col) is None]
        return {
            "predicted_rate": round(rate, 2),
            "rate_per_mile": round(rate / distance, 4) if distance > 0 else None,
            "distance_used": round(distance, 1),
            "imputed_fields": imputed,
            "model_version": self.card.get("trained_at_utc", "unknown"),
        }


@lru_cache(maxsize=1)
def get_predictor() -> FreightRatePredictor:
    """Process-wide singleton: the API and the app load the model exactly once."""
    return FreightRatePredictor.load()
