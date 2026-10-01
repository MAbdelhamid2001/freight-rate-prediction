"""Candidate models, their preprocessing, and the pipeline factory.

Every candidate is the same four-step pipeline

    FrameCleaner -> FreightFeatures -> ColumnTransformer -> TransformedTargetRegressor(log)

so the object that is cross-validated is byte-for-byte the object that ships.
The ColumnTransformer is chosen per model family:

* **linear** (spline ridge): median imputation -> standardisation -> cubic
  B-splines for continuous columns, one-hot for weekday, binary flags passed
  through. This turns ridge into an additive model (a GAM).
* **neural** (MLP): median imputation -> standardisation for continuous
  columns, one-hot weekday, flags passed through. Gradient-based training
  needs every input on a comparable scale.
* **tree** (random forest, gradient boosting): median imputation only. Trees
  are invariant to monotone rescaling, so scaling would add nothing; the
  imputer is a safety net for a NaN reaching the API.

Why a log target: rates span $57-$25k and errors scale with the rate (a $50
miss matters on a 100-mile load, not on a 3,000-mile one). Modelling log(rate)
makes the problem roughly homoscedastic and turns multiplicative effects
(equipment premium, market level) into additive ones. The boosting models use
an absolute-error objective on that log scale, so they estimate the *median*
rate, which also keeps them robust to corrupted labels that slip past the
noise filter.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from xgboost import XGBRegressor

from freight_rate.config import RANDOM_STATE
from freight_rate.data import FrameCleaner
from freight_rate.features import FreightFeatures, ReferenceData

BINARY_COLUMNS = ("is_reefer", "is_flatbed", "is_weekend")
CATEGORICAL_COLUMNS = ("day_of_week",)


# Column selectors are module-level functions (not lambdas) so the fitted
# pipeline stays picklable with joblib.
def _binary_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c in BINARY_COLUMNS]


def _categorical_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c in CATEGORICAL_COLUMNS]


def _continuous_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in BINARY_COLUMNS + CATEGORICAL_COLUMNS]


def _all_columns(X: pd.DataFrame) -> list[str]:
    return list(X.columns)


def linear_preprocessor(n_knots: int = 8) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("splines", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("spline", SplineTransformer(n_knots=n_knots, degree=3)),
            ]), _continuous_columns),
            ("weekday", OneHotEncoder(handle_unknown="ignore", sparse_output=False), _categorical_columns),
            ("flags", SimpleImputer(strategy="most_frequent"), _binary_columns),
        ],
        remainder="drop",
    )


def neural_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("continuous", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), _continuous_columns),
            ("weekday", OneHotEncoder(handle_unknown="ignore", sparse_output=False), _categorical_columns),
            ("flags", SimpleImputer(strategy="most_frequent"), _binary_columns),
        ],
        remainder="drop",
    )


def _mlp(p: dict) -> MLPRegressor:
    """A feed-forward network on log(rate). Early stopping on a 10% slice of the
    training window guards against fitting that window's month-level noise."""
    p = dict(p)
    width = p.pop("width", 128)
    depth = p.pop("depth", 2)
    return MLPRegressor(**{
        # Tapering layers: 128 -> 64 for depth 2, 128 -> 64 -> 32 for depth 3.
        "hidden_layer_sizes": tuple(max(width // 2**i, 8) for i in range(depth)),
        "activation": "relu", "solver": "adam", "alpha": 1e-3,
        "learning_rate_init": 1e-3, "batch_size": 256, "max_iter": 300,
        "early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 15,
        "random_state": RANDOM_STATE, **p,
    })


def tree_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median"), _all_columns)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


class RatePerMileBaseline(BaseEstimator, RegressorMixin):
    """What a pricing analyst would do by hand: median $/mile per equipment
    type and distance band, times the distance. Every model must beat this."""

    def __init__(self, n_bands: int = 10):
        self.n_bands = n_bands

    def _keys(self, X: pd.DataFrame) -> pd.Series:
        equipment = X["is_reefer"].astype(int) * 1 + X["is_flatbed"].astype(int) * 2
        band = np.clip(np.searchsorted(self.edges_, X["distance"].to_numpy(), side="right") - 1,
                       0, self.n_bands - 1)
        return equipment.astype(str) + "_" + pd.Series(band, index=X.index).astype(str)

    def fit(self, X: pd.DataFrame, y, sample_weight=None) -> "RatePerMileBaseline":
        # Recency weights are accepted for interface parity but deliberately
        # ignored: the baseline is the plain, unweighted rule of thumb.
        self.edges_ = np.quantile(X["distance"], np.linspace(0, 1, self.n_bands + 1))
        rpm = pd.Series(np.asarray(y) / X["distance"].to_numpy(), index=X.index)
        self.table_ = rpm.groupby(self._keys(X)).median().to_dict()
        self.global_ = float(rpm.median())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        rpm = self._keys(X).map(self.table_).fillna(self.global_).to_numpy(dtype=float)
        return rpm * X["distance"].to_numpy()


def _lightgbm(p: dict) -> LGBMRegressor:
    return LGBMRegressor(**{
        "objective": "l1", "n_estimators": 1000, "learning_rate": 0.05,
        "num_leaves": 63, "min_child_samples": 20, "subsample": 0.8, "subsample_freq": 1,
        "colsample_bytree": 0.8, "reg_lambda": 1.0,
        "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1, **p,
    })


class LogSpaceBlend(BaseEstimator, RegressorMixin):
    """Weighted average of the spline GAM and LightGBM, fitted on log(rate).

    The two fail differently: the GAM is stable across time but misses lane x
    equipment interactions; the boosted trees capture interactions but react
    to every shift in the training window. Averaging in log space (a weighted
    geometric mean in dollars) keeps most of both. Each component carries its
    own ColumnTransformer, because they need different preprocessing.
    """

    def __init__(self, gam_weight: float = 0.5, gam_params: dict | None = None, gbm_params: dict | None = None):
        self.gam_weight = gam_weight
        self.gam_params = gam_params
        self.gbm_params = gbm_params

    def fit(self, X, y, sample_weight=None) -> "LogSpaceBlend":
        gam = dict(self.gam_params or {})
        self.gam_ = Pipeline([("prep", linear_preprocessor(gam.pop("n_knots", 8))), ("ridge", Ridge(**gam))])
        self.gam_.fit(X, y, ridge__sample_weight=sample_weight)
        self.gbm_ = Pipeline([("prep", tree_preprocessor()), ("gbm", _lightgbm(self.gbm_params or {}))])
        self.gbm_.fit(X, y, gbm__sample_weight=sample_weight)
        self.feature_importances_ = self.gbm_.named_steps["gbm"].feature_importances_
        return self

    def predict(self, X) -> np.ndarray:
        w = float(self.gam_weight)
        return w * self.gam_.predict(X) + (1.0 - w) * self.gbm_.predict(X)


def _log_target(regressor) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(regressor=regressor, func=np.log, inverse_func=np.exp, check_inverse=False)


def _split(params: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in params.items() if k.startswith(prefix)}


# Each entry: (preprocessing family, factory(params) -> estimator).
# Defaults are sensible starting points; Optuna tunes every family in train.py.
MODEL_FACTORIES: dict[str, tuple[str, Callable[[dict], object]]] = {
    "baseline_rpm": ("none", lambda p: RatePerMileBaseline(**p)),
    "ridge": ("linear", lambda p: _log_target(Ridge(**{"alpha": 1.0, **{k: v for k, v in p.items()
                                                                         if k != "n_knots"}}))),
    "random_forest": ("tree", lambda p: _log_target(RandomForestRegressor(**{
        "n_estimators": 300, "min_samples_leaf": 3, "max_features": 0.5,
        "n_jobs": -1, "random_state": RANDOM_STATE, **p,
    }))),
    "hist_gbm": ("tree", lambda p: _log_target(HistGradientBoostingRegressor(**{
        "loss": "absolute_error", "max_iter": 1000, "learning_rate": 0.05,
        "max_leaf_nodes": 63, "min_samples_leaf": 20, "l2_regularization": 0.1,
        "random_state": RANDOM_STATE, **p,
    }))),
    "lightgbm": ("tree", lambda p: _log_target(_lightgbm(p))),
    "xgboost": ("tree", lambda p: _log_target(XGBRegressor(**{
        "objective": "reg:absoluteerror", "n_estimators": 1000, "learning_rate": 0.05,
        "max_depth": 8, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_lambda": 1.0, "tree_method": "hist",
        "random_state": RANDOM_STATE, "n_jobs": -1, **p,
    }))),
    # The network also needs a standardised target: log(rate) sits around 6.5,
    # far from the ~0 an untrained output layer starts at.
    "mlp": ("neural", lambda p: _log_target(TransformedTargetRegressor(
        regressor=_mlp(p), transformer=StandardScaler()))),
    "blend": ("none", lambda p: _log_target(LogSpaceBlend(
        gam_weight=p.get("gam_weight", 0.5),
        gam_params={k: v for k, v in _split(p, "gam_").items() if k != "weight"},
        gbm_params=_split(p, "gbm_"),
    ))),
}

# All models are TransformedTargetRegressor or plain estimators taking
# sample_weight directly, so recency weights route the same way for each.
SAMPLE_WEIGHT_PARAM = "model__sample_weight"


def _preprocessor(family: str, params: dict):
    if family == "linear":
        return linear_preprocessor(params.get("n_knots", 8))
    if family == "tree":
        return tree_preprocessor()
    if family == "neural":
        return neural_preprocessor()
    return "passthrough"


def build_pipeline(
    model_name: str,
    *,
    feature_set: str = "full",
    reference: ReferenceData | None = None,
    params: dict | None = None,
) -> Pipeline:
    """Assemble cleaner -> features -> preprocessing -> model for one candidate."""
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"unknown model {model_name!r}; choose from {list(MODEL_FACTORIES)}")
    params = params or {}
    family, factory = MODEL_FACTORIES[model_name]
    return Pipeline(
        [
            ("clean", FrameCleaner()),
            ("features", FreightFeatures(feature_set=feature_set, reference=reference)),
            ("preprocess", _preprocessor(family, params)),
            ("model", factory(params)),
        ]
    )


def feature_importance(pipeline: Pipeline) -> pd.Series | None:
    """Normalised feature importances of a fitted pipeline, when the model has them."""
    names = list(pipeline.named_steps["features"].feature_names_)
    model = pipeline.named_steps["model"]
    inner = getattr(model, "regressor_", model)
    importances = getattr(inner, "feature_importances_", None)
    if importances is None or len(importances) != len(names):
        return None
    series = pd.Series(np.asarray(importances, dtype=float), index=names)
    return (series / series.sum()).sort_values(ascending=False)
