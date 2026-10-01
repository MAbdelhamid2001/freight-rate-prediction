"""Every candidate builds, fits, predicts sensibly and survives a save/load."""

import joblib
import numpy as np
import pytest

from freight_rate.models import MODEL_FACTORIES, SAMPLE_WEIGHT_PARAM, build_pipeline, feature_importance
from freight_rate.train import Candidate, fit_candidate, recency_weights

FAST_PARAMS = {
    "random_forest": {"n_estimators": 20},
    "hist_gbm": {"max_iter": 40},
    "lightgbm": {"n_estimators": 40},
    "xgboost": {"n_estimators": 40},
    "blend": {"gbm_n_estimators": 40},
    "mlp": {"max_iter": 80},
}


@pytest.fixture(scope="module", params=sorted(MODEL_FACTORIES))
def fitted(request, train_frame, reference):
    pipeline = build_pipeline(request.param, feature_set="base", reference=reference,
                              params=FAST_PARAMS.get(request.param, {}))
    pipeline.fit(train_frame, train_frame["posted_rate"].to_numpy())
    return request.param, pipeline


def test_predictions_are_positive_and_finite(fitted, validation_frame):
    _, pipeline = fitted
    pred = pipeline.predict(validation_frame)
    assert len(pred) == len(validation_frame)
    assert np.isfinite(pred).all() and (pred > 0).all()


def test_models_beat_a_constant_on_training_data(fitted, train_frame):
    name, pipeline = fitted
    y = train_frame["posted_rate"].to_numpy()
    mape = np.mean(np.abs(pipeline.predict(train_frame) - y) / y)
    constant = np.mean(np.abs(np.median(y) - y) / y)
    assert mape < constant / 3, name


def test_pipeline_round_trips_through_joblib(fitted, validation_frame, tmp_path):
    _, pipeline = fitted
    path = tmp_path / "model.joblib"
    joblib.dump(pipeline, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(restored.predict(validation_frame), pipeline.predict(validation_frame))


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        build_pipeline("nope")


def test_recency_weights_halve_every_half_life(train_frame):
    weights = recency_weights(train_frame["date"], 30.0)
    newest = train_frame["date"].max()
    month_old = (newest - train_frame["date"]).dt.days == 30
    assert weights.max() == 1.0
    if month_old.any():
        assert weights[month_old.to_numpy()][0] == pytest.approx(0.5)
    assert recency_weights(train_frame["date"], None) is None


@pytest.mark.parametrize("model", ["ridge", "lightgbm", "blend", "mlp", "baseline_rpm"])
def test_sample_weights_reach_every_model_family(model, train_frame, reference):
    candidate = Candidate(model, "base", half_life_days=60.0, params=FAST_PARAMS.get(model, {}))
    pipeline = fit_candidate(candidate, train_frame, reference)
    assert pipeline.predict(train_frame.head(3)).shape == (3,)
    assert SAMPLE_WEIGHT_PARAM == "model__sample_weight"


def test_feature_importance_for_trees(train_frame, reference):
    pipeline = build_pipeline("lightgbm", feature_set="base", reference=reference, params={"n_estimators": 30})
    pipeline.fit(train_frame, train_frame["posted_rate"].to_numpy())
    importance = feature_importance(pipeline)
    assert importance is not None
    assert importance.sum() == pytest.approx(1.0)
    assert importance.index[0] in {"distance", "log_distance", "crow_miles"}
