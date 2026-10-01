"""The shared predictor and the batch CLI that writes the deliverables."""

import numpy as np
import pandas as pd
import pytest

from freight_rate.inference import FreightRatePredictor, PredictionInputError
from freight_rate.predict import DECEMBER_COLUMNS, predict_december, predict_validation


def test_predict_one_reports_imputed_fields(predictor, valid_load):
    result = predictor.predict_one(valid_load)
    assert result["predicted_rate"] > 0
    assert result["distance_used"] == pytest.approx(360)
    assert {"pickup_lat", "market_index", "quote_signal"} <= set(result["imputed_fields"])
    assert "distance" not in result["imputed_fields"]


def test_predict_one_estimates_missing_distance(predictor, valid_load):
    result = predictor.predict_one({**valid_load, "distance": None})
    assert result["distance_used"] > 0
    assert "distance" in result["imputed_fields"]


def test_unknown_city_without_coordinates_is_refused(predictor, valid_load):
    with pytest.raises(PredictionInputError, match="not in the reference list"):
        predictor.predict_one({**valid_load, "pickup": "Atlantis"})


def test_unknown_city_with_coordinates_is_priced(predictor, valid_load):
    load = {**valid_load, "pickup": "Chicago", "pickup_lat": 41.88, "pickup_lon": -87.63}
    assert predictor.predict_one(load)["predicted_rate"] > 0


def test_bad_equipment_is_refused(predictor, valid_load):
    with pytest.raises(PredictionInputError, match="equipment"):
        predictor.predict_one({**valid_load, "equipment": "Tanker"})


def test_empty_frame_returns_empty(predictor):
    assert predictor.predict_frame(pd.DataFrame()).size == 0


def test_load_raises_a_helpful_error_without_a_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="freight-train"):
        FreightRatePredictor.load(tmp_path / "missing.joblib")


# ---------------------------------------------------------------------------
# Deliverables: the same rules score.py enforces
# ---------------------------------------------------------------------------
def test_validation_predictions_follow_the_template(predictor, validation_frame, tmp_path):
    validation_path = tmp_path / "validation.csv"
    template_path = tmp_path / "template.csv"
    validation_frame.assign(date=validation_frame.date.dt.strftime("%Y-%m-%d")).to_csv(validation_path, index=False)
    template = validation_frame[["load_id"]].sample(frac=1, random_state=0).assign(predicted_rate=np.nan)
    template.to_csv(template_path, index=False)

    out = predict_validation(predictor, validation_path, template_path)
    assert list(out.columns) == ["load_id", "predicted_rate"]
    assert out["load_id"].tolist() == template["load_id"].tolist()  # template order kept
    assert (out["predicted_rate"] > 0).all()


def test_validation_rejects_mismatched_template(predictor, validation_frame, tmp_path):
    validation_path = tmp_path / "validation.csv"
    template_path = tmp_path / "template.csv"
    validation_frame.to_csv(validation_path, index=False)
    pd.DataFrame({"load_id": ["TE-999999"], "predicted_rate": [np.nan]}).to_csv(template_path, index=False)
    with pytest.raises(ValueError, match="same load_id"):
        predict_validation(predictor, validation_path, template_path)


def test_december_file_keeps_its_columns(predictor, tmp_path):
    december = pd.DataFrame({
        "pickup": "Lexington", "delivery": "Fort Wayne", "distance": 360, "equipment": "Dry Van",
        "weight": 32000, "date": pd.date_range("2025-12-01", "2025-12-31").strftime("%Y-%m-%d"),
        "predicted_rate": np.nan,
    })
    path = tmp_path / "december.csv"
    december.to_csv(path, index=False)
    out = predict_december(predictor, path)
    assert list(out.columns) == DECEMBER_COLUMNS
    assert len(out) == 31 and (out["predicted_rate"] > 0).all()
