"""JSON API contract over real HTTP."""

import pytest

import api.routers.predictions as predictions_router


def test_health_reports_a_loaded_model(client, predictor, monkeypatch):
    monkeypatch.setattr(predictions_router, "get_predictor", lambda: predictor)
    body = client.get("/api/health").json()
    assert body == {"status": "ok", "model_loaded": True, "model_version": "test"}


def test_health_degrades_without_a_model(client, monkeypatch):
    def missing():
        raise FileNotFoundError("no model")

    monkeypatch.setattr(predictions_router, "get_predictor", missing)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_predict_returns_a_price(client, valid_load):
    response = client.post("/api/predict", json=valid_load)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_rate"] > 0
    assert body["rate_per_mile"] == pytest.approx(body["predicted_rate"] / 360, rel=1e-3)
    assert "X-Process-Time-Ms" in response.headers


def test_minimal_request_is_enough(client):
    response = client.post("/api/predict", json={"pickup": "Atlanta", "delivery": "Nashville",
                                                 "equipment": "Reefer", "date": "2025-11-20"})
    assert response.status_code == 200
    assert "distance" in response.json()["imputed_fields"]


@pytest.mark.parametrize(
    "change",
    [
        {"equipment": "Tanker"},
        {"date": "not-a-date"},
        {"distance": -5},
        {"weight": 0},
        {"delivery": "Lexington"},
        {"pickup_lat": 38.0},  # latitude without longitude
    ],
)
def test_invalid_requests_get_422(client, valid_load, change):
    assert client.post("/api/predict", json={**valid_load, **change}).status_code == 422


def test_unknown_city_without_coordinates_gets_422(client, valid_load):
    response = client.post("/api/predict", json={**valid_load, "pickup": "Atlantis"})
    assert response.status_code == 422
    assert "Atlantis" in response.json()["detail"]


def test_batch_prices_every_load(client, valid_load):
    loads = [valid_load, {**valid_load, "equipment": "Reefer"}, {**valid_load, "date": "2025-12-25"}]
    body = client.post("/api/predict/batch", json={"loads": loads}).json()
    assert body["count"] == 3 and len(body["predictions"]) == 3
    assert all(p > 0 for p in body["predictions"])


def test_batch_rejects_an_empty_list(client):
    assert client.post("/api/predict/batch", json={"loads": []}).status_code == 422


def test_model_info(client):
    body = client.get("/api/model").json()
    assert body["estimator"] == "lightgbm"
    assert "Lexington" in body["known_cities"]
    assert "distance" in body["features"]


def test_root_redirects_to_docs(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/docs"
