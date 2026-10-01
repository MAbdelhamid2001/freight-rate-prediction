"""JSON pricing endpoints."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from api.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    LoadRequest,
    ModelInfoResponse,
    PredictionResponse,
)
from freight_rate.inference import FreightRatePredictor, PredictionInputError, get_predictor

router = APIRouter(prefix="/api", tags=["predictions"])


def predictor_dependency() -> FreightRatePredictor:
    """Resolve the shared predictor; a missing model is a 503, not a 500."""
    try:
        return get_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Cheap readiness probe for the load balancer, Docker and the Streamlit app."""
    try:
        predictor = get_predictor()
    except FileNotFoundError:
        return HealthResponse(status="degraded", model_loaded=False)
    return HealthResponse(status="ok", model_loaded=True, model_version=predictor.card.get("trained_at_utc"))


@router.get("/model", response_model=ModelInfoResponse)
def model_info(predictor: FreightRatePredictor = Depends(predictor_dependency)) -> ModelInfoResponse:
    """What is deployed: estimator, features, training window and CV scores."""
    card = predictor.card
    return ModelInfoResponse(
        estimator=card.get("estimator"),
        feature_set=card.get("feature_set"),
        features=predictor.feature_names,
        training_period=card.get("training_period"),
        market_calendar_period=card.get("market_calendar_period"),
        cv_metrics={k: float(v) for k, v in card.get("cv_metrics", {}).items()},
        known_cities=sorted(predictor.cities),
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: LoadRequest, predictor: FreightRatePredictor = Depends(predictor_dependency)) -> PredictionResponse:
    """Price a single load."""
    try:
        result = predictor.predict_one(payload.to_record())
    except PredictionInputError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    return PredictionResponse(**result)


@router.post("/predict/batch", response_model=BatchResponse)
def predict_batch(payload: BatchRequest,
                  predictor: FreightRatePredictor = Depends(predictor_dependency)) -> BatchResponse:
    """Price up to 5,000 loads in one vectorised call."""
    frame = pd.DataFrame([load.to_record() for load in payload.loads])
    try:
        rates = predictor.predict_frame(frame)
    except PredictionInputError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    return BatchResponse(predictions=[round(float(r), 2) for r in rates], count=len(rates))
