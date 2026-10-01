"""Pydantic models describing the public shape of the pricing API."""

from __future__ import annotations

from datetime import date as Date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Equipment = Literal["Dry Van", "Reefer", "Flatbed"]


class LoadRequest(BaseModel):
    """One load to price.

    Only lane, equipment and date are required. Coordinates are looked up from
    the city name when omitted; distance, weight, market index and quote signal
    are filled the same way the training pipeline fills them.
    """

    pickup: str = Field(..., min_length=1, max_length=80, description="Pickup city")
    delivery: str = Field(..., min_length=1, max_length=80, description="Delivery city")
    equipment: Equipment
    date: Date = Field(..., description="Pickup date (YYYY-MM-DD)")
    distance: float | None = Field(None, gt=0, le=5_000, description="Road miles")
    weight: float | None = Field(None, gt=0, le=80_000, description="Weight in lb")
    pickup_lat: float | None = Field(None, ge=-90, le=90)
    pickup_lon: float | None = Field(None, ge=-180, le=180)
    delivery_lat: float | None = Field(None, ge=-90, le=90)
    delivery_lon: float | None = Field(None, ge=-180, le=180)
    market_index: float | None = Field(None, gt=0, le=5, description="Defaults to that day's market level")
    quote_signal: float | None = Field(None, gt=0, le=10)

    @model_validator(mode="after")
    def coordinates_come_in_pairs(self) -> "LoadRequest":
        for side in ("pickup", "delivery"):
            lat, lon = getattr(self, f"{side}_lat"), getattr(self, f"{side}_lon")
            if (lat is None) != (lon is None):
                raise ValueError(f"{side}_lat and {side}_lon must be given together")
        if self.pickup.strip().lower() == self.delivery.strip().lower():
            raise ValueError("pickup and delivery must be different cities")
        return self

    def to_record(self) -> dict:
        record = self.model_dump()
        record["date"] = self.date.isoformat()
        return record

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"pickup": "Lexington", "delivery": "Fort Wayne", "equipment": "Dry Van",
                 "date": "2025-12-15", "distance": 360, "weight": 32000}
            ]
        }
    }


class PredictionResponse(BaseModel):
    predicted_rate: float = Field(..., gt=0, description="Predicted posted rate in USD")
    rate_per_mile: float | None = Field(None, description="USD per mile")
    distance_used: float = Field(..., description="Distance the model used (given or estimated)")
    imputed_fields: list[str] = Field(default_factory=list, description="Inputs filled by the model")
    model_version: str


class BatchRequest(BaseModel):
    loads: list[LoadRequest] = Field(..., min_length=1, max_length=5_000)


class BatchResponse(BaseModel):
    predictions: list[float]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    estimator: str | None
    feature_set: str | None
    features: list[str]
    training_period: list[str] | None
    market_calendar_period: list[str] | None
    cv_metrics: dict[str, float]
    known_cities: list[str]
