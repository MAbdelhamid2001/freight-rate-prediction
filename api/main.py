"""FastAPI entry point for the freight-rate pricing service.

Run it with:  uv run uvicorn api.main:app --reload
Docs live at /docs (Swagger) and /redoc.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.routers import predictions
from freight_rate.inference import get_predictor

log = logging.getLogger("freight_rate.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model at startup so the first request does not pay the disk cost,
    # and so a missing artefact shows up in the logs immediately.
    try:
        predictor = get_predictor()
        log.info("model loaded: %s", predictor.card.get("candidate", "unknown"))
    except FileNotFoundError as exc:
        log.error("starting without a model: %s", exc)
    yield


app = FastAPI(
    title="Freight Rate API",
    description="Predicts the posted rate of a truckload from its lane, equipment, weight and date.",
    version="1.0.0",
    lifespan=lifespan,
)

# The Streamlit front end may run on another origin; the API is read-only.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
    return response


app.include_router(predictions.router)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")
