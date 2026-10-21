"""FastAPI app. Run: uvicorn demandcast.api:app --host 0.0.0.0 --port 8000"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from . import __version__
from .service import ForecastService

ARTIFACTS_DIR = os.getenv("DEMANDCAST_ARTIFACTS_DIR", "artifacts")
_state: dict[str, ForecastService] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["svc"] = ForecastService(ARTIFACTS_DIR)
    yield
    _state.clear()


app = FastAPI(
    title="demandcast API",
    version=__version__,
    description="Daily demand forecasts with calibrated prediction intervals.",
    lifespan=lifespan,
)


def _svc() -> ForecastService:
    if "svc" not in _state:
        raise HTTPException(503, "Model not loaded")
    return _state["svc"]


@app.get("/health")
def health() -> dict[str, str]:
    _svc()
    return {"status": "ok", "version": __version__}


@app.get("/model")
def model_info() -> dict[str, Any]:
    return _svc().metadata


@app.get("/series")
def series() -> dict[str, list[str]]:
    return {"series_ids": _svc().series_ids}


@app.get("/forecast/{series_id}")
def forecast(series_id: str, horizon: int = Query(28, ge=1, le=28)) -> dict[str, Any]:
    try:
        return _svc().forecast(series_id, horizon)
    except KeyError as exc:
        raise HTTPException(404, str(exc.args[0])) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
