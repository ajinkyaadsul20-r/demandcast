"""Serving logic shared by CLI and API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .evaluate import apply_offsets
from .train import BUNDLE_FILE, METADATA_FILE


class ForecastService:
    def __init__(self, artifacts_dir: str | Path = "artifacts") -> None:
        art = Path(artifacts_dir)
        if not (art / BUNDLE_FILE).exists():
            raise FileNotFoundError(f"No trained model in '{art}'. Run `demandcast train` first.")
        self.bundle = joblib.load(art / BUNDLE_FILE)
        self.metadata: dict[str, Any] = json.loads((art / METADATA_FILE).read_text())
        self.ds = self.bundle.dataset
        self.cal = self.ds.calendar
        self.Ylog = np.log1p(self.ds.Y)
        self.max_horizon = len(self.ds.dates) - self.ds.n_obs  # planned-promo calendar length

    @property
    def series_ids(self) -> list[str]:
        return self.ds.series_ids

    def forecast(self, series_id: str, horizon: int = 28) -> dict[str, Any]:
        if series_id not in self.ds.series_ids:
            raise KeyError(f"Unknown series_id '{series_id}'")
        if not 1 <= horizon <= self.max_horizon:
            raise ValueError(f"horizon must be between 1 and {self.max_horizon}")
        row = np.array([self.ds.series_ids.index(series_id)])
        f = self.bundle.gbm.forecast(
            self.Ylog, self.ds.n_obs, horizon, row, self.cal, self.ds.promo
        )
        lo, hi = apply_offsets(f.lo, f.hi, self.bundle.offsets)
        dates = self.ds.dates[self.ds.n_obs : self.ds.n_obs + horizon]
        return {
            "series_id": series_id,
            "last_observed": str(self.ds.dates[self.ds.n_obs - 1].date()),
            "interval": f"{round((self.bundle.interval[1] - self.bundle.interval[0]) * 100)}%",
            "forecast": [
                {
                    "date": str(d.date()),
                    "p_lo": round(float(np.expm1(lo[0, h])), 2),
                    "p50": round(float(np.expm1(f.mid[0, h])), 2),
                    "p_hi": round(float(np.expm1(hi[0, h])), 2),
                }
                for h, d in enumerate(dates)
            ],
        }
