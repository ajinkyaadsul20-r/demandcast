"""Forecasters. All return log1p-space arrays of shape (len(rows), horizon); baselines
return the same array for lo / mid / hi (point forecasts only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .data import Calendar
from .features import FEATURES, MIN_HISTORY, features_at


@dataclass
class Forecast:
    lo: np.ndarray
    mid: np.ndarray
    hi: np.ndarray


class SeasonalNaive:
    """Repeat the last observed week."""

    name = "seasonal_naive"

    def forecast(self, Ylog: np.ndarray, t0: int, H: int, rows: np.ndarray, **_: Any) -> Forecast:
        idx = np.array([t0 - 7 + (h % 7) for h in range(H)])
        out = Ylog[rows][:, idx]
        return Forecast(out, out, out)


class SeasonalMean:
    """Average of the same weekday over the last 4 weeks (a noise-robust seasonal naive)."""

    name = "seasonal_mean4"

    def forecast(self, Ylog: np.ndarray, t0: int, H: int, rows: np.ndarray, **_: Any) -> Forecast:
        cols = [
            Ylog[rows][:, [t0 - 7 + (h % 7) - 7 * j for h in range(H)]] for j in range(4)
        ]
        out = np.mean(cols, axis=0)
        return Forecast(out, out, out)


class GBMForecaster:
    """Global (all series share one model) recursive multi-step forecaster with quantile
    heads for prediction intervals. Median predictions are fed back as lags."""

    name = "gbm"

    def __init__(self, quantiles: tuple[float, float], seed: int = 0, **params: Any) -> None:
        self.quantiles = (quantiles[0], 0.5, quantiles[1])
        self.seed = seed
        self.params = {
            "max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31,
            "min_samples_leaf": 40, "l2_regularization": 1.0, **params,
        }  # fmt: skip
        self.models: list[HistGradientBoostingRegressor] = []

    def fit(self, Ylog: np.ndarray, promo: np.ndarray, cal: Calendar, t_end: int) -> GBMForecaster:
        """Train on all days in [MIN_HISTORY, t_end). Nothing at or after t_end is used."""
        rows = np.arange(Ylog.shape[0])
        X = np.vstack([features_at(Ylog, t, rows, cal, promo) for t in range(MIN_HISTORY, t_end)])
        y = np.concatenate([Ylog[:, t] for t in range(MIN_HISTORY, t_end)])
        self.models = [
            HistGradientBoostingRegressor(
                loss="quantile", quantile=q, categorical_features=[FEATURES.index("sid")],
                early_stopping=False, random_state=self.seed, **self.params,
            ).fit(X, y)
            for q in self.quantiles
        ]  # fmt: skip
        return self

    def forecast(
        self,
        Ylog: np.ndarray,
        t0: int,
        H: int,
        rows: np.ndarray,
        cal: Calendar,
        promo: np.ndarray,
    ) -> Forecast:
        """Forecast days t0 .. t0+H-1 using only Ylog[:, :t0]."""
        work = np.zeros((Ylog.shape[0], t0 + H))
        work[:, :t0] = Ylog[:, :t0]
        lo, mid, hi = (np.zeros((len(rows), H)) for _ in range(3))
        for h in range(H):
            t = t0 + h
            X = features_at(work, t, rows, cal, promo)
            q = np.sort(np.column_stack([m.predict(X) for m in self.models]), axis=1)
            q = np.maximum(q, 0.0)
            lo[:, h], mid[:, h], hi[:, h] = q[:, 0], q[:, 1], q[:, 2]
            work[rows, t] = mid[:, h]  # recursive: feed the median back
        return Forecast(lo, mid, hi)
