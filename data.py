"""Seeded synthetic retail demand: trend x weekly x yearly seasonality x promo x holidays,
with overdispersed count noise. Known structure lets us verify the model recovers it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DataConfig

# (month, day) -> multiplicative demand effect
HOLIDAY_EFFECT = {(1, 1): 0.7, (2, 14): 1.2, (7, 4): 1.2, (11, 28): 1.4, (12, 24): 1.3, (12, 25): 0.5}
FUTURE_DAYS = 28  # promo/holiday calendar extends this far past the last observation


@dataclass
class Calendar:
    dow: np.ndarray
    dom: np.ndarray
    doy_sin: np.ndarray
    doy_cos: np.ndarray
    holiday: np.ndarray

    @classmethod
    def from_dates(cls, dates: pd.DatetimeIndex) -> Calendar:
        doy = dates.dayofyear.to_numpy()
        hol = np.array([(d.month, d.day) in HOLIDAY_EFFECT for d in dates], dtype=float)
        return cls(
            dow=dates.dayofweek.to_numpy().astype(float),
            dom=dates.day.to_numpy().astype(float),
            doy_sin=np.sin(2 * np.pi * doy / 365.25),
            doy_cos=np.cos(2 * np.pi * doy / 365.25),
            holiday=hol,
        )


@dataclass
class Dataset:
    dates: pd.DatetimeIndex  # observed days + FUTURE_DAYS
    Y: np.ndarray  # (n_series, n_obs) observed demand
    promo: np.ndarray  # (n_series, n_obs + FUTURE_DAYS) promo flags, planned ahead
    series_ids: list[str]

    @property
    def n_obs(self) -> int:
        return self.Y.shape[1]

    @property
    def calendar(self) -> Calendar:
        return Calendar.from_dates(self.dates)

    def to_frame(self) -> pd.DataFrame:
        """Long format view of the observed data."""
        idx = self.dates[: self.n_obs]
        return pd.DataFrame(
            {
                "series_id": np.repeat(self.series_ids, self.n_obs),
                "date": np.tile(idx, len(self.series_ids)),
                "y": self.Y.ravel(),
                "promo": self.promo[:, : self.n_obs].ravel(),
            }
        )


def generate_dataset(cfg: DataConfig, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    n, T = cfg.n_series, cfg.n_days
    L = T + FUTURE_DAYS
    dates = pd.date_range(cfg.start_date, periods=L, freq="D")
    cal = Calendar.from_dates(dates)
    t = np.arange(L)
    hol_eff = np.ones(L)
    for i, d in enumerate(dates):
        hol_eff[i] = HOLIDAY_EFFECT.get((d.month, d.day), 1.0)

    promo = np.zeros((n, L))
    for s in range(n):
        day = int(rng.integers(0, 30))
        while day < L:
            length = int(rng.integers(3, 8))
            promo[s, day : day + length] = 1.0
            day += length + int(rng.integers(25, 60))

    Y = np.zeros((n, T))
    for s in range(n):
        base = rng.lognormal(mean=3.6, sigma=0.7)  # ~ 10 .. 300 units/day
        slope = rng.uniform(-0.0003, 0.0008)
        weekly = rng.uniform(0.7, 1.3, size=7)
        weekly[5:] *= rng.uniform(0.8, 1.5)
        weekly /= weekly.mean()
        amp, phase = rng.uniform(0.1, 0.3), rng.uniform(0, 2 * np.pi)
        lift = rng.uniform(0.2, 0.6)
        mu = (
            base
            * np.maximum(1 + slope * t, 0.2)
            * weekly[cal.dow.astype(int)]
            * (1 + amp * np.sin(2 * np.pi * t / 365.25 + phase))
            * (1 + lift * promo[s])
            * hol_eff
        )[:T]
        Y[s] = rng.poisson(rng.gamma(shape=15.0, scale=mu / 15.0))
    return Dataset(dates, Y.astype(float), promo, [f"store_{i:02d}" for i in range(n)])
