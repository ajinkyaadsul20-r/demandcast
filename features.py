"""One feature function used for BOTH training and recursive inference, so there is no
train/serve skew. Features at day t only look at Y[:, :t] (no leakage) plus exogenous
inputs that are known in advance (calendar, promo plan)."""

from __future__ import annotations

import numpy as np

from .data import Calendar

FEATURES = [
    "sid", "dow", "dom", "doy_sin", "doy_cos", "holiday", "promo", "promo_sum7",
    "lag1", "lag7", "lag14", "lag28", "roll7", "roll28", "std7", "dow_mean4",
]  # fmt: skip
MIN_HISTORY = 28


def features_at(
    Ylog: np.ndarray, t: int, rows: np.ndarray, cal: Calendar, promo: np.ndarray
) -> np.ndarray:
    """Feature matrix (len(rows), n_features) for day index t. Ylog is log1p demand."""
    if t < MIN_HISTORY:
        raise ValueError(f"need at least {MIN_HISTORY} days of history, got t={t}")
    H = Ylog[rows]
    n = len(rows)
    w7, w28 = H[:, t - 7 : t], H[:, t - 28 : t]
    const = lambda v: np.full(n, v)  # noqa: E731
    return np.column_stack(
        [
            rows.astype(float),
            const(cal.dow[t]),
            const(cal.dom[t]),
            const(cal.doy_sin[t]),
            const(cal.doy_cos[t]),
            const(cal.holiday[t]),
            promo[rows, t],
            promo[rows, t - 6 : t + 1].sum(axis=1),
            H[:, t - 1],
            H[:, t - 7],
            H[:, t - 14],
            H[:, t - 28],
            w7.mean(axis=1),
            w28.mean(axis=1),
            w7.std(axis=1),
            (H[:, t - 7] + H[:, t - 14] + H[:, t - 21] + H[:, t - 28]) / 4,
        ]
    )
