"""Forecast accuracy metrics and split-conformal interval calibration."""

from __future__ import annotations

import numpy as np


def seasonal_scale(Y_train: np.ndarray, season: int = 7) -> np.ndarray:
    """Per-series in-sample MAE of the seasonal-naive forecast (the MASE denominator)."""
    return np.abs(Y_train[:, season:] - Y_train[:, :-season]).mean(axis=1) + 1e-9


def point_metrics(actual: np.ndarray, pred: np.ndarray, scale: np.ndarray) -> dict[str, float]:
    err = pred - actual
    total = actual.sum() + 1e-9
    return {
        "mase": float(np.mean(np.abs(err).mean(axis=1) / scale)),
        "wape": float(np.abs(err).sum() / total),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "smape": float(np.mean(2 * np.abs(err) / (np.abs(actual) + np.abs(pred) + 1e-9))),
        "bias": float(err.sum() / total),
    }


def interval_metrics(actual: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, float]:
    return {
        "coverage": float(((actual >= lo) & (actual <= hi)).mean()),
        "width_ratio": float((hi - lo).mean() / (actual.mean() + 1e-9)),
    }


def cqr_scores(y_log: np.ndarray, lo_log: np.ndarray, hi_log: np.ndarray) -> np.ndarray:
    """Conformalised-quantile-regression nonconformity scores (Romano et al., 2019)."""
    return np.maximum(lo_log - y_log, y_log - hi_log)


def conformal_offsets(scores: list[np.ndarray], alpha: float, bucket: int = 7) -> np.ndarray:
    """Per-week-of-horizon offsets (log space) that widen/narrow the quantile band so that
    coverage reaches 1 - alpha. `scores`: list of (n_series, H) arrays from calibration folds."""
    if not scores:
        return np.zeros(1)
    S = np.vstack(scores)
    H = S.shape[1]
    out = []
    for start in range(0, H, bucket):
        s = S[:, start : start + bucket].ravel()
        level = min(1.0, (1 - alpha) * (1 + 1 / len(s)))
        out.append(float(np.quantile(s, level, method="higher")))
    return np.array(out)


def apply_offsets(
    lo_log: np.ndarray, hi_log: np.ndarray, offsets: np.ndarray, bucket: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    H = lo_log.shape[1]
    off = np.array([offsets[min(h // bucket, len(offsets) - 1)] for h in range(H)])
    return np.maximum(lo_log - off, 0.0), hi_log + off
