"""Rolling-origin backtest, interval calibration, and final model fit."""

from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import sklearn

from . import __version__
from .config import Config
from .data import Dataset, generate_dataset
from .evaluate import (
    apply_offsets,
    conformal_offsets,
    cqr_scores,
    interval_metrics,
    point_metrics,
    seasonal_scale,
)
from .models import GBMForecaster, SeasonalMean, SeasonalNaive
from .plots import save_plots

log = logging.getLogger(__name__)
BUNDLE_FILE = "bundle.joblib"
METADATA_FILE = "metadata.json"


@dataclass
class Bundle:
    gbm: GBMForecaster
    offsets: np.ndarray
    dataset: Dataset
    interval: tuple[float, float]


def fold_origins(n_obs: int, horizon: int, n_folds: int) -> list[int]:
    """Origins of non-overlapping folds; the last fold ends exactly at the last observation."""
    return [n_obs - horizon * (n_folds - i) for i in range(n_folds)]


def _agg(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))}
        for k in rows[0]
    }


def backtest(cfg: Config, ds: Dataset) -> dict[str, Any]:
    H, alpha = cfg.horizon, 1 - (cfg.interval[1] - cfg.interval[0])
    Ylog, cal = np.log1p(ds.Y), ds.calendar
    rows = np.arange(ds.Y.shape[0])
    baselines = [SeasonalNaive(), SeasonalMean()]
    names = [b.name for b in baselines] + ["gbm"]

    fold_metrics: dict[str, list[dict[str, float]]] = {n: [] for n in names}
    interval_raw: list[dict[str, float]] = []
    interval_cal: list[dict[str, float]] = []
    horizon_err = {n: np.zeros(H) for n in names}
    horizon_den = 0.0
    scores: list[np.ndarray] = []
    cal_scores: list[np.ndarray] = []
    offsets = np.zeros(1)
    last: dict[str, Any] = {}

    for i, t0 in enumerate(fold_origins(ds.n_obs, H, cfg.n_folds)):
        is_cal = i < cfg.calibration_folds
        if i == cfg.calibration_folds:  # all calibration folds done -> fix the offsets
            offsets = conformal_offsets(cal_scores, alpha)
        t_start = time.perf_counter()
        gbm = GBMForecaster(cfg.interval, cfg.seed, **cfg.gbm).fit(Ylog, ds.promo, cal, t0)
        actual = ds.Y[:, t0 : t0 + H]
        y_log = np.log1p(actual)
        preds = {b.name: b.forecast(Ylog, t0, H, rows) for b in baselines}
        preds["gbm"] = gbm.forecast(Ylog, t0, H, rows, cal, ds.promo)

        s = cqr_scores(y_log, preds["gbm"].lo, preds["gbm"].hi)
        scores.append(s)
        if is_cal:
            cal_scores.append(s)
            log.info("fold %d/%d (calibration) %.1fs", i + 1, cfg.n_folds, time.perf_counter() - t_start)
            continue

        scale = seasonal_scale(ds.Y[:, :t0])
        for n in names:
            fold_metrics[n].append(point_metrics(actual, np.expm1(preds[n].mid), scale))
            horizon_err[n] += np.abs(np.expm1(preds[n].mid) - actual).sum(axis=0)
        horizon_den += actual.shape[0]
        g = preds["gbm"]
        interval_raw.append(interval_metrics(actual, np.expm1(g.lo), np.expm1(g.hi)))
        lo_c, hi_c = apply_offsets(g.lo, g.hi, offsets)
        interval_cal.append(interval_metrics(actual, np.expm1(lo_c), np.expm1(hi_c)))
        last = {"t0": t0, "actual": actual, "gbm": g, "lo_cal": lo_c, "hi_cal": hi_c}
        log.info(
            "fold %d/%d MASE gbm=%.3f naive=%.3f | cov raw=%.2f cal=%.2f | %.1fs",
            i + 1, cfg.n_folds, fold_metrics["gbm"][-1]["mase"],
            fold_metrics["seasonal_naive"][-1]["mase"],
            interval_raw[-1]["coverage"], interval_cal[-1]["coverage"],
            time.perf_counter() - t_start,
        )  # fmt: skip

    leaderboard = sorted(
        ({"model": n, **{k: v for k, v in _agg(fold_metrics[n]).items()}} for n in names),
        key=lambda r: r["mase"]["mean"],
    )
    return {
        "leaderboard": leaderboard,
        "interval": {
            "nominal": cfg.interval[1] - cfg.interval[0],
            "raw": _agg(interval_raw),
            "calibrated": _agg(interval_cal),
        },
        "horizon_mae": {n: (horizon_err[n] / horizon_den).tolist() for n in names},
        "final_offsets": conformal_offsets(scores, alpha),  # all folds, used for serving
        "last_fold": last,
        "eval_folds": cfg.n_folds - cfg.calibration_folds,
    }


def train(cfg: Config) -> dict[str, Any]:
    started = time.perf_counter()
    ds = generate_dataset(cfg.data, cfg.seed)
    log.info("series=%d days=%d", ds.Y.shape[0], ds.n_obs)
    res = backtest(cfg, ds)

    Ylog = np.log1p(ds.Y)
    final = GBMForecaster(cfg.interval, cfg.seed, **cfg.gbm).fit(Ylog, ds.promo, ds.calendar, ds.n_obs)
    bundle = Bundle(final, res["final_offsets"], ds, cfg.interval)

    out = cfg.artifacts_dir
    out.mkdir(parents=True, exist_ok=True)
    save_plots(res, ds, out / "plots")
    joblib.dump(bundle, out / BUNDLE_FILE)
    meta = {
        "package_version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_seconds": round(time.perf_counter() - started, 1),
        "seed": cfg.seed,
        "horizon": cfg.horizon,
        "n_series": int(ds.Y.shape[0]),
        "n_days": ds.n_obs,
        "last_observed": str(ds.dates[ds.n_obs - 1].date()),
        "backtest_eval_folds": res["eval_folds"],
        "leaderboard": res["leaderboard"],
        "interval": res["interval"],
        "horizon_mae": res["horizon_mae"],
        "conformal_offsets_log": res["final_offsets"].tolist(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (out / METADATA_FILE).write_text(json.dumps(meta, indent=2))
    return meta
