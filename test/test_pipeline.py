import json

import numpy as np
import pytest

from demandcast.service import ForecastService


def test_artifacts_and_leaderboard(trained):
    out, meta = trained
    for f in ("bundle.joblib", "metadata.json"):
        assert (out / f).exists()
    for p in ("model_comparison", "error_by_horizon", "forecast_examples"):
        assert (out / "plots" / f"{p}.png").exists()
    mase = {r["model"]: r["mase"]["mean"] for r in meta["leaderboard"]}
    assert mase["gbm"] < mase["seasonal_naive"]  # beats the naive baseline
    assert meta["backtest_eval_folds"] == 3
    assert json.loads((out / "metadata.json").read_text())["horizon"] == 14


def test_calibrated_coverage_is_reasonable(trained):
    _, meta = trained
    cov = meta["interval"]["calibrated"]["coverage"]["mean"]
    assert 0.6 < cov < 0.95


def test_forecast_shape_and_ordering(trained):
    out, _ = trained
    svc = ForecastService(out)
    res = svc.forecast("store_01", horizon=10)
    f = res["forecast"]
    assert len(f) == 10
    assert all(r["p_lo"] <= r["p50"] <= r["p_hi"] for r in f)
    assert all(r["p_lo"] >= 0 for r in f)
    assert f[0]["date"] > res["last_observed"]


def test_forecast_is_deterministic(trained):
    out, _ = trained
    svc = ForecastService(out)
    assert svc.forecast("store_02", 7) == svc.forecast("store_02", 7)


def test_validation_errors(trained):
    out, _ = trained
    svc = ForecastService(out)
    with pytest.raises(KeyError):
        svc.forecast("nope", 7)
    with pytest.raises(ValueError):
        svc.forecast("store_00", 0)
    with pytest.raises(ValueError):
        svc.forecast("store_00", 10_000)


def test_missing_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError):
        ForecastService(tmp_path)


def test_forecast_values_finite(trained):
    out, _ = trained
    svc = ForecastService(out)
    for sid in svc.series_ids:
        vals = [r["p50"] for r in svc.forecast(sid, 5)["forecast"]]
        assert np.isfinite(vals).all()
