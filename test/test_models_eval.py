import numpy as np

from demandcast.evaluate import (
    apply_offsets,
    conformal_offsets,
    cqr_scores,
    interval_metrics,
    point_metrics,
    seasonal_scale,
)
from demandcast.models import SeasonalMean, SeasonalNaive
from demandcast.train import fold_origins


def periodic(n_weeks=10):
    base = np.array([1.0, 2, 3, 4, 5, 6, 7])
    return np.log1p(np.tile(base, n_weeks)[None, :].repeat(2, axis=0))


def test_baselines_exact_on_periodic_series():
    Y = periodic()
    t0 = Y.shape[1]
    rows = np.arange(2)
    for model in (SeasonalNaive(), SeasonalMean()):
        f = model.forecast(Y, t0, 10, rows)
        expect = Y[:, :7][:, [h % 7 for h in range(10)]]
        assert np.allclose(f.mid, expect), model.name


def test_metrics_on_perfect_forecast():
    Y = np.expm1(periodic())
    m = point_metrics(Y[:, -7:], Y[:, -7:], seasonal_scale(Y))
    assert m["mase"] == 0 and m["wape"] == 0 and m["bias"] == 0


def test_seasonal_scale_of_linear_trend():
    y = np.arange(1.0, 101.0)[None, :]
    scale = seasonal_scale(y)
    assert abs(scale[0] - 7.0) < 1e-6


def test_interval_coverage():
    a = np.array([[1.0, 5.0, 10.0]])
    m = interval_metrics(a, a - 1, a + 1)
    assert m["coverage"] == 1.0
    assert interval_metrics(a, a + 1, a + 2)["coverage"] == 0.0


def test_conformal_widens_an_overconfident_band():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, (400, 14))
    lo, hi = np.full_like(y, -0.3), np.full_like(y, 0.3)  # far too narrow for N(0,1)
    off = conformal_offsets([cqr_scores(y, lo, hi)], alpha=0.2)
    assert (off > 0).all()
    y_new = rng.normal(0, 1, (4000, 14))
    lo_c = np.full_like(y_new, -0.3) - off[0]
    hi_c = np.full_like(y_new, 0.3) + off[0]
    cov = ((y_new >= lo_c) & (y_new <= hi_c)).mean()
    assert 0.75 < cov < 0.85
    lo_a, hi_a = apply_offsets(np.zeros((1, 14)), np.zeros((1, 14)), np.array([0.1, 0.2]))
    assert hi_a[0, 0] == 0.1 and hi_a[0, 13] == 0.2


def test_fold_origins_are_non_overlapping_and_end_at_last_day():
    o = fold_origins(1000, 28, 4)
    assert o == [888, 916, 944, 972]
    assert o[-1] + 28 == 1000
