import numpy as np

from demandcast.config import DataConfig
from demandcast.data import FUTURE_DAYS, generate_dataset
from demandcast.features import FEATURES, MIN_HISTORY, features_at

CFG = DataConfig(n_series=5, n_days=200)


def test_generation_is_deterministic_and_valid():
    a, b = generate_dataset(CFG, 1), generate_dataset(CFG, 1)
    assert np.array_equal(a.Y, b.Y) and np.array_equal(a.promo, b.promo)
    assert a.Y.shape == (5, 200)
    assert a.promo.shape == (5, 200 + FUTURE_DAYS)
    assert len(a.dates) == 200 + FUTURE_DAYS
    assert (a.Y >= 0).all() and a.promo.max() == 1.0


def test_promo_raises_demand():
    ds = generate_dataset(DataConfig(n_series=10, n_days=700), 2)
    p = ds.promo[:, : ds.n_obs].astype(bool)
    assert ds.Y[p].mean() > ds.Y[~p].mean()


def test_features_do_not_leak_the_future():
    ds = generate_dataset(CFG, 3)
    Y, cal, rows, t = np.log1p(ds.Y), ds.calendar, np.arange(5), 100
    before = features_at(Y, t, rows, cal, ds.promo)
    Y2 = Y.copy()
    Y2[:, t:] += 123.0  # corrupt today and everything after
    after = features_at(Y2, t, rows, cal, ds.promo)
    assert np.array_equal(before, after)
    assert before.shape == (5, len(FEATURES))


def test_requires_history():
    ds = generate_dataset(CFG, 3)
    try:
        features_at(np.log1p(ds.Y), MIN_HISTORY - 1, np.arange(5), ds.calendar, ds.promo)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
