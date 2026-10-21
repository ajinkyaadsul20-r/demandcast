"""Shared fixtures: a small, fast end-to-end run."""

from __future__ import annotations

import pytest

from demandcast.config import Config, DataConfig
from demandcast.train import train

SMALL = Config(
    seed=3,
    horizon=14,
    n_folds=4,
    calibration_folds=1,
    data=DataConfig(n_series=6, n_days=420),
    gbm={"max_iter": 60, "min_samples_leaf": 20},
)


@pytest.fixture(scope="session")
def small_cfg():
    return SMALL


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    from dataclasses import replace

    out = tmp_path_factory.mktemp("artifacts")
    meta = train(replace(SMALL, artifacts_dir=out))
    return out, meta
