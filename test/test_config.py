from dataclasses import replace
from pathlib import Path

import pytest

from demandcast.config import Config, DataConfig, load_config


def test_default_config_loads():
    cfg = load_config(Path(__file__).parents[1] / "configs" / "default.yaml")
    assert cfg.horizon == 28 and cfg.interval == (0.1, 0.9)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("nope.yaml")


def test_validation():
    base = Config()
    base.validate()
    with pytest.raises(ValueError):
        replace(base, interval=(0.6, 0.9)).validate()
    with pytest.raises(ValueError):
        replace(base, calibration_folds=base.n_folds).validate()
    with pytest.raises(ValueError):
        replace(base, data=DataConfig(n_days=100)).validate()
