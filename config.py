"""Typed configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    n_series: int = 20
    n_days: int = 1095
    start_date: str = "2023-01-01"


@dataclass(frozen=True)
class Config:
    seed: int = 42
    horizon: int = 28
    n_folds: int = 6
    calibration_folds: int = 2
    interval: tuple[float, float] = (0.1, 0.9)
    artifacts_dir: Path = Path("artifacts")
    data: DataConfig = field(default_factory=DataConfig)
    gbm: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        lo, hi = self.interval
        if not 0 < lo < 0.5 < hi < 1:
            raise ValueError("interval must satisfy 0 < lo < 0.5 < hi < 1")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if not 0 <= self.calibration_folds < self.n_folds:
            raise ValueError("need 0 <= calibration_folds < n_folds")
        need = 28 + 2 * self.horizon + self.horizon * self.n_folds
        if self.data.n_days < need:
            raise ValueError(f"n_days too small: need at least {need} for this backtest")
        if self.data.n_series < 1:
            raise ValueError("n_series must be >= 1")


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path or "configs/default.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    raw["data"] = DataConfig(**raw.get("data", {}))
    raw["interval"] = tuple(raw.get("interval", (0.1, 0.9)))
    if "artifacts_dir" in raw:
        raw["artifacts_dir"] = Path(raw["artifacts_dir"])
    cfg = Config(**raw)
    cfg.validate()
    return cfg
