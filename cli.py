"""CLI: demandcast {train,forecast,serve}."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence

from . import __version__
from .config import load_config
from .service import ForecastService
from .train import train


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="demandcast", description=__doc__)
    p.add_argument("--version", action="version", version=f"demandcast {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Backtest, calibrate and fit the final model")
    t.add_argument("--config", default=None)

    f = sub.add_parser("forecast", help="Forecast one series")
    f.add_argument("series_id", help="e.g. store_03")
    f.add_argument("-H", "--horizon", type=int, default=14)
    f.add_argument("--artifacts", default="artifacts")

    s = sub.add_parser("serve", help="Start the REST API (needs the [api] extra)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--artifacts", default="artifacts")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = _parser().parse_args(argv)
    if a.command == "train":
        meta = train(load_config(a.config))
        print(json.dumps({"leaderboard": meta["leaderboard"], "interval": meta["interval"]}, indent=2))
    elif a.command == "forecast":
        print(json.dumps(ForecastService(a.artifacts).forecast(a.series_id, a.horizon), indent=2))
    elif a.command == "serve":
        import uvicorn

        os.environ["DEMANDCAST_ARTIFACTS_DIR"] = a.artifacts
        uvicorn.run("demandcast.api:app", host=a.host, port=a.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
