"""Result plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .data import Dataset  # noqa: E402

COLORS = {"seasonal_naive": "#9aa5b1", "seasonal_mean4": "#e08a3c", "gbm": "#3b6fb6"}


def save_plots(res: dict[str, Any], ds: Dataset, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    board = res["leaderboard"]
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    names = [r["model"] for r in board]
    means = [r["mase"]["mean"] for r in board]
    ax.bar(names, means, yerr=[r["mase"]["std"] for r in board], capsize=4,
           color=[COLORS.get(n, "#555") for n in names])  # fmt: skip
    for i, v in enumerate(means):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_ylabel("MASE (lower is better; 1 = in-sample seasonal naive)")
    ax.set_title("Rolling-origin backtest")
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4))
    for n, v in res["horizon_mae"].items():
        ax.plot(range(1, len(v) + 1), v, label=n, color=COLORS.get(n), lw=2)
    ax.set_xlabel("Days ahead")
    ax.set_ylabel("MAE (units/day)")
    ax.set_title("Error by forecast horizon")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "error_by_horizon.png", dpi=150)
    plt.close(fig)

    last = res["last_fold"]
    if last:
        t0, H = last["t0"], last["actual"].shape[1]
        picks = np.argsort(ds.Y[:, :t0].mean(axis=1))[[0, len(ds.Y) // 3, 2 * len(ds.Y) // 3, -1]]
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=False)
        for ax, s in zip(axes.ravel(), picks, strict=False):
            hist = slice(t0 - 56, t0)
            ax.plot(range(t0 - 56, t0), ds.Y[s, hist], color="#555", lw=1, label="history")
            xs = range(t0, t0 + H)
            ax.plot(xs, last["actual"][s], color="#222", lw=1.5, label="actual")
            ax.plot(xs, np.expm1(last["gbm"].mid[s]), color=COLORS["gbm"], lw=2, label="forecast")
            ax.fill_between(xs, np.expm1(last["lo_cal"][s]), np.expm1(last["hi_cal"][s]),
                            color=COLORS["gbm"], alpha=0.2, label="80% interval")  # fmt: skip
            ax.set_title(ds.series_ids[s], fontsize=10)
            ax.grid(alpha=0.3)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("28-day forecasts on the last backtest fold")
        fig.tight_layout()
        fig.savefig(out / "forecast_examples.png", dpi=150)
        plt.close(fig)
