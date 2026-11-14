"""Shared helpers for the ML scripts: data loading, the time split, and one plot style."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

SHOW = "--show" in sys.argv          # add --show to open each plot in a window as well
if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PLOTS = Path(__file__).resolve().parent / "plots"
SPLIT = "2016-07-01"   # train on arrivals before this date, test on arrivals from it onwards

# One palette for every chart. Series colours are used in this fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
NEUTRAL = "#c3c2b7"                                  # context marks (baselines, "everything else")
DIVERGING = ["#2a78d6", "#f0efec", "#e34948"]        # below average - average - above average


def load() -> pd.DataFrame:
    """train.csv with real dates. booking_id is never used as a feature: the file lists
    cancelled bookings before the ones that stayed, so row order gives the answer away."""
    return pd.read_csv(ROOT / "train.csv", parse_dates=["arrival_date", "booking_date"])


def time_split(df: pd.DataFrame, at: str = SPLIT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Past -> train, future -> test. A random split would let the model peek at the future."""
    at = pd.Timestamp(at)
    return df[df["arrival_date"] < at], df[df["arrival_date"] >= at]


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 12, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2, "xtick.major.size": 0, "ytick.major.size": 0,
        "lines.linewidth": 2, "legend.frameon": False, "figure.dpi": 100,
    })


def save(fig, name: str) -> Path:
    """Write the figure to ML/plots/<name>.png and say where it went."""
    PLOTS.mkdir(exist_ok=True)
    path = PLOTS / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  plot saved: {path.relative_to(ROOT)}")
    if SHOW:
        plt.show()
    plt.close(fig)
    return path
