"""Late bookings and overbooking: what the policy does in each booking period.

Replays July-December 2016 (trained only on earlier data) with the late-booking rule off
and on, and shows:
  - how the booking periods differ (who books when, who shows up, what they pay),
  - how many requests are accepted in each period,
  - how far the hotels are overbooked night by night.

    python ML/late_bookings.py                    prints the tables, saves plots to ML/plots/
    python ML/late_bookings.py --model boosting   use another stay model
    python ML/late_bookings.py --show             also opens each plot in a window
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import common
from backtest import hindsight, net_revenue, split
from booking_model import MODEL_KINDS, PERIODS, ROOMS, BookingPolicy, booking_period
from common import plt

WINDOW = ("2016-07-01", "2016-12-31")
ORDER = [name for _, name in PERIODS][::-1]          # early -> last minute, the order of time


def plot_periods(df: pd.DataFrame):
    d = df.assign(period=df["days_ahead"].map(booking_period), stays=1 - df["cancelled"])
    t = d.groupby("period").agg(bookings=("stays", "size"), stay=("stays", "mean"),
                                rate=("nightly_rate", "mean")).loc[ORDER]
    t["sure"] = t["stay"] * t["rate"]
    panels = [("stay", "Guests who show up", lambda v: f"{v:.0%}"),
              ("rate", "Average nightly rate offered", lambda v: f"{v:.0f}"),
              ("sure", "Rate x chance to show up\n(money we can count on per night)", lambda v: f"{v:.0f}")]
    labels = [n.replace(" (", "\n(") for n in ORDER]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    for ax, (col, title, fmt) in zip(axes, panels):
        ax.bar(labels, t[col], color=common.SERIES[0], width=0.6)
        for i, v in enumerate(t[col]):
            ax.text(i, v + t[col].max() * 0.02, fmt(v), ha="center", color=common.INK_2, fontsize=9)
        ax.set_title(title, fontsize=10.5)
        ax.set_ylim(0, t[col].max() * 1.14)
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", labelsize=8.5)
    fig.suptitle("Late bookers do not pay more per night, but they almost always show up",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    return fig, t


def plot_overbooking(policy: BookingPolicy, taken: pd.DataFrame, capacity: dict):
    """Bookings confirmed per night against the rooms and against who really turned up."""
    hotels = list(ROOMS)
    fig, axes = plt.subplots(len(hotels), 1, figsize=(11, 6.6), sharex=True)
    for ax, hotel in zip(axes, hotels):
        cap = capacity[hotel]
        n = len(cap)
        shows = np.zeros(n + 400)
        g = taken[(taken["hotel"] == hotel) & (taken["cancelled"] == 0)]
        for d, k in zip((g["arrival_date"] - cap.index.min()).dt.days, g["nights"]):
            shows[d:d + k] += 1
        ax.plot(cap.index, policy.booked[hotel][:n], color=common.SERIES[0], lw=1.6, label="Bookings confirmed")
        ax.plot(cap.index, shows[:n], color=common.SERIES[1], lw=1.6, label="Guests who turned up")
        ax.plot(cap.index, cap.to_numpy(), color=common.INK_2, lw=1, ls="--", label="Rooms")
        ratio = policy.booked[hotel][:n].sum() / cap.to_numpy().sum()
        ax.set_title(f"{hotel}: {ratio:.1f} bookings confirmed per room, "
                     f"{shows[:n].sum() / cap.to_numpy().sum():.1f} guests turned up per room", fontsize=10.5)
        ax.set_ylabel("Per night")
        ax.set_ylim(bottom=0)
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="lower right", ncols=3)
    fig.suptitle("Overbooking: we confirm more bookings than rooms, because many guests cancel",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    return fig


def plot_acceptance(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    x = np.arange(len(table))
    labels = [n.replace(" (", "\n(") for n in table.index]
    for off, col, color, name in [(-0.2, "off", common.NEUTRAL, "Late rule off"),
                                  (0.2, "on", common.SERIES[0], "Late rule on")]:
        share = table[col] / table["requests"] * 100
        ax.bar(x + off, share, width=0.36, color=color, label=name)
        for xi, v in zip(x + off, share):
            ax.text(xi, v + 1, f"{v:.0f}%", ha="center", color=common.INK_2, fontsize=9)
    ax.set_xticks(x, labels)
    ax.set(ylabel="Requests accepted (%)", ylim=(0, 100),
           title="Share of requests accepted in each booking period")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncols=2)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="combined", choices=MODEL_KINDS)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    common.style()
    df = common.load()

    fig, t = plot_periods(df)
    print("Booking periods (all 2015-2016 bookings):")
    print(f"  {'period':<26}{'bookings':>9}{'show up':>9}{'avg rate':>10}{'rate x show-up':>16}")
    for name, r in t.iterrows():
        print(f"  {name:<26}{int(r.bookings):>9,}{r.stay:>9.0%}{r.rate:>10.0f}{r.sure:>16.0f}")
    common.save(fig, "late_periods")

    history, commitments, requests, capacity = split(df, *WINDOW)
    hidden = requests.drop(columns="cancelled")
    records = hidden.to_dict("records")
    period = requests["days_ahead"].map(booking_period).to_numpy()
    best = None
    table = pd.DataFrame({"requests": pd.Series(period).value_counts()}).loc[ORDER]
    print(f"\nReplay of {WINDOW[0]}..{WINDOW[1]} with the '{args.model}' stay model:")
    for key, late in (("off", False), ("on", True)):
        policy = BookingPolicy(history, commitments.drop(columns="cancelled"), capacity,
                               model_kind=args.model, late_rule=late)
        policy.preload(hidden)
        mask = np.array([policy.decide(r) for r in records])
        taken = pd.concat([commitments, requests[mask]])
        res = net_revenue(taken[taken["cancelled"] == 0], capacity)
        if best is None:
            ideal = hindsight(commitments, requests, capacity)
            ideal_taken = pd.concat([commitments, requests[ideal]])
            best = net_revenue(ideal_taken[ideal_taken["cancelled"] == 0], capacity)["net"]
        table[key] = pd.Series(period[mask]).value_counts().reindex(ORDER).fillna(0).astype(int)
        print(f"  late rule {key:<4} net revenue {res['net']:>10,.0f}   {res['net'] / best:.1%} of the best possible   "
              f"accepted {mask.mean():.1%}   walked {res['walked']:,}")

    print("\n  Accepted requests per booking period:")
    print(f"  {'period':<26}{'requests':>9}{'rule off':>10}{'rule on':>9}{'change':>8}")
    for name, r in table.iterrows():
        print(f"  {name:<26}{r.requests:>9,}{r.off:>10,}{r.on:>9,}{r.on - r.off:>+8,}")
    print()
    common.save(plot_acceptance(table), "late_acceptance")
    common.save(plot_overbooking(policy, taken, capacity), "late_overbooking")


if __name__ == "__main__":
    main()
