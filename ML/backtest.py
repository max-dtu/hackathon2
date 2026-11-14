"""Backtest: replay a past window with the evaluation rules, using only earlier data to train.

    python ML/backtest.py                  full report: every replay window, month by month
    python ML/backtest.py --models boosting   one model only
    python ML/backtest.py --segments          stay rates per segment (for the presentation)
    python ML/backtest.py --explain 5         print the reasoning for 5 example requests
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from booking_model import MODEL_KINDS, ROOMS, WALK_PENALTY, BookingPolicy, load_history

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root, for solution.py
import ML.solution as solution  # noqa: E402

MODEL_KIND = getattr(solution, "MODEL_KIND", None)   # the model solution.py uses, if it names one

# The first window covers every date we can replay (2015 is needed to train on). The other
# two train on 12 and 14 months of history, which is closest to the real 2017 run (18 months).
WINDOWS = [("2016-01-01", "2016-12-31"), ("2016-07-01", "2016-12-31"), ("2016-09-01", "2016-12-31")]
DEFAULT_MODELS = ["segments", "boosting", "combined"]


def net_revenue(stays: pd.DataFrame, capacity: dict[str, pd.Series]) -> dict:
    """The scoring rules. `stays` = confirmed bookings whose guest really showed up."""
    out = {"revenue": 0.0, "walk_penalty": 0.0, "walked": 0, "by_month": {}}
    for hotel, g in stays.groupby("hotel"):
        cap = capacity[hotel]
        origin = pd.Timestamp(cap.index.min())
        rooms = np.full(len(cap) + 400, float(ROOMS[hotel]))
        rooms[: len(cap)] = cap.to_numpy(float)
        inhouse = np.zeros(len(rooms))
        g = g.assign(day=(g["arrival_date"] - origin).dt.days, value=g["nightly_rate"] * g["nights"])
        for day, arr in g.sort_values("value", ascending=False).groupby("day", sort=True):
            free = int(max(rooms[day] - inhouse[day], 0))
            seated, walked = arr.iloc[:free], arr.iloc[free:]   # cheapest arrivals are walked
            for d, n in zip(seated["day"], seated["nights"]):
                inhouse[d:d + n] += 1
            out["revenue"] += seated["value"].sum()
            out["walk_penalty"] += WALK_PENALTY * walked["value"].sum()
            out["walked"] += len(walked)
            month = (origin + pd.Timedelta(days=int(day))).strftime("%Y-%m")
            out["by_month"][month] = (out["by_month"].get(month, 0.0) + seated["value"].sum()
                                      - WALK_PENALTY * walked["value"].sum())
    out["net"] = out["revenue"] - out["walk_penalty"]
    return out


def split(df: pd.DataFrame, start: str, end: str):
    """Cut the data the way the real evaluation does, for an earlier window."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    history = df[df["arrival_date"] < start]
    window = df[(df["arrival_date"] >= start) & (df["arrival_date"] <= end)]
    commitments = window[window["booking_date"] < start]
    requests = window[window["booking_date"] >= start].sort_values(["booking_date", "booking_id"])
    capacity = {}
    for hotel, rooms in ROOMS.items():
        cap = pd.Series(float(rooms), index=pd.date_range(start, end))
        h = history[(history["hotel"] == hotel) & (history["cancelled"] == 0)]
        for a, n in zip(h["arrival_date"], h["nights"]):
            if a + pd.Timedelta(days=int(n)) > start:
                cap[(cap.index >= a) & (cap.index < a + pd.Timedelta(days=int(n)))] -= 1
        capacity[hotel] = cap.clip(lower=0)
    return history, commitments, requests, capacity


def hindsight(commitments, requests, capacity) -> np.ndarray:
    """Benchmark that knows the future: seats showing guests, highest rate first, when free."""
    mask = np.zeros(len(requests), bool)
    pos = {b: i for i, b in enumerate(requests["booking_id"])}
    for hotel, rooms in ROOMS.items():
        cap = capacity[hotel]
        origin = cap.index.min()
        free = np.full(len(cap) + 400, float(rooms))
        free[: len(cap)] = cap.to_numpy(float)
        c = commitments[(commitments["hotel"] == hotel) & (commitments["cancelled"] == 0)]
        for a, n in zip((c["arrival_date"] - origin).dt.days, c["nights"]):
            free[a:a + n] -= 1
        r = requests[(requests["hotel"] == hotel) & (requests["cancelled"] == 0)]
        r = r.assign(day=(r["arrival_date"] - origin).dt.days).sort_values("nightly_rate", ascending=False)
        for b, d, n in zip(r["booking_id"], r["day"], r["nights"]):
            if (free[d:d + n] >= 1).all():
                free[d:d + n] -= 1
                mask[pos[b]] = True
    return mask


def segments(df: pd.DataFrame) -> None:
    d = df.assign(stays=1 - df["cancelled"],
                  lead_time=pd.cut(df["days_ahead"], [-1, 7, 30, 90, 180, 2000],
                                   labels=["0-7 days", "8-30", "31-90", "91-180", "180+"]))
    for col in ["deposit", "sales_segment", "lead_time", "guest_type", "hotel"]:
        t = d.groupby(col, observed=True).agg(bookings=("stays", "size"), stay_rate=("stays", "mean"),
                                              avg_rate=("nightly_rate", "mean"))
        print(t.sort_values("stay_rate", ascending=False).round(2).to_string(), "\n")


def report_months(monthly: dict, kind: str, tag: str) -> None:
    """Net revenue for every arrival month: rejecting everything, our model, and hindsight."""
    t = pd.DataFrame(monthly).fillna(0.0).sort_index()
    t = t[[c for c in ("Reject all", "Our model", "Best possible") if c in t]]
    print(f"\n  Net revenue by arrival month ('{kind}' model):")
    print(f"  {'month':<9}{'reject all':>12}{'our model':>12}{'best possible':>15}{'how close':>11}")
    for month, r in t.iterrows():
        print(f"  {month:<9}{r['Reject all']:>12,.0f}{r['Our model']:>12,.0f}{r['Best possible']:>15,.0f}"
              f"{r['Our model'] / r['Best possible']:>11.0%}")
    tot = t.sum()
    print(f"  {'total':<9}{tot['Reject all']:>12,.0f}{tot['Our model']:>12,.0f}{tot['Best possible']:>15,.0f}"
          f"{tot['Our model'] / tot['Best possible']:>11.0%}")

    import common
    common.style()
    fig, ax = common.plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(t))
    for off, col, color in ((-0.27, "Reject all", common.NEUTRAL), (0, "Our model", common.SERIES[0]),
                            (0.27, "Best possible", common.INK_2)):
        ax.bar(x + off, t[col] / 1e3, width=0.25, color=color, label=col)
    ax.set_xticks(x, [pd.Timestamp(m).strftime("%b") for m in t.index])
    ax.grid(axis="x", visible=False)
    ax.set(ylabel="Net revenue (thousands)",
           title=f"Net revenue by arrival month: {tot['Our model'] / 1e6:.2f}M of a possible "
                 f"{tot['Best possible'] / 1e6:.2f}M ({tot['Our model'] / tot['Best possible']:.0%})")
    ax.legend(loc="upper left", ncols=3)
    common.save(fig, f"backtest_monthly_{tag}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS, choices=MODEL_KINDS,
                    help="stay models to compare (the last one gets the month-by-month report)")
    ap.add_argument("--segments", action="store_true")
    ap.add_argument("--explain", type=int, default=0)
    ap.add_argument("--window", nargs=2, metavar=("START", "END"),
                    help="replay one custom window, e.g. --window 2016-01-01 2016-12-31")
    ap.add_argument("--no-late-rule", action="store_true", help="switch the late-booking rule off")
    args = ap.parse_args()

    df = load_history()
    df["arrival_date"] = pd.to_datetime(df["arrival_date"])
    df["booking_date"] = pd.to_datetime(df["booking_date"])
    if args.segments:
        return segments(df)

    closeness = {kind: [] for kind in args.models}   # share of the hindsight optimum, per window
    for start, end in ([tuple(args.window)] if args.window else WINDOWS):
        history, commitments, requests, capacity = split(df, start, end)
        shows = (requests["cancelled"] == 0).to_numpy()
        hidden = requests.drop(columns="cancelled")

        def score(mask):
            taken = pd.concat([commitments, requests[mask]])
            return net_revenue(taken[taken["cancelled"] == 0], capacity)

        best = score(hindsight(commitments, requests, capacity))["net"]
        print(f"\nWindow {start}..{end}: {len(commitments):,} commitments, {len(requests):,} requests")
        print(f"  {'policy':<16}{'net revenue':>13}{'vs hindsight':>14}{'room revenue':>14}{'walk cost':>12}"
              f"{'accepted':>10}{'walked':>8}{'AUC':>7}")

        def line(name, mask, auc=""):
            r = score(mask)
            print(f"  {name:<16}{r['net']:>13,.0f}{r['net'] / best:>13.1%}{r['revenue']:>14,.0f}"
                  f"{r['walk_penalty']:>12,.0f}{mask.mean():>10.1%}{r['walked']:>8,}{auc:>7}", flush=True)
            if name in closeness:
                closeness[name].append(r["net"] / best)

        monthly = {}

        def keep(name, mask):
            monthly[name] = score(mask)["by_month"]

        keep("Reject all", np.zeros(len(requests), bool))
        line("reject all", np.zeros(len(requests), bool))
        line("accept all", np.ones(len(requests), bool))
        for kind in args.models:
            t0 = time.time()
            policy = BookingPolicy(history, commitments.drop(columns="cancelled"), capacity, model_kind=kind,
                                  late_rule=not args.no_late_rule)
            policy.preload(hidden)
            p = np.array([policy._known[b] for b in requests["booking_id"]])
            records = hidden.to_dict("records")
            mask = np.array([policy.decide(r) for r in records])
            line(kind, mask, f"{roc_auc_score(shows, p):.3f}")
            if kind == args.models[-1]:
                keep("Our model", mask)
            if args.explain and kind == args.models[-1]:
                fresh = BookingPolicy(history, commitments.drop(columns="cancelled"), capacity, model_kind=kind,
                                  late_rule=not args.no_late_rule)
                step = max(len(records) // args.explain, 1)
                for r in records[step // 2::step][: args.explain]:
                    print("\n" + fresh.explain(r))
        line("hindsight", hindsight(commitments, requests, capacity))
        keep("Best possible", hindsight(commitments, requests, capacity))
        if (start, end) == WINDOWS[0] or args.window:
            report_months(monthly, args.models[-1], f"{start[:7]}_{end[:7]}")

    print("\nHow close to the best possible result (100% = knowing the future):")
    for kind, vals in sorted(closeness.items(), key=lambda kv: -np.mean(kv[1])):
        chosen = "   <- used in solution.py" if kind == MODEL_KIND else ""
        per_window = ", ".join(f"{v:.1%}" for v in vals)
        print(f"  {kind:<10}{np.mean(vals):>7.1%}   (per window: {per_window}){chosen}")


if __name__ == "__main__":
    main()
