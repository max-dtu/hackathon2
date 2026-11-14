"""Backtest: replay a past window with the evaluation rules, using only earlier data to train.

    python ML/backtest.py                  compare all stay models on two 2016 windows
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
import solution  # noqa: E402

MODEL_KIND = getattr(solution, "MODEL_KIND", None)   # the model solution.py uses, if it names one

WINDOWS = [("2016-07-01", "2016-12-31"), ("2016-09-01", "2016-12-31")]


def net_revenue(stays: pd.DataFrame, capacity: dict[str, pd.Series]) -> dict:
    """The scoring rules. `stays` = confirmed bookings whose guest really showed up."""
    out = {"revenue": 0.0, "walk_penalty": 0.0, "walked": 0}
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="*", default=list(MODEL_KINDS))
    ap.add_argument("--segments", action="store_true")
    ap.add_argument("--explain", type=int, default=0)
    ap.add_argument("--no-late-rule", action="store_true", help="switch the late-booking rule off")
    args = ap.parse_args()

    df = load_history()
    df["arrival_date"] = pd.to_datetime(df["arrival_date"])
    df["booking_date"] = pd.to_datetime(df["booking_date"])
    if args.segments:
        return segments(df)

    closeness = {kind: [] for kind in args.models}   # share of the hindsight optimum, per window
    for start, end in WINDOWS:
        history, commitments, requests, capacity = split(df, start, end)
        shows = (requests["cancelled"] == 0).to_numpy()
        hidden = requests.drop(columns="cancelled")

        def score(mask):
            taken = pd.concat([commitments, requests[mask]])
            return net_revenue(taken[taken["cancelled"] == 0], capacity)

        best = score(hindsight(commitments, requests, capacity))["net"]
        print(f"\nWindow {start}..{end}: {len(commitments):,} commitments, {len(requests):,} requests")
        print(f"  {'policy':<16}{'net revenue':>13}{'vs hindsight':>14}{'accepted':>10}{'walked':>8}{'AUC':>7}")

        def line(name, mask, auc=""):
            r = score(mask)
            print(f"  {name:<16}{r['net']:>13,.0f}{r['net'] / best:>13.1%}{mask.mean():>10.1%}"
                  f"{r['walked']:>8,}{auc:>7}", flush=True)
            if name in closeness:
                closeness[name].append(r["net"] / best)

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
            if args.explain and kind == args.models[-1]:
                fresh = BookingPolicy(history, commitments.drop(columns="cancelled"), capacity, model_kind=kind,
                                  late_rule=not args.no_late_rule)
                step = max(len(records) // args.explain, 1)
                for r in records[step // 2::step][: args.explain]:
                    print("\n" + fresh.explain(r))
        line("hindsight", hindsight(commitments, requests, capacity))

    print("\nHow close to the best possible result (100% = knowing the future):")
    for kind, vals in sorted(closeness.items(), key=lambda kv: -np.mean(kv[1])):
        chosen = "   <- used in solution.py" if kind == MODEL_KIND else ""
        per_window = ", ".join(f"{v:.1%}" for v in vals)
        print(f"  {kind:<10}{np.mean(vals):>7.1%}   (per window: {per_window}){chosen}")


if __name__ == "__main__":
    main()
