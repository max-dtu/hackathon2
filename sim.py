"""Local re-implementation of the hidden evaluate.py, replayed on historical data.

Scenario for a window [start, end]:
  * train       = rows with arrival_date < start (what the Solution may learn from; all outcomes known)
  * commitments = arrivals in window booked before `start` (pre-confirmed, no `cancelled` column)
  * capacity    = rooms minus shown guests who arrived before `start` and are still in house
  * requests    = arrivals in window booked on/after `start`, replayed in booking-date order

Scoring (from solution.py docstring): walk through arrival days in order; seated guests keep
their room for their whole stay; arriving show-ups are seated most-valuable-first while rooms
are free, the rest (cheapest) are walked: 0 revenue and -2.5 x stay value.

Usage:  .venv/bin/python sim.py [module:Class ...]
"""
import importlib
import sys
import time

import numpy as np
import pandas as pd

ROOMS = {"Downtown Hotel": 124, "Seaside Hotel": 103}
WALK_COST = 2.5

SCENARIOS = {
    # name: (start, end)
    "2016_full": ("2016-01-01", "2016-12-31"),  # full-year shape like the real task, but only 6m train (stress test)
    "2016_H2": ("2016-07-01", "2016-12-31"),  # 12 months of training data
    "2016_Sep": ("2016-09-01", "2016-12-31"),  # 14 months
    "2016_Q4": ("2016-10-01", "2016-12-31"),  # 15 months
}
TUNE_ON = ["2016_H2", "2016_Sep", "2016_Q4"]


def load(path="train.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["arrival_date"] = pd.to_datetime(df["arrival_date"])
    df["booking_date"] = pd.to_datetime(df["booking_date"])
    return df


def make_scenario(df: pd.DataFrame, start: str, end: str):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    train = df[df.arrival_date < start].copy()
    window = df[(df.arrival_date >= start) & (df.arrival_date <= end)]
    commitments = window[window.booking_date < start].sort_values(["booking_date", "booking_id"])
    requests = window[window.booking_date >= start].sort_values(["booking_date", "booking_id"])

    days = pd.date_range(start, end, freq="D")
    capacity = {}
    prior = train[train.cancelled == 0]
    for hotel, rooms in ROOMS.items():
        p = prior[prior.hotel == hotel]
        dep = p.arrival_date + pd.to_timedelta(p.nights, unit="D")
        occ = np.zeros(len(days), dtype=int)
        for d in dep[dep > start]:
            occ[: (d - start).days] += 1  # nights start .. d-1
        capacity[hotel] = pd.Series(rooms - occ, index=days)
    return train, commitments, requests, capacity


def score(confirmed: pd.DataFrame, capacity: dict, start, end) -> dict:
    """confirmed: rows (with `cancelled`) that the hotel holds. Returns revenue breakdown."""
    out = {"revenue": 0.0, "penalty": 0.0, "seated": 0, "walked": 0, "shows": 0}
    shows = confirmed[confirmed.cancelled == 0]
    out["shows"] = len(shows)
    for hotel in ROOMS:
        cap = capacity[hotel]
        s = shows[shows.hotel == hotel]
        value = (s.nightly_rate * s.nights).to_numpy()
        arr = ((s.arrival_date - pd.Timestamp(start)).dt.days).to_numpy()
        nights = s.nights.to_numpy()
        n_days = len(cap)
        horizon = n_days + int(nights.max() if len(nights) else 0) + 1
        in_house = np.zeros(horizon, dtype=int)
        capv = np.concatenate([cap.to_numpy(), np.full(horizon - n_days, ROOMS[hotel])])
        order = np.lexsort((-value, arr))  # by arrival day, then most valuable first
        i = 0
        while i < len(order):
            day = arr[order[i]]
            j = i
            while j < len(order) and arr[order[j]] == day:
                j += 1
            free = capv[day] - in_house[day]
            for k, idx in enumerate(order[i:j]):
                if k < free:
                    out["revenue"] += value[idx]
                    out["seated"] += 1
                    in_house[day : day + nights[idx]] += 1
                else:
                    out["penalty"] += WALK_COST * value[idx]
                    out["walked"] += 1
            i = j
        out["empty_" + hotel.split()[0]] = int(np.clip(capv[:n_days] - in_house[:n_days], 0, None).sum())
        out["roomnights_" + hotel.split()[0]] = int(in_house[:n_days].sum())
    out["net"] = out["revenue"] - out["penalty"]
    return out


def run(solution_cls, df, start, end, verbose=True) -> dict:
    train, commitments, requests, capacity = make_scenario(df, start, end)
    t0 = time.time()
    sol = solution_cls(commitments.drop(columns=["cancelled"]), {h: c.copy() for h, c in capacity.items()}, train=train)
    t1 = time.time()
    req = requests.drop(columns=["cancelled"])
    if hasattr(sol, "prepare"):  # optional speed hook (row-wise predictions only)
        sol.prepare(req)
    accepted = np.zeros(len(req), dtype=bool)
    for i, row in enumerate(req.to_dict("records")):
        accepted[i] = bool(sol.decide(row))
    t2 = time.time()
    confirmed = pd.concat([commitments, requests[accepted]])
    res = score(confirmed, capacity, start, end)
    res.update(accepted=int(accepted.sum()), requests=len(req), fit_s=round(t1 - t0, 1), decide_s=round(t2 - t1, 1))
    if verbose:
        print(
            f"  {start}..{end}: net={res['net']:>12,.0f}  rev={res['revenue']:>12,.0f}  pen={res['penalty']:>10,.0f}  "
            f"acc={res['accepted']}/{res['requests']}  walked={res['walked']}  ({res['fit_s']}s fit, {res['decide_s']}s decide)"
        )
    return res


def load_cls(spec):
    mod, _, cls = spec.partition(":")
    return getattr(importlib.import_module(mod), cls or "Solution")


if __name__ == "__main__":
    df = load()
    specs = sys.argv[1:] or ["solution:Solution"]
    for spec in specs:
        print(spec)
        for name, (s, e) in SCENARIOS.items():
            run(load_cls(spec), df, s, e)
