"""Replay a past year through Solution, the way evaluate.py will replay 2017.

    python backtest.py            # replay 2016 with a model trained on 2015 arrivals only
    python backtest.py --tune     # also search the policy settings (ALPHA and HOLD)

evaluate.py is not available until the end, so this is OUR reconstruction of the rules:
  - commitments = arrivals of the year booked before 1 January;
  - capacity    = rooms minus guests who arrived the year before and are still in house;
  - at check-in, if arrivals exceed free rooms, the cheapest are walked (--cheapest value|rate).
"""
import argparse
import time

import numpy as np
import pandas as pd

import solution
from solution import ROOMS, Solution
from demand_forecast import build_profile
from train_show_model import features, make_model


def load(path="train.csv"):
    df = pd.read_csv(path, parse_dates=["arrival_date", "booking_date"])
    df["value"] = df["nightly_rate"] * df["nights"]
    return df


def setup(df, year):
    """Build what evaluate.py hands to Solution, for `year`."""
    start = pd.Timestamp(year=year, month=1, day=1)
    nights = pd.date_range(start, pd.Timestamp(year=year, month=12, day=31))
    test = df[df["arrival_date"].dt.year == year].sort_values("booking_date", kind="stable")
    commitments = test[test["booking_date"] < start].drop(columns=["cancelled", "value"])
    requests = test[test["booking_date"] >= start]
    capacity = {}
    prev = df[(df["arrival_date"] < start) & (df["cancelled"] == 0)]
    for hotel, rooms in ROOMS.items():
        p = prev[prev["hotel"] == hotel]
        dep = p["arrival_date"] + pd.to_timedelta(p["nights"], unit="D")
        in_house = np.array([(dep > d).sum() for d in nights])
        capacity[hotel] = pd.Series(np.clip(rooms - in_house, 0, None), index=nights)
    return commitments, capacity, requests


def score(commitments_full, capacity, requests, accept, cheapest="value"):
    """Net revenue given the accept decisions (commitments are always in)."""
    booked = pd.concat([commitments_full, requests[accept]])
    shows = booked[booked["cancelled"] == 0]
    out = {}
    for hotel, cap in capacity.items():
        g = shows[shows["hotel"] == hotel]
        day0 = cap.index[0]
        free = np.concatenate([cap.to_numpy(), np.full(400, ROOMS[hotel])]).astype(int)
        occ = np.zeros(len(free), int)
        a = (g["arrival_date"] - day0).dt.days.to_numpy()
        n, v = g["nights"].to_numpy(), g["value"].to_numpy()
        key = v if cheapest == "value" else g["nightly_rate"].to_numpy()
        revenue = penalty = walks = 0
        for i in np.lexsort((-key, a)):            # by day, most expensive first
            if occ[a[i]] < free[a[i]]:
                occ[a[i]:a[i] + n[i]] += 1
                revenue += v[i]
            else:                                  # hotel full at check-in: walked
                penalty += 2.5 * v[i]
                walks += 1
        out[hotel] = dict(net=revenue - penalty, revenue=revenue, penalty=penalty,
                          walks=walks, guests=len(g))
    return out


def replay(df, year, policy="solution", cheapest="value"):
    commitments, capacity, requests = setup(df, year)
    commitments_full = df.loc[commitments.index]
    if policy == "accept all":
        accept = np.ones(len(requests), bool)
    elif policy == "no overbooking":               # confirmed bookings never exceed free rooms
        accept = np.zeros(len(requests), bool)
        day0 = capacity[next(iter(capacity))].index[0]
        booked = {h: np.zeros(len(c) + 400) for h, c in capacity.items()}
        free = {h: np.concatenate([c.to_numpy(), np.full(400, ROOMS[h])]) for h, c in capacity.items()}
        for h, a, n in zip(commitments["hotel"], (commitments["arrival_date"] - day0).dt.days, commitments["nights"]):
            booked[h][a:a + n] += 1
        for k, (h, a, n) in enumerate(zip(requests["hotel"], (requests["arrival_date"] - day0).dt.days, requests["nights"])):
            if np.all(booked[h][a:a + n] + 1 <= free[h][a:a + n]):
                booked[h][a:a + n] += 1
                accept[k] = True
    else:
        s = Solution(commitments, capacity)
        rows = requests.drop(columns=["cancelled", "value"]).to_dict("records")
        accept = np.array([s.decide(b) for b in rows], bool)
    res = score(commitments_full, capacity, requests, accept, cheapest)
    res["accept_rate"] = accept.mean()
    return res


def show(name, res):
    total = sum(r["net"] for h, r in res.items() if h in ROOMS)
    parts = " | ".join(f"{h.split()[0]} {r['net'] / 1e6:5.2f}M, {r['walks']:4d} walks of {r['guests']}"
                       for h, r in res.items() if h in ROOMS)
    print(f"{name:28s} NET {total / 1e6:6.3f}M | accepted {res['accept_rate']:4.0%} | {parts}")
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2016)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--cheapest", choices=["value", "rate"], default="value")
    args = ap.parse_args()

    df = load()
    past = df[df["arrival_date"].dt.year < args.year]          # no peeking at the replayed year
    Solution.model = make_model().fit(features(past), 1 - past["cancelled"])
    Solution.profile = build_profile(past)
    print(f"Replay of {args.year}; show-up model trained on {len(past)} earlier bookings; "
          f"walk the cheapest by {args.cheapest}\n")

    for policy in ["accept all", "no overbooking"]:
        show(policy, replay(df, args.year, policy, args.cheapest))
    t = time.time()
    show("our Solution", replay(df, args.year, "solution", args.cheapest))
    print(f"(Solution replay took {time.time() - t:.1f}s)")

    if args.tune:
        print("\nSearch of the settings (net revenue per hotel, M):")
        best = {}
        for hold in [0.0, 0.5, 0.75, 1.0, 1.25, 1.5]:
            for alpha in [1.0, 1.05, 1.1, 1.15]:
                solution.HOLD = hold
                solution.ALPHA = {h: alpha for h in ROOMS}
                res = replay(df, args.year, "solution", args.cheapest)
                line = f"  HOLD {hold:.2f}  ALPHA {alpha:.2f} :"
                for h in ROOMS:
                    line += f"  {h.split()[0]} {res[h]['net'] / 1e6:5.3f}"
                    if res[h]["net"] > best.get(h, (-1e18,))[0]:
                        best[h] = (res[h]["net"], alpha, hold)
                print(line)
        for h, (net, alpha, hold) in best.items():
            print(f"best for {h}: ALPHA {alpha}, HOLD {hold} -> {net / 1e6:.3f}M")
