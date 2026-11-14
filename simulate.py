"""Local stand-in for the organisers' evaluate.py.

Replays one historical period (default: all of 2016) exactly as the judges replay 2017:
  * the strategy sees only bookings whose arrival is before the period starts (history),
  * commitments = arrivals in the period that were booked before it started (forced accepts),
  * capacity    = rooms minus guests who arrived before the period and are still in house,
  * every other booking for the period is passed to decide() in booking-date order.

Scoring follows the rules in solution.py: at each check-in, if arriving guests who show up do
not fit, the cheapest (by stay value = nightly_rate x nights) are walked; a walked guest earns
nothing and costs 2.5 x their stay value. Seated guests keep their room.

Usage:  .venv/bin/python simulate.py [--start 2016-01-01 --end 2017-01-01]
"""
import argparse
import time

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

ROOMS = {"Downtown Hotel": 124, "Seaside Hotel": 103}
WALK_PENALTY = 2.5
HIDDEN_COLS = ["cancelled", "deposit"]  # never visible to decide(); deposit is dropped by the organisers


def load(path="train.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["arrival_date", "booking_date"])
    df["value"] = df["nightly_rate"] * df["nights"]
    return df


class Scenario:
    def __init__(self, df: pd.DataFrame, start: str, end: str):
        self.start, self.end = pd.Timestamp(start), pd.Timestamp(end)
        self.nights = pd.date_range(self.start, self.end - pd.Timedelta(days=1), freq="D")
        in_period = (df.arrival_date >= self.start) & (df.arrival_date < self.end)

        self.history = df[df.arrival_date < self.start].drop(columns=["deposit", "value"])
        self.commit = df[in_period & (df.booking_date < self.start)]
        self.requests = df[in_period & (df.booking_date >= self.start)].sort_values(
            ["booking_date", "booking_id"], kind="stable")

        # Rooms free each night: rooms minus earlier guests (who showed up) still in house.
        earlier = df[(df.arrival_date < self.start) & (df.cancelled == 0)]
        self.capacity = {}
        for hotel, rooms in ROOMS.items():
            occ = np.zeros(len(self.nights), dtype=int)
            for a, n in earlier.loc[earlier.hotel == hotel, ["arrival_date", "nights"]].itertuples(index=False):
                lo, hi = self._span(a, n)
                occ[lo:hi] += 1
            self.capacity[hotel] = pd.Series(np.maximum(rooms - occ, 0), index=self.nights)

    def _span(self, arrival, nights):
        """Night indices [lo, hi) of a stay, clipped to the period."""
        lo = (arrival - self.start).days
        return max(lo, 0), min(max(lo + nights, 0), len(self.nights))

    def public(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.drop(columns=HIDDEN_COLS + ["value"])


def run(strategy_cls, sc: Scenario, **kwargs) -> dict:
    """Replay the period through a strategy and score it. kwargs go to the strategy constructor."""
    t0 = time.time()
    strat = strategy_cls(sc.public(sc.commit), {h: s.copy() for h, s in sc.capacity.items()},
                         history=sc.history, **kwargs)
    pub = sc.public(sc.requests)
    accepted = [bool(strat.decide(rec)) for rec in pub.to_dict("records")]
    booked = pd.concat([sc.commit, sc.requests[np.array(accepted, dtype=bool)]])
    res = score(booked, sc)
    res.update(accepted=int(sum(accepted)), seconds=round(time.time() - t0, 1))
    return res


def score(booked: pd.DataFrame, sc: Scenario) -> dict:
    revenue = penalty = 0.0
    walked = 0
    for hotel, cap in sc.capacity.items():
        cap = cap.to_numpy()
        occ = np.zeros(len(cap), dtype=int)
        shows = booked[(booked.hotel == hotel) & (booked.cancelled == 0)]
        for arrival, group in shows.groupby("arrival_date"):
            day = (arrival - sc.start).days
            free = max(int(cap[day] - occ[day]), 0)
            group = group.sort_values("value", ascending=False, kind="stable")
            seated, out = group.iloc[:free], group.iloc[free:]
            revenue += seated.value.sum()
            penalty += WALK_PENALTY * out.value.sum()
            walked += len(out)
            for n in seated.nights:
                lo, hi = sc._span(arrival, n)
                occ[lo:hi] += 1
    return dict(net=round(revenue - penalty), revenue=round(revenue), penalty=round(penalty), walked=walked)


def hindsight_ceiling(sc: Scenario) -> dict:
    """Upper bound: knowing exactly who shows up, choose the best set of guests (LP relaxation).

    Requests that show: x in [0,1], earns value.  Commitments that show: s in [0,1] (seated) —
    earning value if seated and -2.5 x value if walked, i.e. 3.5*value*s - 2.5*value.
    One capacity constraint per night. No real policy can beat this number."""
    total = 0.0
    for hotel, cap in sc.capacity.items():
        c = sc.commit[(sc.commit.hotel == hotel) & (sc.commit.cancelled == 0)]
        r = sc.requests[(sc.requests.hotel == hotel) & (sc.requests.cancelled == 0)]
        guests = pd.concat([c, r])
        obj = np.concatenate([(1 + WALK_PENALTY) * c.value.to_numpy(), r.value.to_numpy()])
        rows, cols = [], []
        for j, (a, n) in enumerate(guests[["arrival_date", "nights"]].itertuples(index=False)):
            lo, hi = sc._span(a, n)
            rows.extend(range(lo, hi))
            cols.extend([j] * (hi - lo))
        A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(cap), len(guests)))
        lp = linprog(-obj, A_ub=A, b_ub=cap.to_numpy(), bounds=(0, 1), method="highs")
        total += -lp.fun - WALK_PENALTY * c.value.sum()
    return dict(net=round(total))


if __name__ == "__main__":
    from baselines import AcceptAll, AcceptUntilFull, FixedOverbook, EVRule

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2017-01-01")
    args = ap.parse_args()

    sc = Scenario(load(), args.start, args.end)
    print(f"Period {args.start}..{args.end}: {len(sc.history)} history rows, "
          f"{len(sc.commit)} commitments, {len(sc.requests)} requests\n")

    rows = [("Accept everything", run(AcceptAll, sc)),
            ("Accept until full (no overbooking)", run(AcceptUntilFull, sc))]
    for f in (1.1, 1.2, 1.3, 1.4, 1.5):
        rows.append((f"Fixed overbooking x{f}", run(FixedOverbook, sc, factor=f)))
    rows.append(("EV rule, p = historical average", run(EVRule, sc, p_source="average")))
    from solution import Solution
    rows.append(("EV rule, p = ML model (solution.py)", run(Solution, sc)))
    rows.append(("EV rule, p = true outcome (cheat)",
                 run(EVRule, sc, p_source="oracle", outcomes=pd.concat([sc.commit, sc.requests])
                     .set_index("booking_id").cancelled)))
    rows.append(("Perfect hindsight (LP ceiling)", hindsight_ceiling(sc)))

    base = rows[1][1]["net"]
    print(f"{'Strategy':38} {'Net':>12} {'vs no-OB':>10} {'Accepted':>9} {'Walked':>7} {'Penalty':>11}")
    for name, r in rows:
        print(f"{name:38} {r['net']:>12,} {r['net'] - base:>+10,} {r.get('accepted', ''):>9} "
              f"{r.get('walked', ''):>7} {r.get('penalty', 0):>11,}")
