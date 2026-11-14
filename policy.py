"""Policy v1: expected-occupancy control with a load-dependent value threshold.

State: for every hotel-night, E[d] = sum of P(show) of confirmed guests in house that night.
Accept a request (p, rate, nights) iff
    load = max_d (E[d] + p) / cap[d]   <= K
    p * rate >= base * exp(gamma * load)          (pickier as nights fill up)
"""
import numpy as np
import pandas as pd

import model as M

DATES = ["arr_month", "arr_dow", "book_month", "book_dow"]


def load_train():
    df = pd.read_csv("train.csv")
    return df


class Policy:
    params = dict(K=1.0, base=20.0, gamma=1.0, shift=-0.5, pmin=0.0)

    def __init__(self, commitments, capacity, train=None, **kw):
        self.p = {**self.params, **kw}
        if train is None:
            train = load_train()
        M.NUM = [c for c in M.NUM if c not in DATES]
        self.m = M.ShowModel().fit(train)
        self.start = min(c.index.min() for c in capacity.values())
        n = max(len(c) for c in capacity.values()) + 80
        self.cap, self.E = {}, {}
        for h, c in capacity.items():
            cv = np.full(n, float(c.iloc[-1]))
            cv[: len(c)] = c.to_numpy()
            self.cap[h] = np.maximum(cv, 1.0)
            self.E[h] = np.zeros(n)
        self.cache = {}
        if len(commitments):
            pc = self._calib(self.m.predict(commitments))
            for (h, a, nn), pp in zip(commitments[["hotel", "arrival_date", "nights"]].itertuples(index=False), pc):
                d0 = (pd.Timestamp(a) - self.start).days
                self.E[h][d0 : d0 + nn] += pp

    def _calib(self, p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        z = np.log(p / (1 - p)) + self.p["shift"]
        return 1 / (1 + np.exp(-z))

    def prepare(self, requests: pd.DataFrame):
        """Speed-only hook used by sim.py: batch-predict rows (each prediction uses only its own row)."""
        self.cache = dict(zip(requests.booking_id, self._calib(self.m.predict(requests))))

    def pshow(self, b):
        if b["booking_id"] in self.cache:
            return self.cache[b["booking_id"]]
        return float(self._calib(np.array([self.m.predict_one(b)]))[0])

    def decide(self, b):
        h, n = b["hotel"], int(b["nights"])
        d0 = (pd.Timestamp(b["arrival_date"]) - self.start).days
        if n <= 0 or d0 < 0:
            return False
        p = self.pshow(b)
        sl = slice(d0, d0 + n)
        load = ((self.E[h][sl] + p) / self.cap[h][sl]).max()
        P = self.p
        ok = p >= P["pmin"] and load <= P["K"] and p * b["nightly_rate"] >= P["base"] * np.exp(P["gamma"] * load)
        if ok:
            self.E[h][sl] += p
        return bool(ok)
