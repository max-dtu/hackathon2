"""Reference strategies for the simulator ladder. Same interface as solution.Solution, plus a
`history` keyword (bookings whose outcome is already known) supplied by simulate.py."""
import math

import numpy as np
import pandas as pd


class _Book:
    """Per-hotel, per-night tallies of confirmed bookings (commitments + accepted requests)."""

    def __init__(self, capacity):
        self.start = {h: s.index[0] for h, s in capacity.items()}
        self.cap = {h: s.to_numpy().astype(float) for h, s in capacity.items()}
        self.count = {h: np.zeros(len(c)) for h, c in self.cap.items()}
        self.mu = {h: np.zeros(len(c)) for h, c in self.cap.items()}   # expected shows
        self.var = {h: np.zeros(len(c)) for h, c in self.cap.items()}  # variance of shows

    def span(self, b):
        h = b["hotel"]
        lo = (b["arrival_date"] - self.start[h]).days
        return h, slice(max(lo, 0), min(lo + int(b["nights"]), len(self.cap[h])))

    def add(self, b, p):
        h, s = self.span(b)
        self.count[h][s] += 1
        self.mu[h][s] += p
        self.var[h][s] += p * (1 - p)


class _Base:
    def __init__(self, commitments, capacity, history=None, **kw):
        self.book = _Book(capacity)
        self.setup(history, **kw)
        for b in commitments.to_dict("records"):
            self.book.add(b, self.p(b))

    def setup(self, history, **kw):
        pass

    def p(self, b):
        return 1.0

    def decide(self, b):
        ok = self.accept(b)
        if ok:
            self.book.add(b, self.p(b))
        return ok


class AcceptAll(_Base):
    def accept(self, b):
        return True


class FixedOverbook(_Base):
    """Accept while confirmed bookings on every night of the stay are below capacity x factor."""

    def setup(self, history, factor=1.0):
        self.factor = factor

    def accept(self, b):
        h, s = self.book.span(b)
        return bool(np.all(self.book.count[h][s] < self.book.cap[h][s] * self.factor))


class AcceptUntilFull(FixedOverbook):
    def setup(self, history, factor=1.0):
        self.factor = 1.0


class EVRule(_Base):
    """Accept if  p * [(1-q) * V - q * 2.5 * V] >= 0, q = P(hotel already full on the stay's
    tightest night), from a normal approximation of shows among confirmed bookings.

    p_source: "average" -> historical show rate per hotel; "oracle" -> true outcome (cheat)."""

    def setup(self, history, p_source="average", outcomes=None, penalty=2.5):
        self.src, self.outcomes, self.penalty = p_source, outcomes, penalty
        self.avg = 1 - history.groupby("hotel").cancelled.mean()

    def p(self, b):
        if self.src == "oracle":
            return float(1 - self.outcomes[b["booking_id"]])
        return float(self.avg[b["hotel"]])

    def q(self, b):
        h, s = self.book.span(b)
        mu, var, cap = self.book.mu[h][s], self.book.var[h][s], self.book.cap[h][s]
        if len(cap) == 0:
            return 0.0
        z = cap - 0.5 - mu  # continuity correction: P(shows >= cap)
        sd = np.sqrt(var)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(sd > 0, 0.5 * np.vectorize(math.erfc)(z / (sd * math.sqrt(2))), (z < 0) * 1.0)
        return float(q.max())

    def accept(self, b):
        p, q = self.p(b), self.q(b)
        v = b["nightly_rate"] * b["nights"]
        return p * ((1 - q) * v - q * self.penalty * v) >= 0
