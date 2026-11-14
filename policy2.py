"""Policy v2: bid prices from a historical demand forecast + expected-occupancy cap.

For hotel h and night d, with tau days until that night:
  future demand D(d, tau) = historical shown-guest nights in the same (month, weekday) group
                            whose lead-time-to-that-night < tau, per calendar night, x alpha
  free(d) = K * cap[d] - E[d]      (E = sum of calibrated P(show) of confirmed guests)
  bid lambda(d) = rate of the free(d)-th most valuable future guest (0 if demand < free)
Accept iff  rate >= mean_d lambda(d) * beta   and   max_d (E[d] + p) / cap[d] <= K
"""
import numpy as np
import pandas as pd

import model as M

DATES = ["arr_month", "arr_dow", "book_month", "book_dow"]


class Policy:
    params = dict(K=1.15, shift=-0.5, alpha=1.0, beta=1.0, pmin=0.0, label="actual")

    def __init__(self, commitments, capacity, train=None, **kw):
        self.p = {**self.params, **kw}
        if train is None:
            train = pd.read_csv("train.csv")
        train = train.copy()
        train["arrival_date"] = pd.to_datetime(train["arrival_date"])
        train["booking_date"] = pd.to_datetime(train["booking_date"])
        M.NUM = [c for c in M.NUM if c not in DATES]
        self.m = M.ShowModel().fit(train)
        self.start = min(c.index.min() for c in capacity.values())
        n = max(len(c) for c in capacity.values()) + 80
        self.days = pd.date_range(self.start, periods=n, freq="D")
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
        self._build_demand(train)
        self.growth = self._growth(train, commitments) if self.p.get("adapt", True) else {}
        self._bid_cache = {}

    def _growth(self, train, com):
        """Per-hotel demand scale: commitment room-nights vs what history (same month/dow, lead >= days
        from start) says should already be on the books. Uses raw bookings incl. cancellations."""
        out = {}
        horizon = min(self.p.get("growth_days", 180), len(self.days))

        def nights_table(df):
            nn = df.nights.to_numpy()
            rep = np.repeat(np.arange(len(df)), nn)
            k = np.concatenate([np.arange(x) for x in nn]) if len(nn) else np.array([], int)
            night = pd.DatetimeIndex(pd.to_datetime(df.arrival_date).to_numpy()[rep] + k.astype("timedelta64[D]"))
            return pd.DataFrame({"hotel": df.hotel.to_numpy()[rep], "night": night,
                                 "lead": df.days_ahead.to_numpy()[rep] + k})

        H = nights_table(train)
        H = H[H.night <= pd.to_datetime(train.arrival_date).max()]  # only fully observed nights
        C = nights_table(com) if len(com) else None
        for h in self.cap:
            obs = exp = 0.0
            Hh = H[H.hotel == h]
            Hh = Hh.assign(m=Hh.night.dt.month, dow=Hh.night.dt.dayofweek)
            ncal = Hh.groupby(["m", "dow"]).night.nunique()
            for d in range(horizon):
                day = self.days[d]
                key = (day.month, day.dayofweek)
                if key not in ncal.index:
                    continue
                sub = Hh[(Hh.m == key[0]) & (Hh.dow == key[1])]
                exp += (sub.lead >= d).sum() / ncal[key]
            if C is not None and horizon > 0:
                Ch = C[(C.hotel == h) & (C.night <= self.days[horizon - 1])]
                obs = len(Ch)
            out[h] = float(np.clip(obs / exp, 0.5, 2.5)) if exp > 0 else 1.0
        return out

    # ---------- demand model ----------
    def _build_demand(self, train):
        if self.p["label"] == "actual":
            t = train[train.cancelled == 0]
            w = np.ones(len(t))
        else:
            t = train
            w = self._calib(self.m.predict(train))
        nights = t.nights.to_numpy()
        rep = np.repeat(np.arange(len(t)), nights)
        k = np.concatenate([np.arange(x) for x in nights])
        night = t.arrival_date.to_numpy()[rep] + k.astype("timedelta64[D]")
        night = pd.DatetimeIndex(night)
        lead = t.days_ahead.to_numpy()[rep] + k
        g = pd.DataFrame({"hotel": t.hotel.to_numpy()[rep], "m": night.month, "dow": night.dayofweek,
                          "night": night, "lead": lead, "rate": t.nightly_rate.to_numpy()[rep], "w": w[rep]})
        self.demand = {}
        for key, grp in g.groupby(["hotel", "m", "dow"]):
            n_cal = grp.night.nunique()
            o = np.argsort(grp.lead.to_numpy())
            self.demand[key] = (grp.lead.to_numpy()[o], grp.rate.to_numpy()[o], grp.w.to_numpy()[o], n_cal)
        # fallback by weekday only (months never seen in training)
        for key, grp in g.groupby(["hotel", "dow"]):
            o = np.argsort(grp.lead.to_numpy())
            self.demand[key] = (grp.lead.to_numpy()[o], grp.rate.to_numpy()[o], grp.w.to_numpy()[o], grp.night.nunique())

    def _future(self, h, day, tau):
        key = (h, day.month, day.dayofweek)
        if key not in self.demand:
            key = (h, day.dayofweek)
        ck = (key, tau)
        if ck not in self._bid_cache:
            lead, rate, w, n_cal = self.demand[key]
            i = np.searchsorted(lead, tau, side="left")
            r, ww = rate[:i], w[:i]
            o = np.argsort(-r)
            self._bid_cache[ck] = (r[o], np.cumsum(ww[o]) / n_cal * self.p["alpha"] * self.growth.get(h, 1.0))
        return self._bid_cache[ck]

    def bid(self, h, d, tau):
        free = self.p["K"] * self.cap[h][d] - self.E[h][d]
        rates, cumd = self._future(h, self.days[d], tau)
        if len(cumd) == 0 or cumd[-1] <= free:
            return 0.0
        if free <= 0:
            return np.inf if len(rates) == 0 else rates[0]
        return rates[np.searchsorted(cumd, free)]

    # ---------- show prob ----------
    def _calib(self, p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        z = np.log(p / (1 - p)) + self.p["shift"]
        return 1 / (1 + np.exp(-z))

    def prepare(self, requests):
        self.cache = dict(zip(requests.booking_id, self._calib(self.m.predict(requests))))

    def pshow(self, b):
        if b["booking_id"] in self.cache:
            return self.cache[b["booking_id"]]
        return float(self._calib(np.array([self.m.predict_one(b)]))[0])

    # ---------- decision ----------
    def decide(self, b):
        h, n = b["hotel"], int(b["nights"])
        d0 = (pd.Timestamp(b["arrival_date"]) - self.start).days
        if n <= 0 or d0 < 0:
            return False
        p = self.pshow(b)
        sl = slice(d0, d0 + n)
        P = self.p
        if p < P["pmin"] or ((self.E[h][sl] + p) / self.cap[h][sl]).max() > P["K"]:
            return False
        lead = int(b["days_ahead"])
        lam = np.mean([self.bid(h, d0 + k, lead + k) for k in range(n)])
        if b["nightly_rate"] < P["beta"] * lam:
            return False
        self.E[h][sl] += p
        return True
