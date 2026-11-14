"""Policy v3 = v2 (demand-forecast bid prices) + self-calibration + variance-aware overbooking.

* Self-calibration: inside __init__, hold out the last `hold_months` of training arrivals, replay them as
  a "request stream" (booked after the cut-off), fit a Platt layer  logit(p') = a*logit(p) + b  on it.
  This fixes the systematic optimism of the raw model on in-stream bookings, with no hand-tuned shift.
* Overbooking: per hotel-night keep mean E and variance V of the number of shows; accept only if
      E + p + z * sqrt(V + p(1-p)) <= K * cap     on every night of the stay.
* Bid price: free(d) = K*cap - E (in expected shows); lambda(d) = rate of the free(d)-th most valuable
  future guest expected to show (history of same hotel/month/weekday, lead < tau), x alpha.
  Accept iff rate >= beta * mean lambda.
"""
import numpy as np
import pandas as pd

import model as M


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _sig(z):
    return 1 / (1 + np.exp(-z))


class Policy:
    params = dict(K=1.0, z=0.5, alpha=2.0, beta=1.0, hold_months=4, selfcal=True, shift=0.0, growth=True,
                  growth_days=180)

    def __init__(self, commitments, capacity, train=None, **kw):
        self.p = {**self.params, **kw}
        if train is None:
            train = pd.read_csv("train.csv")
        train = train.copy()
        train["arrival_date"] = pd.to_datetime(train["arrival_date"])
        train["booking_date"] = pd.to_datetime(train["booking_date"])

        self.a, self.b = 1.0, 0.0
        if self.p["selfcal"]:
            self._self_calibrate(train)
        self.b += self.p["shift"]
        self.m = M.ShowModel().fit(train)

        self.start = min(c.index.min() for c in capacity.values())
        n = max(len(c) for c in capacity.values()) + 80
        self.days = pd.date_range(self.start, periods=n, freq="D")
        self.cap, self.E, self.V = {}, {}, {}
        for h, c in capacity.items():
            cv = np.full(n, float(c.iloc[-1]))
            cv[: len(c)] = c.to_numpy()
            self.cap[h] = np.maximum(cv, 0.0)
            self.E[h] = np.zeros(n)
            self.V[h] = np.zeros(n)
        if len(commitments):
            pc = self._calib(self.m.predict(commitments))
            for (h, a, nn), pp in zip(commitments[["hotel", "arrival_date", "nights"]].itertuples(index=False), pc):
                d0 = (pd.Timestamp(a) - self.start).days
                self.E[h][d0 : d0 + nn] += pp
                self.V[h][d0 : d0 + nn] += pp * (1 - pp)
        self._build_demand(train)
        self.growth = self._growth(train, commitments) if self.p["growth"] else {}
        self._bid_cache = {}
        self.cache = {}

    # ---------- calibration ----------
    def _self_calibrate(self, train):
        last = train.arrival_date.max()
        cut = (last - pd.DateOffset(months=self.p["hold_months"])).normalize() + pd.Timedelta(days=1)
        inner = train[train.arrival_date < cut]
        hold = train[(train.arrival_date >= cut) & (train.booking_date >= cut)]
        if len(inner) < 2000 or len(hold) < 500:
            return
        m = M.ShowModel().fit(inner)
        z = _logit(m.predict(hold))
        y = 1 - hold.cancelled.to_numpy()
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=1e4).fit(z.reshape(-1, 1), y)
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])

    def _calib(self, p):
        return _sig(self.a * _logit(np.asarray(p)) + self.b)

    def prepare(self, requests):
        """Speed-only hook used by sim.py: batch-predict rows (each prediction uses only its own row)."""
        self.cache = dict(zip(requests.booking_id, self._calib(self.m.predict(requests))))

    def pshow(self, b):
        if b["booking_id"] in self.cache:
            return float(self.cache[b["booking_id"]])
        return float(self._calib(np.array([self.m.predict_one(b)]))[0])

    # ---------- demand forecast ----------
    def _build_demand(self, train):
        t = train[train.cancelled == 0]
        nights = t.nights.to_numpy()
        rep = np.repeat(np.arange(len(t)), nights)
        k = np.concatenate([np.arange(x) for x in nights])
        night = pd.DatetimeIndex(t.arrival_date.to_numpy()[rep] + k.astype("timedelta64[D]"))
        g = pd.DataFrame({"hotel": t.hotel.to_numpy()[rep], "m": night.month, "dow": night.dayofweek,
                          "night": night, "lead": t.days_ahead.to_numpy()[rep] + k,
                          "rate": t.nightly_rate.to_numpy()[rep]})
        g = g[g.night <= train.arrival_date.max()]
        self.demand = {}
        for cols in (["hotel", "m", "dow"], ["hotel", "dow"]):
            for key, grp in g.groupby(cols):
                o = np.argsort(grp.lead.to_numpy())
                self.demand[key] = (grp.lead.to_numpy()[o], grp.rate.to_numpy()[o], grp.night.nunique())

    def _growth(self, train, com):
        horizon = min(self.p["growth_days"], len(self.days))
        out = {}

        def nights_table(df):
            nn = df.nights.to_numpy()
            rep = np.repeat(np.arange(len(df)), nn)
            k = np.concatenate([np.arange(x) for x in nn]) if len(nn) else np.array([], int)
            night = pd.DatetimeIndex(pd.to_datetime(df.arrival_date).to_numpy()[rep] + k.astype("timedelta64[D]"))
            return pd.DataFrame({"hotel": df.hotel.to_numpy()[rep], "night": night, "lead": df.days_ahead.to_numpy()[rep] + k})

        H = nights_table(train)
        H = H[H.night <= train.arrival_date.max()]
        H = H.assign(m=H.night.dt.month, dow=H.night.dt.dayofweek)
        C = nights_table(com) if len(com) else None
        for h in self.cap:
            Hh = H[H.hotel == h]
            ncal = Hh.groupby(["m", "dow"]).night.nunique()
            exp = 0.0
            for d in range(horizon):
                day = self.days[d]
                key = (day.month, day.dayofweek)
                if key in ncal.index:
                    sub = Hh[(Hh.m == key[0]) & (Hh.dow == key[1])]
                    exp += (sub.lead >= d).sum() / ncal[key]
            obs = len(C[(C.hotel == h) & (C.night <= self.days[horizon - 1])]) if C is not None and horizon > 0 else 0
            out[h] = float(np.clip(obs / exp, 0.5, 2.5)) if exp > 0 else 1.0
        return out

    def _future(self, h, day, tau):
        key = (h, day.month, day.dayofweek)
        if key not in self.demand:
            key = (h, day.dayofweek)
        ck = (key, tau)
        if ck not in self._bid_cache:
            lead, rate, n_cal = self.demand[key]
            i = np.searchsorted(lead, tau, side="left")
            r = np.sort(rate[:i])[::-1]
            self._bid_cache[ck] = (r, np.arange(1, len(r) + 1) / n_cal * self.p["alpha"] * self.growth.get(h, 1.0))
        return self._bid_cache[ck]

    def bid(self, h, d, tau):
        free = self.p["K"] * self.cap[h][d] - self.E[h][d]
        rates, cumd = self._future(h, self.days[d], tau)
        if len(cumd) == 0 or cumd[-1] <= free:
            return 0.0
        if free <= 0:
            return rates[0]
        return rates[np.searchsorted(cumd, free)]

    # ---------- decision ----------
    def decide(self, b):
        h, n = b["hotel"], int(b["nights"])
        d0 = (pd.Timestamp(b["arrival_date"]) - self.start).days
        if n <= 0 or d0 < 0 or h not in self.cap:
            return False
        p = self.pshow(b)
        sl = slice(d0, d0 + n)
        P = self.p
        E, V, cap = self.E[h][sl] + p, self.V[h][sl] + p * (1 - p), self.cap[h][sl]
        if (E + P["z"] * np.sqrt(V) > P["K"] * cap).any():
            return False
        lead = int(b["days_ahead"])
        lam = np.mean([self.bid(h, d0 + k, lead + k) for k in range(n)])
        if b["nightly_rate"] < P["beta"] * lam:
            return False
        self.E[h][sl] += p
        self.V[h][sl] += p * (1 - p)
        return True
