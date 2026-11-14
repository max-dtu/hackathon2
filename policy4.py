"""Policy v4: bid-price revenue management with shift-resistant show model & modular refinements.

Extends policy2 with:
- Blend of full-feature and no-country HGB classifiers + optional recency decay (via model.ShowModel)
- Per-hotel capacity multiplier (K_dt, K_ss), demand scale (alpha_dt, alpha_ss), logit shift (shift_dt, shift_ss)
- Optional recency weighting on the historical demand curve (demand_recency)
- Optional minimum show probability (pmin), reliability power (gamma_p), bottleneck blend (w_max),
  late-booking walk-risk floor (walk_floor, K_soft), and high-rate arrival bump allowance (bump_mult, K_bump)
"""
import math

import numpy as np
import pandas as pd

import model as M

_HOTEL_SFX = {"Downtown Hotel": "dt", "Seaside Hotel": "ss"}


class Policy:
    params = dict(
        blend_nocountry=0.35,
        recency_decay=0.0,
        shift=0.0,
        shift_dt=None,
        shift_ss=None,
        K=1.20,
        K_dt=None,
        K_ss=None,
        alpha=2.0,
        alpha_dt=None,
        alpha_ss=None,
        beta=1.0,
        demand_recency=0.0,
        pmin=0.0,
        gamma_p=0.0,
        w_max=0.0,
        walk_floor=0.0,
        K_soft=1.0,
        bump_mult=0.0,
        K_bump=1.35,
        seat_cap_mult=None,
        wm=0.0,
        wdow=0.0,
        adapt=False,
        growth_days=180,
    )

    def __init__(self, commitments, capacity, train=None, **kw):
        self.p = {**self.params, **kw}
        if train is None:
            train = pd.read_csv("train.csv")
        train = train.copy()
        train["arrival_date"] = pd.to_datetime(train["arrival_date"])
        train["booking_date"] = pd.to_datetime(train["booking_date"])

        self.m = M.ShowModel(
            blend_nocountry=self.p["blend_nocountry"],
            recency_decay=self.p["recency_decay"],
        ).fit(train)

        self.start = min(c.index.min() for c in capacity.values())
        n = max(len(c) for c in capacity.values()) + 120
        self.days = pd.date_range(self.start, periods=n, freq="D")
        self.cap, self.E = {}, {}
        for h, c in capacity.items():
            full_rooms = max(float(c.iloc[-1]), float(c.max()))
            cv = np.full(n, full_rooms)
            cv[: len(c)] = c.to_numpy()
            self.cap[h] = np.maximum(cv, 0.0)
            self.E[h] = np.zeros(n)
        self.cache = {}
        if len(commitments):
            pc = self._calib(self.m.predict(commitments), commitments["hotel"].to_numpy())
            scm = self.p.get("seat_cap_mult")
            if scm is None:
                for (h, a, nn), pp in zip(commitments[["hotel", "arrival_date", "nights"]].itertuples(index=False), pc):
                    d0 = (pd.Timestamp(a) - self.start).days
                    if d0 >= 0 and h in self.E:
                        self.E[h][d0 : d0 + int(nn)] += pp
            else:
                by_hd = {}
                for (h, a, nn, rate), pp in zip(
                    commitments[["hotel", "arrival_date", "nights", "nightly_rate"]].itertuples(index=False), pc
                ):
                    d0 = (pd.Timestamp(a) - self.start).days
                    n_stay = int(nn)
                    if d0 >= 0 and h in self.E and n_stay > 0:
                        by_hd.setdefault((h, d0), []).append((float(rate) * n_stay, n_stay, float(pp)))
                scm_f = float(scm)
                for h in self.E:
                    in_h = np.zeros(n, dtype=float)
                    for d in range(n):
                        items = by_hd.get((h, d), [])
                        free = max(0.0, scm_f * self.cap[h][d] - in_h[d])
                        tot_arr = sum(pp for _, _, pp in items)
                        self.E[h][d] = in_h[d] + tot_arr
                        if items:
                            items.sort(key=lambda x: -x[0])
                            rem = free
                            for val, n_stay, pp in items:
                                seated_p = min(pp, rem)
                                rem = max(0.0, rem - seated_p)
                                d1 = min(d + n_stay, n)
                                if d1 > d + 1 and seated_p > 0:
                                    in_h[d + 1 : d1] += seated_p
        self._build_demand(train)
        self.growth = self._growth(train, commitments) if self.p["adapt"] else {}
        self._bid_cache = {}

    def _hp(self, h: str, name: str) -> float:
        sfx = _HOTEL_SFX.get(h)
        if sfx is not None:
            v = self.p.get(f"{name}_{sfx}")
            if v is not None:
                return float(v)
        return float(self.p[name])

    # ---------- show probability ----------
    def _calib_scalar(self, p: float, h: str) -> float:
        s = self._hp(h, "shift")
        if s == 0.0:
            return float(p)
        p = min(max(float(p), 1e-4), 1.0 - 1e-4)
        z = math.log(p / (1.0 - p)) + s
        return 1.0 / (1.0 + math.exp(-z))

    def _calib(self, p, hotels=None):
        p = np.asarray(p, dtype=float)
        if hotels is None:
            s = np.full(len(p), float(self.p["shift"]))
        else:
            s = np.array([self._hp(str(h), "shift") for h in hotels], dtype=float)
        if np.all(s == 0.0):
            return p
        pc = np.clip(p, 1e-4, 1.0 - 1e-4)
        z = np.log(pc / (1.0 - pc)) + s
        return 1.0 / (1.0 + np.exp(-z))

    def prepare(self, requests: pd.DataFrame):
        """Speed-only hook used by sim.py: batch-predict rows (each prediction uses only its own row)."""
        probs = self._calib(self.m.predict(requests), requests["hotel"].to_numpy())
        self.cache = dict(zip(requests.booking_id, probs))

    def pshow(self, b: dict) -> float:
        bid = b.get("booking_id")
        if bid in self.cache:
            return float(self.cache[bid])
        return self._calib_scalar(self.m.predict_one(b), b["hotel"])

    # ---------- demand forecast ----------
    def _growth(self, train: pd.DataFrame, com: pd.DataFrame) -> dict:
        out = {}
        horizon = min(int(self.p["growth_days"]), len(self.days))
        if horizon <= 0:
            return out

        def nights_table(df):
            nn = df.nights.to_numpy(dtype=int)
            rep = np.repeat(np.arange(len(df)), nn)
            k = np.concatenate([np.arange(x) for x in nn]) if len(nn) else np.array([], int)
            night = pd.DatetimeIndex(pd.to_datetime(df.arrival_date).to_numpy()[rep] + k.astype("timedelta64[D]"))
            return pd.DataFrame({"hotel": df.hotel.to_numpy()[rep], "night": night,
                                 "lead": df.days_ahead.to_numpy(dtype=int)[rep] + k})

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
                if key not in ncal.index:
                    continue
                sub = Hh[(Hh.m == key[0]) & (Hh.dow == key[1])]
                exp += (sub.lead >= d).sum() / ncal[key]
            obs = len(C[(C.hotel == h) & (C.night <= self.days[horizon - 1])]) if C is not None else 0
            out[h] = float(np.clip(obs / exp, 0.5, 2.5)) if exp > 0 else 1.0
        return out

    def _build_demand(self, train: pd.DataFrame):
        max_night = train["arrival_date"].max()
        t = train[train.cancelled == 0]
        nights = t.nights.to_numpy(dtype=int)
        rep = np.repeat(np.arange(len(t)), nights)
        k = np.concatenate([np.arange(x) for x in nights])
        night = pd.DatetimeIndex(t.arrival_date.to_numpy()[rep] + k.astype("timedelta64[D]"))
        g = pd.DataFrame({
            "hotel": t.hotel.to_numpy()[rep],
            "m": night.month,
            "dow": night.dayofweek,
            "night": night,
            "lead": t.days_ahead.to_numpy(dtype=int)[rep] + k,
            "rate": t.nightly_rate.to_numpy(dtype=float)[rep],
        })
        g = g[g.night <= max_night]
        dr = float(self.p["demand_recency"])
        if dr > 0.0:
            age_yr = (max_night - g["night"]).dt.days.to_numpy(dtype=float) / 365.0
            g["nw"] = np.exp(-dr * age_yr)
        else:
            g["nw"] = 1.0
        self.demand = {}
        for tag, cols in (("mdow", ["hotel", "m", "dow"]), ("m", ["hotel", "m"]), ("dow", ["hotel", "dow"])):
            for key, grp in g.groupby(cols):
                n_cal = float(grp.drop_duplicates("night")["nw"].sum())
                o = np.argsort(grp.lead.to_numpy())
                self.demand[(tag, key)] = (
                    grp.lead.to_numpy()[o],
                    grp.rate.to_numpy()[o],
                    grp.nw.to_numpy()[o],
                    max(n_cal, 1e-6),
                )

    def _future_key(self, h: str, dkey: tuple, tau: int):
        ck = (dkey, tau)
        if ck not in self._bid_cache:
            lead, rate, nw, n_cal = self.demand[dkey]
            i = np.searchsorted(lead, tau, side="left")
            r, w = rate[:i], nw[:i]
            o = np.argsort(-r)
            scale = self._hp(h, "alpha") * self.growth.get(h, 1.0)
            self._bid_cache[ck] = (r[o], np.cumsum(w[o]) / n_cal * scale)
        return self._bid_cache[ck]

    @staticmethod
    def _q_lookup(rates: np.ndarray, cumd: np.ndarray, free: float) -> float:
        if len(cumd) == 0 or cumd[-1] <= free:
            return 0.0
        if free <= 0:
            return float(rates[0]) if len(rates) else 0.0
        return float(rates[np.searchsorted(cumd, free)])

    def bid(self, h: str, d: int, tau: int, K_h: float) -> float:
        free = K_h * self.cap[h][d] - self.E[h][d]
        day = self.days[d]
        k_mdow = ("mdow", (h, day.month, day.dayofweek))
        k_dow = ("dow", (h, day.dayofweek))
        b_mdow = self._q_lookup(*self._future_key(h, k_mdow if k_mdow in self.demand else k_dow, tau), free)
        wm = float(self.p["wm"])
        wdow = float(self.p["wdow"])
        if wm > 0.0 or wdow > 0.0:
            k_m = ("m", (h, day.month))
            b_dow = self._q_lookup(*self._future_key(h, k_dow, tau), free)
            b_m = self._q_lookup(*self._future_key(h, k_m, tau), free) if k_m in self.demand else b_dow
            return (1.0 - wm - wdow) * b_mdow + wm * b_m + wdow * b_dow
        return b_mdow

    # ---------- decision ----------
    def decide(self, b: dict) -> bool:
        h = b["hotel"]
        n = int(b["nights"])
        d0 = (pd.Timestamp(b["arrival_date"]) - self.start).days
        if n <= 0 or d0 < 0 or h not in self.cap or d0 + n > len(self.cap[h]):
            return False
        rate = float(b["nightly_rate"])
        if not math.isfinite(rate):
            return False
        p = self.pshow(b)
        P = self.p
        if p < P["pmin"]:
            return False

        K_h = self._hp(h, "K")
        sl = slice(d0, d0 + n)
        E_next = self.E[h][sl] + p
        cap_sl = self.cap[h][sl]
        bump_on = P["bump_mult"] > 0.0 and P["K_bump"] > K_h
        max_arr_cap = (P["K_bump"] if bump_on else K_h) * cap_sl[0]
        if E_next[0] > max_arr_cap or (n > 1 and (E_next[1:] > K_h * cap_sl[1:]).any()):
            return False

        lead = int(b["days_ahead"])
        bids = [self.bid(h, d0 + k, lead + k, K_h) for k in range(n)]
        if P["walk_floor"] > 0.0:
            K_soft = P["K_soft"]
            denom = max(K_h - K_soft, 0.05)
            for k in range(n):
                load_k = E_next[k] / max(cap_sl[k], 1.0)
                if load_k > K_soft:
                    bids[k] = max(bids[k], P["walk_floor"] * ((load_k - K_soft) / denom))

        w_max = P["w_max"]
        lam = (1.0 - w_max) * float(np.mean(bids)) + w_max * float(np.max(bids)) if w_max > 0.0 else float(np.mean(bids))

        if E_next[0] > K_h * cap_sl[0]:
            if not (bump_on and rate >= P["bump_mult"] * max(lam, 100.0)):
                return False

        eff_rate = rate * (p ** P["gamma_p"]) if P["gamma_p"] > 0.0 else rate
        if eff_rate < P["beta"] * lam:
            return False

        self.E[h][sl] += p
        return True
