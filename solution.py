"""Your team's solution. At the end of the hackathon you pass this class to evaluate(Solution)
from evaluate.py, which creates one Solution and replays every 2017 booking request through it.

RULES -- Downtown Hotel has 124 rooms, Seaside Hotel 103. A guest who shows up earns
nightly_rate x nights. When a hotel is full at check-in, the cheapest arriving guests are
walked: they earn nothing and cost 2.5 x their stay value. A seated guest keeps the room.
Your result is the net revenue: revenue of seated guests - walk penalties.

Reading any 2017 data yourself is cheating.
"""
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

_CAT = [
    "hotel", "meal", "country", "sales_segment", "sales_channel",
    "room_type", "deposit", "guest_type", "agent", "company",
]
_NUM = [
    "days_ahead", "weekend_nights", "weekday_nights", "nights", "adults", "children",
    "babies", "returning_guest", "past_cancellations", "past_stays", "amendments",
    "waitlist_days", "nightly_rate", "special_requests", "value",
]
_HOTEL_SFX = {"Downtown Hotel": "dt", "Seaside Hotel": "ss"}


def _num(v) -> float:
    if v is None:
        return math.nan
    try:
        f = float(v)
        return f if not math.isnan(f) else math.nan
    except (TypeError, ValueError):
        return math.nan


def _key(v) -> str:
    """Canonical string key for a categorical value (treats None and NaN uniformly)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "nan"
    return str(v)


class _HGBMember:
    """Single HistGradientBoostingClassifier with ordinal-remapped fast single-row inference."""

    def __init__(self, num_cols, cat_cols, min_count=20, **hgb_params):
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.min_count = min_count
        self.hgb_params = dict(hgb_params)

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        n_num, n_cat = len(self.num_cols), len(self.cat_cols)
        X = np.empty((len(df), n_num + n_cat), dtype=float)
        for j, c in enumerate(self.num_cols):
            if c == "value":
                rates = pd.to_numeric(df["nightly_rate"], errors="coerce").to_numpy()
                nights = pd.to_numeric(df["nights"], errors="coerce").to_numpy()
                X[:, j] = rates * nights
            else:
                X[:, j] = pd.to_numeric(df[c], errors="coerce").to_numpy()
        for k, c in enumerate(self.cat_cols):
            m = self.codes[c]
            other = len(m)
            X[:, n_num + k] = [m.get(_key(v), other) for v in df[c].tolist()]
        return X

    def fit(self, train: pd.DataFrame, y: np.ndarray, sample_weight=None):
        self.codes = {}
        for c in self.cat_cols:
            vc = pd.Series([_key(v) for v in train[c].tolist()]).value_counts()
            keep = sorted(vc[vc >= self.min_count].index.tolist())[:250]
            self.codes[c] = {v: i for i, v in enumerate(keep)}
        mask = [False] * len(self.num_cols) + [True] * len(self.cat_cols)
        self.clf = HistGradientBoostingClassifier(
            categorical_features=mask if self.cat_cols else None,
            **self.hgb_params,
        )
        self.clf.fit(self._matrix(train), y, sample_weight=sample_weight)
        self._init_fast_predict()
        return self

    def _init_fast_predict(self):
        n_cat, n_num = len(self.cat_cols), len(self.num_cols)
        self._row_buf = np.empty((1, n_cat + n_num), dtype=float)
        self._raw_buf = np.zeros((1, 1), dtype=self.clf._baseline_prediction.dtype, order="F")
        self._cat_maps = []
        if self.clf._preprocessor is not None:
            enc = self.clf._preprocessor.named_transformers_["encoder"]
            for c, cats in zip(self.cat_cols, enc.categories_):
                orig_map = self.codes[c]
                ord_lookup = {float(v): float(idx) for idx, v in enumerate(cats)}
                other_ord = ord_lookup.get(float(len(orig_map)), math.nan)
                new_map = {k: ord_lookup.get(float(v), other_ord) for k, v in orig_map.items()}
                self._cat_maps.append((new_map, other_ord))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(self._matrix(df))[:, 1]

    def predict_one(self, b: dict) -> float:
        x = self._row_buf
        n_cat = len(self._cat_maps)
        if n_cat:
            for k, c in enumerate(self.cat_cols):
                mp, oth = self._cat_maps[k]
                x[0, k] = mp.get(_key(b.get(c)), oth)
            num_off = n_cat
        else:
            num_off = 0
            for k, c in enumerate(self.cat_cols):
                mp = self.codes[c]
                x[0, len(self.num_cols) + k] = mp.get(_key(b.get(c)), len(mp))
        for j, c in enumerate(self.num_cols):
            x[0, num_off + j] = (
                _num(b.get("nightly_rate")) * _num(b.get("nights"))
                if c == "value"
                else _num(b.get(c))
            )
        raw = self._raw_buf
        raw[0, 0] = self.clf._baseline_prediction[0, 0]
        self.clf._predict_iterations(x, self.clf._predictors, raw, False, 1)
        return 1.0 / (1.0 + math.exp(-float(raw[0, 0])))


class _ShowModel:
    """Blended show-up classifier combining full-feature and shift-resistant no-country models."""

    def __init__(self, blend_nocountry=0.35, recency_decay=0.004, min_count=20):
        self.blend_nocountry = float(blend_nocountry)
        self.recency_decay = float(recency_decay)
        self.min_count = min_count

    def fit(self, train: pd.DataFrame):
        y = 1 - train["cancelled"].to_numpy()
        sw = None
        if self.recency_decay > 0.0:
            arr = pd.to_datetime(train["arrival_date"])
            days_ago = (arr.max() - arr).dt.days.to_numpy(dtype=float)
            sw = np.exp(-self.recency_decay * days_ago)
        self.m1 = _HGBMember(
            _NUM, _CAT, self.min_count,
            max_iter=200, learning_rate=0.07, max_leaf_nodes=63,
            min_samples_leaf=40, l2_regularization=1.0, random_state=0,
        ).fit(train, y, sample_weight=sw)
        self.m2 = None
        if self.blend_nocountry > 0.0:
            nocnt = [c for c in _CAT if c != "country"]
            self.m2 = _HGBMember(
                _NUM, nocnt, self.min_count,
                max_iter=150, learning_rate=0.07, max_leaf_nodes=31,
                min_samples_leaf=80, l2_regularization=2.0, random_state=1,
            ).fit(train, y, sample_weight=sw)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        p1 = self.m1.predict(df)
        if self.m2 is None:
            return p1
        w = self.blend_nocountry
        return (1.0 - w) * p1 + w * self.m2.predict(df)

    def predict_one(self, b: dict) -> float:
        m1 = self.m1
        if self.m2 is None:
            return m1.predict_one(b)
        m2 = self.m2
        x1, x2 = m1._row_buf, m2._row_buf
        k2 = 0
        for k1, c in enumerate(m1.cat_cols):
            kv = _key(b.get(c))
            mp1, oth1 = m1._cat_maps[k1]
            x1[0, k1] = mp1.get(kv, oth1)
            if c != "country":
                mp2, oth2 = m2._cat_maps[k2]
                x2[0, k2] = mp2.get(kv, oth2)
                k2 += 1
        n_cat1 = len(m1.cat_cols)
        n_cat2 = len(m2.cat_cols)
        rate_v = _num(b.get("nightly_rate"))
        nights_v = _num(b.get("nights"))
        for j, c in enumerate(m1.num_cols):
            if c == "value":
                x1[0, n_cat1 + j] = rate_v * nights_v
            elif c == "nightly_rate":
                x1[0, n_cat1 + j] = rate_v
            elif c == "nights":
                x1[0, n_cat1 + j] = nights_v
            else:
                x1[0, n_cat1 + j] = _num(b.get(c))
        x2[0, n_cat2:] = x1[0, n_cat1:]
        raw1 = m1._raw_buf
        raw1[0, 0] = m1.clf._baseline_prediction[0, 0]
        m1.clf._predict_iterations(x1, m1.clf._predictors, raw1, False, 1)
        p1 = 1.0 / (1.0 + math.exp(-float(raw1[0, 0])))
        raw2 = m2._raw_buf
        raw2[0, 0] = m2.clf._baseline_prediction[0, 0]
        m2.clf._predict_iterations(x2, m2.clf._predictors, raw2, False, 1)
        p2 = 1.0 / (1.0 + math.exp(-float(raw2[0, 0])))
        w = self.blend_nocountry
        return (1.0 - w) * p1 + w * p2


class Solution:
    params = dict(
        blend_nocountry=0.35,
        recency_decay=0.004,
        K=1.20,
        K_dt=1.19,
        K_ss=1.20,
        alpha=1.9,
        alpha_dt=2.0,
        alpha_ss=1.8,
        beta=1.0,
        demand_recency=0.5,
        pmin=0.05,
        gamma_p=0.35,
        w_max=0.30,
        walk_floor=60.0,
        K_soft=0.95,
        bump_mult=2.5,
        K_bump=1.40,
        seat_cap_mult=1.22,
        wm=0.20,
        wdow=0.15,
    )

    def __init__(self, commitments: pd.DataFrame, capacity: dict[str, pd.Series], train: pd.DataFrame = None) -> None:
        """Called once, before 2017 starts.

        commitments  2017 arrivals booked before 2017-01-01. The hotel already confirmed them, so
                     decide() is never called for them. Same columns as train.csv, except:
                     - no cancelled column;
                     - arrival_date and booking_date are already pandas Timestamps, not strings.

        capacity     hotel -> rooms free on each night of 2017 (indexed by date): the rooms minus
                     guests who arrived in 2016 and are still in house.
        """
        self.p = dict(self.params)
        if train is None:
            train = pd.read_csv("train.csv")
        train = train.copy()
        train["arrival_date"] = pd.to_datetime(train["arrival_date"])
        train["booking_date"] = pd.to_datetime(train["booking_date"])

        self.m = _ShowModel(
            blend_nocountry=self.p["blend_nocountry"],
            recency_decay=self.p["recency_decay"],
        ).fit(train)

        self.start = min(pd.Timestamp(c.index.min()) for c in capacity.values())
        self.start_date = self.start.date()
        n_buf = max(len(c) for c in capacity.values()) + 200
        self.days = pd.date_range(self.start, periods=n_buf, freq="D")
        self.cap, self.E = {}, {}
        for h, c in capacity.items():
            full_rooms = max(float(c.iloc[-1]), float(c.max()))
            cv = np.full(n_buf, full_rooms)
            cv[: len(c)] = c.to_numpy(dtype=float)
            self.cap[h] = np.maximum(cv, 0.0)
            self.E[h] = np.zeros(n_buf, dtype=float)

        self.cache = {}
        if len(commitments):
            pc = self.m.predict(commitments)
            scm = self.p.get("seat_cap_mult")
            if scm is None:
                for (h, a, nn), pp in zip(commitments[["hotel", "arrival_date", "nights"]].itertuples(index=False), pc):
                    if h not in self.E:
                        continue
                    d0 = (pd.Timestamp(a).date() - self.start_date).days
                    n_stay = int(nn)
                    if d0 >= 0 and n_stay > 0:
                        d1 = min(d0 + n_stay, len(self.E[h]))
                        if d1 > d0:
                            self.E[h][d0:d1] += pp
            else:
                by_hd = {}
                for (h, a, nn, rate), pp in zip(
                    commitments[["hotel", "arrival_date", "nights", "nightly_rate"]].itertuples(index=False), pc
                ):
                    if h not in self.E:
                        continue
                    d0 = (pd.Timestamp(a).date() - self.start_date).days
                    n_stay = int(nn)
                    if d0 >= 0 and n_stay > 0:
                        by_hd.setdefault((h, d0), []).append((float(rate) * n_stay, n_stay, float(pp)))
                scm_f = float(scm)
                for h in self.E:
                    in_h = np.zeros(n_buf, dtype=float)
                    for d in range(n_buf):
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
                                d1 = min(d + n_stay, n_buf)
                                if d1 > d + 1 and seated_p > 0:
                                    in_h[d + 1 : d1] += seated_p

        self._build_demand(train)
        self._bid_cache = {}

    def _hp(self, h: str, name: str) -> float:
        sfx = _HOTEL_SFX.get(h)
        if sfx is not None:
            v = self.p.get(f"{name}_{sfx}")
            if v is not None:
                return float(v)
        return float(self.p[name])

    def _build_demand(self, train: pd.DataFrame) -> None:
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
                lead_a = grp.lead.to_numpy()
                rate_a = grp.rate.to_numpy()
                nw_a = grp.nw.to_numpy()
                o1 = np.argsort(lead_a)
                lead_a, rate_a, nw_a = lead_a[o1], rate_a[o1], nw_a[o1]
                o2 = np.argsort(-rate_a, kind="mergesort")
                self.demand[(tag, key)] = (
                    lead_a[o2],
                    rate_a[o2],
                    nw_a[o2],
                    max(n_cal, 1e-6),
                )

    def _future_key(self, h: str, dkey: tuple, tau: int):
        ck = (dkey, tau)
        if ck not in self._bid_cache:
            lead, rate, nw, n_cal = self.demand[dkey]
            mask = lead < tau
            r, w = rate[mask], nw[mask]
            scale = self._hp(h, "alpha")
            self._bid_cache[ck] = (r, np.cumsum(w) / n_cal * scale)
        return self._bid_cache[ck]

    @staticmethod
    def _q_lookup(rates: np.ndarray, cumd: np.ndarray, free: float) -> float:
        if len(cumd) == 0 or cumd[-1] <= free:
            return 0.0
        if free <= 0:
            return float(rates[0]) if len(rates) else 0.0
        return float(rates[np.searchsorted(cumd, free)])

    def _bid(self, h: str, d: int, tau: int, K_h: float) -> float:
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

    def prepare(self, requests: pd.DataFrame) -> None:
        """Optional batch-prediction speed hook used by sim.py (each prediction uses only its own row)."""
        if len(requests):
            self.cache = dict(zip(requests.booking_id, self.m.predict(requests)))

    def _pshow(self, booking: dict) -> float:
        bid = booking.get("booking_id")
        if bid in self.cache:
            return float(self.cache[bid])
        return self.m.predict_one(booking)

    def decide(self, booking: dict) -> bool:
        """Called once per 2017 booking request, in booking-date order. `booking` is one row with
        the same columns as commitments, as {column: value}. Return True to confirm the booking,
        False to reject it."""
        h = booking.get("hotel")
        if h not in self.cap:
            return False
        try:
            n = int(booking["nights"])
            lead = int(booking["days_ahead"])
            rate = float(booking["nightly_rate"])
        except (KeyError, TypeError, ValueError):
            return False
        if n <= 0 or not math.isfinite(rate):
            return False
        arr = booking.get("arrival_date")
        arr_date = arr.date() if hasattr(arr, "date") else pd.Timestamp(arr).date()
        d0 = (arr_date - self.start_date).days
        if d0 < 0 or d0 + n > len(self.cap[h]):
            return False

        p = self._pshow(booking)
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

        bids = [self._bid(h, d0 + k, lead + k, K_h) for k in range(n)]
        walk_floor = P["walk_floor"]
        if walk_floor > 0.0:
            K_soft = P["K_soft"]
            denom = max(K_h - K_soft, 0.05)
            for k in range(n):
                load_k = E_next[k] / max(cap_sl[k], 1.0)
                if load_k > K_soft:
                    bids[k] = max(bids[k], walk_floor * ((load_k - K_soft) / denom))

        w_max = P["w_max"]
        lam = (1.0 - w_max) * float(np.mean(bids)) + w_max * float(np.max(bids))

        if E_next[0] > K_h * cap_sl[0]:
            if not (bump_on and rate >= P["bump_mult"] * max(lam, 100.0)):
                return False

        eff_rate = rate * (p ** P["gamma_p"])
        if eff_rate < P["beta"] * lam:
            return False

        self.E[h][sl] += p
        return True
