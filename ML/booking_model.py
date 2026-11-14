"""Booking decision model: who stays, and which rooms to keep for a better bidder.

The decision for every booking request rests on two questions.

1. WHO STAYS?  (`StayModel`)
   A machine-learning model, trained on the 2015-2016 bookings, estimates the chance that
   this guest really shows up. The default ("combined") averages two views: a plain table
   of stay rates per segment (deposit, sales segment, lead time), which is steady, and a
   gradient-boosting model that also reads agent, country, requests and more, which is
   sharper. Recent bookings count more than old ones. Finally the model is checked against
   the most recent three months it has NOT seen and corrected for how much it over-promised
   there, because cancellations keep rising (`StayModel._calibrate`).

2. IS THE ROOM WORTH MORE IF WE KEEP IT?  (`BookingPolicy._price_floor`)
   Who is still to come for this night? Two estimates are averaged:
   - last year: who booked that same night last year AFTER this point (`DemandForecast`);
   - recent weeks: who booked late for the same weekday in the last four weeks, scaled to
     how busy and how dear this night already is (`RecentDemand`).
   Those bidders are lined up from the highest rate down and handed the rooms we expect to
   have free. The rate of the first bidder who would NOT get a room is the price floor of
   that night: selling below it gives a room away too cheaply.

3. LATE BOOKINGS AND OVERBOOKING  (`BookingPolicy`, late_rule=True)
   We cannot hold a request and wait for a better one: every request is answered at once.
   So "give the room to whoever pays most" is done in three ways:
   - Early on, rooms are held back for the late bookers we expect (the price floor above).
   - We overbook: more bookings are confirmed than there are rooms, because many cancel.
     What counts is the number of guests we EXPECT to show up, not the number of bookings.
   - When the hotel is full on a day, the rules walk the cheapest arriving guest. A late
     request worth more than that guest therefore keeps its room and bumps them. Bumping
     costs their lost revenue plus the 2.5x penalty (3.5x their stay value in total), so the
     late request is only accepted when it is worth more than that.

A request is accepted when its expected gain is positive:

    risk_full = chance that the guests already accepted fill the hotel that night.
    Room free   -> we earn the rate, minus the price floor (what a later bidder would pay).
    Hotel full  -> either this guest is the cheapest and is walked (-2.5 x stay value),
                   or it outbids the cheapest arrival, who is walked instead (-3.5 x theirs).
"""
from __future__ import annotations

import bisect
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

TRAIN_CSV = Path(__file__).resolve().parent.parent / "train.csv"   # repo root
ROOMS = {"Downtown Hotel": 124, "Seaside Hotel": 103}
WALK_PENALTY = 2.5                    # a walked guest costs 2.5x the stay value
BUMP_COST = 1 + WALK_PENALTY          # bumping a seated guest: their revenue + the penalty
# Booking periods, by how many days before arrival the request comes in.
PERIODS = [(7, "last minute (0-7 days)"), (30, "late (8-30 days)"),
           (90, "mid (31-90 days)"), (10 ** 6, "early (90+ days)")]


def booking_period(days_ahead: float) -> str:
    return next(name for limit, name in PERIODS if days_ahead <= limit)


def _period_index(days_ahead: np.ndarray) -> np.ndarray:
    limits = [limit for limit, _ in PERIODS[:-1]]
    return np.searchsorted(limits, np.nan_to_num(days_ahead, nan=0.0), side="left")


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1 - p))


def _matching_shift(logit: np.ndarray, y: np.ndarray) -> float:
    """The log-odds shift that makes the average predicted stay chance equal the real one."""
    lo, hi = -3.0, 3.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(logit - mid)))).mean() > y.mean():
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
YEAR = pd.Timedelta(days=364)         # 52 weeks: the same weekday one year later

# --------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------
CAT_COLS = ["hotel", "meal", "country", "sales_segment", "sales_channel", "room_type",
            "deposit", "guest_type", "agent", "company"]
NUM_COLS = ["days_ahead", "weekend_nights", "weekday_nights", "adults", "children", "babies",
            "returning_guest", "past_cancellations", "past_stays", "amendments",
            "waitlist_days", "nightly_rate", "special_requests", "nights"]
EXTRA_COLS = ["arrival_dow", "booking_dow", "stay_value", "guests", "rate_per_guest"]
# booking_id and anything derived from the order of the rows are deliberately NOT used:
# train.csv lists cancelled bookings before the ones that stayed, so row order gives the
# answer away. A model that leans on it learns nothing about guests.
FEATURES = CAT_COLS + NUM_COLS + EXTRA_COLS
MODEL_KINDS = ("segments", "tree", "logistic", "forest", "boosting", "combined")


def _key(value) -> str:
    """One spelling per category level, whether it arrives as 9, 9.0, '9' or NaN."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-1"
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    for col in ("arrival_date", "booking_date"):
        df[col] = pd.to_datetime(df[col]).dt.normalize()
    df["agent"] = df["agent"].fillna(-1)
    df["company"] = df["company"].fillna(-1)
    return df


class _SegmentTable:
    """The simplest model: the stay rate of each (deposit, sales segment, lead-time band)."""

    BANDS = [7, 30, 90, 180]

    def _keys(self, X):
        dep, seg, lead = (X[:, FEATURES.index(c)] for c in ("deposit", "sales_segment", "days_ahead"))
        return list(zip(np.round(dep, 6), np.round(seg, 6), np.digitize(lead, self.BANDS)))

    def fit(self, X, y, sample_weight):
        tot, hit = defaultdict(float), defaultdict(float)
        for k, yi, wi in zip(self._keys(X), y, sample_weight):
            tot[k] += wi
            hit[k] += wi * yi
        self.prior = float(np.average(y, weights=sample_weight))
        self.table = {k: (hit[k] + 5 * self.prior) / (tot[k] + 5) for k in tot}
        return self

    def predict_proba(self, X):
        p = np.array([self.table.get(k, self.prior) for k in self._keys(X)])
        return np.column_stack([1 - p, p])


class _Combined:
    """Average of the segment table (steady, hard to fool) and gradient boosting (sharp, but
    leans on inputs such as country whose meaning drifts). Averaged in log-odds."""

    def fit(self, X, y, sample_weight):
        self.parts = [_make_model(k).fit(X, y, sample_weight=sample_weight) for k in ("segments", "boosting")]
        return self

    def predict_proba(self, X):
        logits = [_logit(np.clip(m.predict_proba(X)[:, 1], 0.001, 0.999)) for m in self.parts]
        p = 1 / (1 + np.exp(-np.mean(logits, axis=0)))
        return np.column_stack([1 - p, p])


def _make_model(kind: str):
    if kind == "combined":
        return _Combined()
    if kind == "segments":
        return _SegmentTable()
    if kind == "tree":
        return DecisionTreeClassifier(max_leaf_nodes=24, min_samples_leaf=200, random_state=0)
    if kind == "logistic":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=1000))
    if kind == "forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1,
                                      random_state=0)
    if kind == "boosting":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                              min_samples_leaf=100, l2_regularization=1.0,
                                              random_state=0)
    raise ValueError(f"unknown model kind {kind!r}; choose from {MODEL_KINDS}")


class StayModel:
    """Estimates P(guest shows up) for a booking."""

    def __init__(self, kind: str = "combined", recency_days: float = 60.0,
                 calibration: str = "period", calibration_days: int = 90, ignore: tuple = ()):
        self.kind = kind
        self.ignore = [FEATURES.index(f) for f in ignore]   # inputs to blank out (for tests)
        self.recency_days = recency_days   # a booking made this long ago counts 1/e as much
        self.calibration = calibration     # "period" or "none" -- see _calibrate
        self.calibration_days = calibration_days
        self.shift = np.zeros(len(PERIODS))   # per booking period, in log-odds
        self.levels: dict[str, dict[str, float]] = {}

    # -- training ----------------------------------------------------------------------
    def fit(self, history: pd.DataFrame) -> "StayModel":
        df = _clean(history)
        if self.calibration != "none":
            self._calibrate(df)
        y = 1 - df["cancelled"].to_numpy()
        age = (df["booking_date"].max() - df["booking_date"]).dt.days.to_numpy()
        w = np.exp(-age / self.recency_days)
        w = w / w.mean()
        self.prior = float(np.average(y, weights=w))
        # Each category level is replaced by its (smoothed, recency-weighted) stay rate, so
        # "agent 240" becomes "an agent whose guests show up 57% of the time".
        for col in CAT_COLS:
            g = pd.DataFrame({"k": df[col].map(_key), "w": w, "wy": w * y}).groupby("k")
            stats = g.agg(n=("w", "size"), w=("w", "sum"), wy=("wy", "sum"))
            stats = stats[stats["n"] >= 5]
            self.levels[col] = ((stats["wy"] + 20 * self.prior) / (stats["w"] + 20)).to_dict()
        self.model = _make_model(self.kind)
        X = self._matrix(df)
        if self.kind == "logistic":
            self.model.fit(X, y, logisticregression__sample_weight=w)
        else:
            self.model.fit(X, y, sample_weight=w)
        return self

    def _calibrate(self, df: pd.DataFrame) -> None:
        """Reality check on the most recent months. Cancellations keep rising, so a model
        trained on the past promises too many guests. We train a copy WITHOUT the last
        `calibration_days` of arrivals, see how far it over-promised on them, and lower the
        final model's predictions by that much (separately for each booking period)."""
        cut = df["arrival_date"].max() - pd.Timedelta(days=self.calibration_days)
        old, recent = df[df["arrival_date"] <= cut], df[df["arrival_date"] > cut]
        if len(old) < 2000 or len(recent) < 500:
            return
        probe = StayModel(self.kind, self.recency_days, calibration="none",
                          ignore=[FEATURES[j] for j in self.ignore]).fit(old)
        y = 1 - recent["cancelled"].to_numpy()
        logit = _logit(probe.predict(recent))
        overall = _matching_shift(logit, y)
        band = _period_index(recent["days_ahead"].to_numpy(float))
        for i in range(len(PERIODS)):
            m = band == i
            trust = m.sum() / (m.sum() + 300)     # small periods lean on the overall figure
            own = _matching_shift(logit[m], y[m]) if m.sum() >= 50 else overall
            self.shift[i] = trust * own + (1 - trust) * overall

    # -- batch features (history, commitments, backtests) ------------------------------
    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        cols = [df[c].map(_key).map(self.levels[c]).to_numpy(float) for c in CAT_COLS]
        num = {c: pd.to_numeric(df[c], errors="coerce").to_numpy(float) for c in NUM_COLS}
        cols += [num[c] for c in NUM_COLS]
        guests = np.nan_to_num(num["adults"]) + np.nan_to_num(num["children"]) + np.nan_to_num(num["babies"])
        cols += [df["arrival_date"].dt.dayofweek.to_numpy(float),
                 df["booking_date"].dt.dayofweek.to_numpy(float),
                 num["nightly_rate"] * num["nights"], guests,
                 num["nightly_rate"] / np.maximum(guests, 1)]
        X = np.column_stack(cols)
        X[:, self.ignore] = 0.0
        return X

    def _proba(self, X: np.ndarray) -> np.ndarray:
        p = np.clip(self.model.predict_proba(X)[:, 1], 0.001, 0.999)
        if not self.shift.any():
            return p
        shift = self.shift[_period_index(X[:, FEATURES.index("days_ahead")])]
        return np.clip(1 / (1 + np.exp(-(_logit(p) - shift))), 0.001, 0.999)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """P(stay) for a table of bookings."""
        if len(df) == 0:
            return np.zeros(0)
        return self._proba(self._matrix(_clean(df)))

    # -- one live request ---------------------------------------------------------------
    def predict_one(self, b: dict) -> float:
        """P(stay) for one incoming request."""
        row = [self.levels[c].get(_key(b.get(c)), np.nan) for c in CAT_COLS]
        num = {c: _num(b.get(c)) for c in NUM_COLS}
        row += [num[c] for c in NUM_COLS]
        arrival = pd.Timestamp(b["arrival_date"]).normalize()
        booked = pd.Timestamp(b["booking_date"]).normalize()
        guests = sum(0.0 if math.isnan(num[c]) else num[c] for c in ("adults", "children", "babies"))
        row += [arrival.dayofweek, booked.dayofweek, num["nightly_rate"] * num["nights"], guests,
                num["nightly_rate"] / max(guests, 1)]
        X = np.asarray(row, dtype=float).reshape(1, -1)
        X[:, self.ignore] = 0.0
        return float(self._proba(X)[0])


# --------------------------------------------------------------------------------------
# Future bidders -> price floor of a room-night
# --------------------------------------------------------------------------------------
class DemandForecast:
    """Last year's bookings moved 52 weeks forward: the bidders we still expect."""

    def __init__(self, rows: pd.DataFrame, weight: np.ndarray, origin: pd.Timestamp):
        self.table: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if len(rows) == 0:
            return
        arrival = (rows["arrival_date"] + YEAR - origin).dt.days.to_numpy()
        booked = (rows["booking_date"] + YEAR - origin).dt.days.to_numpy()
        nights = rows["nights"].to_numpy(int)
        rep = np.repeat(np.arange(len(rows)), nights)
        offset = np.concatenate([np.arange(n) for n in nights])
        long = pd.DataFrame({"hotel": rows["hotel"].to_numpy()[rep], "night": arrival[rep] + offset,
                             "booked": booked[rep], "w": weight[rep],
                             "rate": rows["nightly_rate"].to_numpy(float)[rep]})
        long = long.sort_values("rate", ascending=False, kind="stable")
        for (hotel, night), g in long.groupby(["hotel", "night"], sort=False):
            self.table[(hotel, int(night))] = (g["booked"].to_numpy(), g["rate"].to_numpy(),
                                               g["w"].to_numpy())

    def price_floor(self, hotel: str, night: int, today: int, free_rooms: float) -> float:
        """Rate of the first future bidder who no longer fits in `free_rooms`."""
        entry = self.table.get((hotel, night))
        if entry is None:
            return 0.0
        booked, rate, w = entry
        future = booked > today
        if not future.any():
            return 0.0
        if free_rooms <= 0:
            return float(rate[future][0])
        expected_guests = np.cumsum(w * future)   # highest rate first
        if expected_guests[-1] <= free_rooms:
            return 0.0                            # everyone still to come fits: no floor
        return float(rate[np.searchsorted(expected_guests, free_rooms, side="right")])


class RecentDemand:
    """The same question -- who is still to come? -- answered from the last four weeks.

    For night N, booked L days ahead, we look at the same weekday 1-4 weeks before today
    (nights whose bookings are all in) and take the requests that came in LATER than L days
    ahead. They are scaled to night N: if N already has twice the bookings those nights had
    at the same point, we expect twice the late bidders; if its bookings are 30% dearer, we
    expect 30% dearer bids. Needs no data from last year, and follows this year's demand.
    """

    def __init__(self):
        self.log = defaultdict(list)       # (hotel, night) -> [(days ahead, rate, P(stay))]
        self._done: dict = {}              # past nights, as arrays sorted by rate

    def observe(self, hotel: str, start: int, nights: int, rate: float, p: float, booked: int) -> None:
        for t in range(start, start + nights):
            self.log[(hotel, t)].append((t - booked, rate, p))

    def _past_night(self, key):
        arrays = self._done.get(key)
        if arrays is None:
            rows = np.array(sorted(self.log.get(key, ()), key=lambda r: -r[1]), dtype=float).reshape(-1, 3)
            arrays = self._done[key] = (rows[:, 0], rows[:, 1], rows[:, 2])
        return arrays

    def price_floor(self, hotel: str, night: int, today: int, free_rooms: float) -> float:
        lead = night - today
        first = lead // 7 + 1                       # nearest same weekday that is already past
        rates, weights, used, had_n, had_rate = [], [], 0, 0, 0.0
        for k in range(first, first + 4):
            ahead, rate, p = self._past_night((hotel, night - 7 * k))
            if len(ahead) == 0:
                continue
            used += 1
            later = ahead < lead
            rates.append(rate[later])
            weights.append(p[later])
            had_n += int((~later).sum())            # what that night had on the books at this point
            had_rate += float(rate[~later].sum())
        if not used:
            return 0.0
        now = self.log.get((hotel, night), ())
        more_demand = float(np.clip((len(now) + 10) / (had_n / used + 10), 0.3, 3.0))
        dearer = 1.0
        if len(now) >= 5 and had_n >= 5:
            dearer = float(np.clip((sum(r[1] for r in now) / len(now)) / (had_rate / had_n), 0.5, 2.5))
        rate = np.concatenate(rates) * dearer
        if len(rate) == 0:
            return 0.0
        order = np.argsort(-rate)
        rate = rate[order]
        if free_rooms <= 0:
            return float(rate[0])
        expected_guests = np.cumsum(np.concatenate(weights)[order]) * more_demand / used
        if expected_guests[-1] <= free_rooms:
            return 0.0
        return float(rate[np.searchsorted(expected_guests, free_rooms, side="right")])


# --------------------------------------------------------------------------------------
# The policy
# --------------------------------------------------------------------------------------
def _normal_sf(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2))


class BookingPolicy:
    def __init__(self, history: pd.DataFrame, commitments: pd.DataFrame,
                 capacity: dict[str, pd.Series], model_kind: str = "combined",
                 demand_multiplier: float = 1.0, late_rule: bool = True,
                 calibration: str = "period", ignore: tuple = (), forecast: str = "blend",
                 boldness: float = 0.0, drift_per_year: float = 1.0):
        history = _clean(history)
        # Both lower the stay chances the policy works with, so it books more guests per room.
        # boldness applies everywhere; drift_per_year grows with the distance from the training
        # data, because the model over-promises more the further ahead it looks. In log-odds.
        self.boldness, self.drift_per_year = boldness, drift_per_year
        self.forecast_mode = forecast   # "last_year", "recent", or "blend" of the two
        commitments = _clean(commitments)
        self.model = StayModel(model_kind, calibration=calibration, ignore=ignore).fit(history)
        self._known: dict = {}   # booking_id -> P(stay), filled by preload() in backtests
        self.late_rule = late_rule

        dates = [pd.Timestamp(d) for s in capacity.values() for d in s.index]
        self.origin = min(dates).normalize()
        last = max(dates).normalize()
        self.size = (last - self.origin).days + 121   # room for stays that run past the end

        # Per hotel and night: rooms, expected guests in house, and the variance of that.
        # stay = expected guests who arrived earlier and are still there (not that day's arrivals).
        # arrivals = per arrival day, the confirmed guests as (-stay value, P(stay)), dearest first.
        # booked = number of confirmed bookings per night (to show how far we overbook).
        self.cap, self.mu, self.var, self.stay, self.arrivals, self.booked = {}, {}, {}, {}, {}, {}
        for hotel, rooms in ROOMS.items():
            cap = np.full(self.size, float(rooms))
            if hotel in capacity:
                s = capacity[hotel]
                idx = (pd.to_datetime(s.index).normalize() - self.origin).days.to_numpy()
                ok = (idx >= 0) & (idx < self.size)
                cap[idx[ok]] = s.to_numpy(float)[ok]
            self.cap[hotel] = cap
            self.mu[hotel] = np.zeros(self.size)
            self.var[hotel] = np.zeros(self.size)
            self.stay[hotel] = np.zeros(self.size)
            self.booked[hotel] = np.zeros(self.size)
            self.arrivals[hotel] = defaultdict(list)
        self._seat_commitments(commitments)

        # Bidders still to come: last year's bookings for the same nights, weighted by their
        # chance to show up and by how much busier (or quieter) this year is so far.
        prev = history[(history["arrival_date"] + YEAR >= self.origin)
                       & (history["arrival_date"] + YEAR <= last)]
        self.pace = {}
        for hotel in ROOMS:
            then = ((prev["hotel"] == hotel) & (prev["booking_date"] + YEAR < self.origin)).sum()
            now = (commitments["hotel"] == hotel).sum()
            self.pace[hotel] = float(np.clip(now / then, 0.6, 1.8)) if then >= 50 else 1.0
        weight = (self.model.predict(prev)
                  * prev["hotel"].map(self.pace).to_numpy(float) * demand_multiplier)
        self.forecast = DemandForecast(prev, weight, self.origin)

        # The second view of future bidders: what the last weeks of requests looked like.
        self.recent = RecentDemand()
        late = history[history["arrival_date"] >= self.origin - pd.Timedelta(days=150)]
        for rows, p in ((late, 1.0 - late["cancelled"].to_numpy(float)),
                        (commitments, self.model.predict(commitments))):
            start = (rows["arrival_date"] - self.origin).dt.days.to_numpy()
            booked = (rows["booking_date"] - self.origin).dt.days.to_numpy()
            for h, a, n, r, pi, b in zip(rows["hotel"], start, rows["nights"], rows["nightly_rate"], p, booked):
                self.recent.observe(h, int(a), int(n), float(r), float(pi), int(b))

    def _price_floor(self, hotel: str, night: int, today: int, free_rooms: float) -> float:
        if self.forecast_mode == "last_year":
            return self.forecast.price_floor(hotel, night, today, free_rooms)
        recent = self.recent.price_floor(hotel, night, today, free_rooms)
        if self.forecast_mode == "recent" or (hotel, night) not in self.forecast.table:
            return recent
        return 0.5 * (recent + self.forecast.price_floor(hotel, night, today, free_rooms))

    def _bolder(self, p, day):
        """Stay chance after the boldness correction, for arrival day `day` (days from the start)."""
        shift = self.boldness + self.drift_per_year * np.maximum(day, 0) / 365.0
        if not np.any(shift):
            return p
        return np.clip(1 / (1 + np.exp(-(_logit(np.clip(p, 0.001, 0.999)) - shift))), 0.001, 0.999)

    def _seat_commitments(self, commitments: pd.DataFrame) -> None:
        """Fill the hotels with the confirmed bookings, day by day. Where they already
        overbook a day, the cheapest arrivals will be walked and so free their later nights."""
        if len(commitments) == 0:
            return
        day = (commitments["arrival_date"] - self.origin).dt.days.to_numpy()
        p = self._bolder(self.model.predict(commitments), day)
        value = (commitments["nightly_rate"] * commitments["nights"]).to_numpy(float)
        arrivals = defaultdict(list)
        for h, d, v, n, pi in zip(commitments["hotel"], day, value, commitments["nights"], p):
            if h in ROOMS and 0 <= d < self.size:
                arrivals[(h, int(d))].append((v, int(n), pi))
        for (hotel, d) in sorted(arrivals, key=lambda k: k[1]):
            mu, var = self.mu[hotel], self.var[hotel]
            free = max(self.cap[hotel][d] - mu[d], 0.0)
            for v, n, pi in sorted(arrivals[(hotel, d)], reverse=True):   # dearest first
                end = min(d + n, self.size)
                self.booked[hotel][d:end] += 1
                bisect.insort(self.arrivals[hotel][d], (-v, pi))
                seated = min(pi, free)
                if seated <= 0:
                    continue
                free -= pi
                mu[d:end] += seated
                var[d:end] += seated * (1 - seated)
                self.stay[hotel][d + 1:end] += seated

    def preload(self, requests: pd.DataFrame) -> None:
        """Backtests only: score a whole (historical) request stream in one batch."""
        self._known = dict(zip(requests["booking_id"], self.model.predict(requests)))

    # -- one request -------------------------------------------------------------------
    def _evaluate(self, b: dict) -> dict:
        hotel = b.get("hotel")
        rate, nights = _num(b.get("nightly_rate")), _num(b.get("nights"))
        if hotel not in ROOMS or math.isnan(rate) or math.isnan(nights):
            return {"accept": False, "p": 0.0, "gain": 0.0, "nights": [], "idx": []}
        p = self._known.get(b.get("booking_id"))
        if p is None:
            p = self.model.predict_one(b)
        start = (pd.Timestamp(b["arrival_date"]).normalize() - self.origin).days
        p = float(self._bolder(p, start))
        today = (pd.Timestamp(b["booking_date"]).normalize() - self.origin).days
        idx = [t for t in range(start, start + max(int(nights), 1)) if 0 <= t < self.size]
        cap, mu, var = self.cap[hotel], self.mu[hotel], self.var[hotel]
        out = {"p": p, "idx": idx, "value": rate * max(int(nights), 1), "outbid": None,
               "seen": (hotel, start, max(int(nights), 1), rate, p, today),
               "period": booking_period(_num(b.get("days_ahead")) if b.get("days_ahead") is not None
                                        else start - today)}
        if not idx:
            return out | {"accept": False, "gain": 0.0, "nights": []}
        risks, floors = [], []
        for t in idx:
            sd = math.sqrt(var[t])
            gap = cap[t] - 0.5 - mu[t]
            risks.append((1.0 if gap < 0 else 0.0) if sd < 1e-9 else _normal_sf(gap / sd))
            floors.append(self._price_floor(hotel, t, today, cap[t] - mu[t]))

        if not self.late_rule:
            # Simple version: on a full night this guest is always the one who is walked.
            gains = [(1 - q) * (rate - f) - q * WALK_PENALTY * rate for q, f in zip(risks, floors)]
        else:
            value = out["value"]
            bump = [self._cheapest_seated(hotel, t) for t in idx]
            # Arrival day: if the hotel is full, does this guest outbid the cheapest arrival?
            outbid = idx[0] == start and bump[0] is not None and value > bump[0]
            q0 = risks[0]
            seated = (1 - q0) + (q0 if outbid else 0.0)
            gains = [seated * rate - (1 - q0) * floors[0]
                     - q0 * (BUMP_COST * bump[0] if outbid else WALK_PENALTY * value)]
            # Later nights: keeping the room either costs a later bidder (the price floor) or,
            # if the hotel is full, bumps the cheapest guest arriving that day.
            for q, f, m in zip(risks[1:], floors[1:], bump[1:]):
                gains.append(seated * (rate - (1 - q) * f - q * BUMP_COST * (m or 0.0)))
            out["outbid"] = bump[0] if outbid else None
        rows = [(t, cap[t], mu[t], q, f, g) for t, q, f, g in zip(idx, risks, floors, gains)]
        total = p * sum(gains)
        return out | {"accept": total > 0, "gain": total, "nights": rows}

    def _cheapest_seated(self, hotel: str, day: int) -> float | None:
        """Stay value of the arrival who loses the room if one more guest needs it that day:
        the cheapest one we still expect to seat. None when nobody (else) arrives that day."""
        arrivals = self.arrivals[hotel].get(day)
        if not arrivals:
            return None
        free = self.cap[hotel][day] - self.stay[hotel][day]   # rooms left for that day's arrivals
        expected = 0.0
        for neg_value, p in arrivals:                          # dearest first
            expected += p
            if expected >= free:
                return -neg_value
        return -arrivals[-1][0]

    def decide(self, booking: dict) -> bool:
        r = self.last = self._evaluate(booking)   # kept so reports can see why
        if "seen" in r:
            self.recent.observe(*r["seen"])        # every request, accepted or not, is demand
        if r["accept"]:
            hotel, idx, p = booking["hotel"], r["idx"], r["p"]
            self.mu[hotel][idx] += p
            self.var[hotel][idx] += p * (1 - p)
            self.stay[hotel][idx[1:]] += p
            self.booked[hotel][idx] += 1
            bisect.insort(self.arrivals[hotel][idx[0]], (-r["value"], p))
        return bool(r["accept"])

    def explain(self, booking: dict) -> str:
        """Plain-language reasoning for one request (does not change any state)."""
        r = self._evaluate(booking)
        rate = _num(booking.get("nightly_rate"))
        lines = [f"Booking {booking.get('booking_id', '?')}: {booking.get('hotel')}, "
                 f"{booking.get('nights')} night(s) from {pd.Timestamp(booking['arrival_date']).date()} "
                 f"at {rate:.0f} per night, booked in the {r.get('period', '?')} period",
                 f"  Chance the guest shows up: {r['p']:.0%}"]
        for t, cap, mu, risk, floor, gain in r["nights"]:
            date = (self.origin + pd.Timedelta(days=t)).date()
            lines.append(f"  {date}: {mu:5.1f} of {cap:.0f} rooms expected taken, "
                         f"risk of being full {risk:4.0%}, price floor {floor:4.0f} "
                         f"-> worth {gain:+7.1f}")
        if r.get("outbid") is not None and r["nights"][0][3] >= 0.01:   # only when it matters
            lines.append(f"  If the arrival day is full, this guest (stay value {r['value']:.0f}) outbids "
                         f"the cheapest arrival (stay value {r['outbid']:.0f}), who is walked instead")
        lines.append(f"  Expected gain {r['gain']:+.1f} -> {'ACCEPT' if r['accept'] else 'REJECT'}")
        return "\n".join(lines)


def load_history(path: Path = TRAIN_CSV) -> pd.DataFrame:
    return pd.read_csv(path)
