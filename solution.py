"""Your team's solution. At the end of the hackathon you pass this class to evaluate(Solution)
from evaluate.py, which creates one Solution and replays every 2017 booking request through it.

RULES -- Downtown Hotel has 124 rooms, Seaside Hotel 103. A guest who shows up earns
nightly_rate x nights. When a hotel is full at check-in, the cheapest arriving guests are
walked: they earn nothing and cost 2.5 x their stay value. A seated guest keeps the room.
Your result is the net revenue: revenue of seated guests - walk penalties.

Reading any 2017 data yourself is cheating.

HOW IT WORKS (two steps per booking request)
  1. Show-up model:  p = probability this guest actually arrives (gradient-boosted trees trained
     on past bookings, then calibrated on the most recent 90 days so 70% really means 70%).
  2. Decision rule:  q = probability the hotel is already full on a night of the stay, from the
     show-up probabilities of every booking already confirmed for that night. If it overflows,
     the cheapest guest arriving that day is walked (earns 0, costs 2.5x their stay value):
       - if we are the cheapest, we are walked:      gain = (1-q) * V  -  q * 2.5 * V
       - otherwise we bump the cheapest guest (C):   gain = V  -  q * 3.5 * C
     minus the same bump risk on later nights of the stay.
  3. Bid price (saving rooms for better bookings): from the same nights last year, estimate how
     many guests will still book after today and what they pay. Each night of the stay costs
     P(hotel would fill anyway) x that later rate. Accept if p x (gain - bid price) >= 0.
"""
import bisect
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

WALK_PENALTY = 2.5
CALIB_DAYS = 90

# Features known when the request arrives. Deliberately excluded:
#   deposit            -- dropped by the organisers (and near-perfectly leaks cancellation)
#   past_cancellations -- meaning changed between years (2015: 99% cancel, 2016: 20%)
#   arrival month/week/day/year -- seasonality the training window can't cover reliably
CAT = ["hotel", "meal", "country", "sales_segment", "sales_channel", "room_type", "guest_type",
       "agent", "company"]
NUM = ["days_ahead", "weekend_nights", "weekday_nights", "adults", "children", "babies",
       "returning_guest", "past_stays", "amendments", "waitlist_days", "nightly_rate",
       "special_requests", "nights"]


class ShowUpModel:
    """P(guest shows up | booking). Gradient-boosted trees + cross-fitted Platt calibration."""

    def __init__(self, history: pd.DataFrame):
        history = history.sort_values("arrival_date")
        cut = history.arrival_date.max() - pd.Timedelta(days=CALIB_DAYS)
        early = history[history.arrival_date <= cut]
        recent = history[(history.arrival_date > cut) & (history.booking_date > cut)]

        # Calibration: a model that never saw the last 90 days, scored on the last 90 days,
        # tells us how over/under-confident the trees are on fresh bookings.
        probe = self._fit(early)
        raw = self._raw(probe, recent)
        self.calib = LogisticRegression().fit(self._logit(raw)[:, None], 1 - recent.cancelled)

        self.trees = self._fit(history)  # final model uses everything

    # -- internals ---------------------------------------------------------------------------
    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    def _fit(self, df):
        maps = {c: {v: i for i, v in enumerate(df[c].astype(str).value_counts().index[:200])}
                for c in CAT}
        clf = HistGradientBoostingClassifier(
            categorical_features=list(range(len(NUM), len(NUM) + len(CAT))),
            max_iter=200, learning_rate=0.05, random_state=0)
        clf.fit(self._matrix(df, maps), 1 - df.cancelled)
        return clf, maps

    @staticmethod
    def _matrix(df, maps):
        num = df[NUM].astype(float).to_numpy()
        cat = np.column_stack([df[c].astype(str).map(maps[c]).fillna(-1).to_numpy() for c in CAT])
        return np.hstack([num, cat])

    def _raw(self, model, df):
        clf, maps = model
        return clf.predict_proba(self._matrix(df, maps))[:, 1]

    # -- public ------------------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raw = self._raw(self.trees, df)
        return self.calib.predict_proba(self._logit(raw)[:, None])[:, 1]

    def predict_one(self, b: dict) -> float:
        clf, maps = self.trees
        row = [float(b[c]) if b[c] is not None and b[c] == b[c] else np.nan for c in NUM]
        row += [maps[c].get(str(b[c]), -1) for c in CAT]
        raw = clf.predict_proba(np.array([row], dtype=float))[:, 1]
        return float(self.calib.predict_proba(self._logit(raw)[:, None])[0, 1])


class PickupForecast:
    """Bookings still to come for each night, read off the same night 52 weeks earlier (same
    weekday): of last year's bookings for that night, which were made closer to the night than
    'now' is, how many of those guests showed up, and what they paid per night."""

    def __init__(self, history: pd.DataFrame, capacity: dict, commitments: pd.DataFrame = None):
        self.start, self.table, self.growth = {}, {}, {}
        for h, cap in capacity.items():
            # Booking pace: this period's bookings already on the books vs. last year's at the
            # same point. 2016 ran well ahead of 2015, so copying last year 1:1 undershoots.
            g = 1.0
            if commitments is not None and len(commitments):
                ly0 = cap.index[0] - pd.Timedelta(days=364)
                ly1 = ly0 + pd.Timedelta(days=len(cap))
                last = ((history.hotel == h) & (history.arrival_date >= ly0) &
                        (history.arrival_date < ly1) & (history.booking_date < ly0)).sum()
                if last > 0:
                    g = float(np.clip((commitments.hotel == h).sum() / last, 0.5, 2.0))
            self.growth[h] = g
            hist = history[history.hotel == h]
            n = hist.nights.to_numpy().astype(int)
            idx = np.repeat(np.arange(len(hist)), n)
            offset = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)  # 0..nights-1 per stay
            night = hist.arrival_date.to_numpy()[idx] + offset.astype("timedelta64[D]")
            lead = (night - hist.booking_date.to_numpy()[idx]).astype("timedelta64[D]").astype(int)
            df = pd.DataFrame({"night": night, "lead": lead,
                               "show": 1 - hist.cancelled.to_numpy()[idx],
                               "rate": hist.nightly_rate.to_numpy()[idx]}).sort_values("lead")
            df["show_rate"] = df.show * df.rate
            table = {}
            for night_date, g in df.groupby("night"):
                table[night_date] = (g.lead.to_numpy(), np.r_[0, g.show.cumsum()],
                                     np.r_[0, g.show_rate.cumsum()])
            self.start[h], self.table[h] = cap.index[0], table

    def still_to_come(self, h, night_idx, lead_days):
        """(expected future shows, their average nightly rate) for one night of the period."""
        key = self.start[h] + pd.Timedelta(days=night_idx - 364)
        if key not in self.table[h]:
            return 0.0, 0.0
        leads, cum_show, cum_rev = self.table[h][key]
        k = np.searchsorted(leads, lead_days, side="left")  # booked closer to the night than now
        shows = float(cum_show[k])
        return shows * self.growth[h], (float(cum_rev[k]) / shows if shows else 0.0)


class Solution:
    def __init__(self, commitments: pd.DataFrame, capacity: dict, history: pd.DataFrame = None,
                 p_shift: float = -0.4, walk_model: str = "cheapest", model: ShowUpModel = None,
                 bid_scale: float = 1.0, var_scale: float = 2.0) -> None:
        """history:    bookings with known outcomes. The judges' evaluate.py doesn't pass it, so we
                       read train.csv; simulate.py passes an earlier slice to replay past years.
        p_shift:    safety knob, added to every show-up probability on the log-odds scale.
                    Negative = assume more no-shows (book more aggressively).
        walk_model: "cheapest" = real rule (cheapest arrival that day is walked);
                    "simple"   = assume this guest is the one walked (2.5 x own value).
        model:      a pre-trained ShowUpModel, so parameter sweeps don't retrain.
        bid_scale:  weight on the bid price (room held back for later, better-paying guests);
                    0 = first-come first-served.
        var_scale:  widens the uncertainty of each night's head count. Cancellations are not
                    independent (a group cancels together), so occupancy swings more than
                    independent coin flips suggest."""
        if history is None:
            history = pd.read_csv(Path(__file__).with_name("train.csv"),
                                  parse_dates=["arrival_date", "booking_date"])
        history = history.drop(columns=["deposit"], errors="ignore")
        if model is None:
            model = ShowUpModel(history)
        self.model, self.p_shift, self.walk_model = model, p_shift, walk_model
        self.bid_scale, self.var_scale = bid_scale, var_scale
        self.pickup = PickupForecast(history, capacity, commitments) if bid_scale else None
        # One-row predictions are ~30x faster single-threaded (thread start-up dominates).
        threadpool_limits(1, user_api="openmp")

        # Running forecast per hotel and night: expected shows (mu) and their variance, plus the
        # cheapest confirmed guest arriving each day (the one who'd be walked if it overflows).
        self.start = {h: s.index[0] for h, s in capacity.items()}
        self.cap = {h: s.to_numpy().astype(float) for h, s in capacity.items()}
        self.mu = {h: np.zeros(len(c)) for h, c in self.cap.items()}
        self.var = {h: np.zeros(len(c)) for h, c in self.cap.items()}
        self.arrivals = {h: [[] for _ in c] for h, c in self.cap.items()}  # sorted (value, p)
        if len(commitments):
            for b, p in zip(commitments.to_dict("records"), self.model.predict(commitments)):
                self._add(b, self._shift(p))

    def _shift(self, p):
        if self.p_shift == 0:
            return p
        return 1 / (1 + np.exp(-(ShowUpModel._logit(p) + self.p_shift)))

    def _span(self, b):
        h = b["hotel"]
        lo = (pd.Timestamp(b["arrival_date"]) - self.start[h]).days
        return h, lo, slice(max(lo, 0), min(lo + int(b["nights"]), len(self.cap[h])))

    def _add(self, b, p):
        h, lo, s = self._span(b)
        self.mu[h][s] += p
        self.var[h][s] += p * (1 - p) * self.var_scale
        if 0 <= lo < len(self.cap[h]):
            bisect.insort(self.arrivals[h][lo], (b["nightly_rate"] * b["nights"], p))

    def _full_risk(self, h, s):
        """Per night: q = P(already-confirmed guests who show up fill every room), and the
        expected number of guests already over capacity, E[max(shows - rooms, 0)]."""
        mu, sd, cap = self.mu[h][s], np.sqrt(self.var[h][s]), self.cap[h][s]
        q, over = np.zeros(len(mu)), np.zeros(len(mu))
        for k, (m, d, c) in enumerate(zip(mu, sd, cap)):
            if d > 0:
                z = (m - (c - 0.5)) / d  # continuity correction
                q[k] = 0.5 * math.erfc(-z / math.sqrt(2))
                over[k] = max(m - c, 0) if z > 8 else (m - c) * q[k] + d * math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
            else:
                q[k], over[k] = float(m >= c), max(m - c, 0)
        return q, over

    def _victim(self, h, day, already_over):
        """Stay value of the next guest to be walked on `day`: the cheapest arrivals are walked
        first, so skip the (expected) number already over capacity. None if nobody is left."""
        mass = 0.0
        for value, p in self.arrivals[h][day]:
            mass += p
            if mass > already_over:
                return value
        return None

    def explain(self, b: dict) -> dict:
        """Everything behind one decision -- handy for the presentation."""
        p = float(self._shift(self.model.predict_one(b)))
        v = b["nightly_rate"] * b["nights"]
        h, lo, s = self._span(b)
        q, over = self._full_risk(h, s)
        if len(q) == 0:
            return dict(p_show=p, q_full=0.0, stay_value=v, expected_value=p * v, accept=True)

        if self.walk_model == "simple":
            qmax = float(q.max())
            ev = p * ((1 - qmax) * v - qmax * WALK_PENALTY * v)
            return dict(p_show=p, q_full=qmax, stay_value=v, expected_value=ev, accept=ev >= 0)

        # Arrival day: if the hotel overflows, the next-cheapest arrival that day is walked.
        q0 = float(q[0]) if lo >= 0 else 0.0
        c0 = self._victim(h, s.start, over[0]) if lo >= 0 else None
        if c0 is None or v <= c0:   # we'd be the one walked
            ev_arrive, seated = (1 - q0) * v - q0 * WALK_PENALTY * v, 1 - q0
        else:                       # we bump that guest: lose their stay + pay 2.5x it
            ev_arrive, seated = v - q0 * (1 + WALK_PENALTY) * c0, 1.0
        # Later nights: our occupied room can force a walk among that day's arrivals.
        later = 0.0
        for k in range(1, len(q)):
            c = self._victim(h, s.start + k, over[k])
            if c is not None:
                later = max(later, q[k] * (1 + WALK_PENALTY) * c)
        bid = self._bid_price(h, s, b) if self.pickup else 0.0
        ev = p * (ev_arrive - seated * (later + bid))
        return dict(p_show=p, q_full=q0, stay_value=v, bumps_value=c0, bid_price=bid,
                    expected_value=ev, accept=ev >= 0)

    def _bid_price(self, h, s, b) -> float:
        """Value of the later bookings this stay is likely to crowd out: per night, the chance
        that confirmed + still-to-come guests fill the hotel x what those later guests pay."""
        t = pd.Timestamp(b["booking_date"])
        total = 0.0
        for i in range(s.start, s.stop):
            lead = (self.start[h] + pd.Timedelta(days=i) - t).days
            shows, rate = self.pickup.still_to_come(h, i, lead)
            if shows <= 0:
                continue
            m, sd = self.mu[h][i] + shows, math.sqrt(self.var[h][i] + shows)
            fills = 0.5 * math.erfc(-(m - (self.cap[h][i] - 0.5)) / (sd * math.sqrt(2)))
            total += fills * rate
        return self.bid_scale * total

    def decide(self, booking: dict) -> bool:
        e = self.explain(booking)
        if e["accept"]:
            self._add(booking, e["p_show"])
        return bool(e["accept"])
