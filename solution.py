"""Your team's solution. At the end of the hackathon you pass this class to evaluate(Solution)
from evaluate.py, which creates one Solution and replays every 2017 booking request through it.

RULES -- Downtown Hotel has 124 rooms, Seaside Hotel 103. A guest who shows up earns
nightly_rate x nights. When a hotel is full at check-in, the cheapest arriving guests are
walked: they earn nothing and cost 2.5 x their stay value. A seated guest keeps the room.
Your result is the net revenue: revenue of seated guests - walk penalties.

Reading any 2017 data yourself is cheating.

HOW THIS SOLUTION DECIDES
1. Every booking counts as a fraction of a guest: its show-up probability, predicted by a
   logistic regression trained on the 2015-2016 bookings (train_show_model.py). It does not
   use the deposit column, which is not available in the evaluation.
2. For every hotel and night we keep a ledger: the expected number of guests in house.
3. A request is accepted only if, on every night of its stay, the expected guests stay within
   the room budget (ALPHA x free rooms).
4. We also hold rooms for better-paying guests who have not booked yet. From past years we
   know how many guests usually want a room on such a night and how late they book
   (demand_forecast.py); if a night is filling faster or slower than usual we scale that up or
   down. A request must leave room for the share of those future guests who would pay more
   than it does. The hold shrinks to zero as the night approaches.
"""
import bisect
from pathlib import Path

import numpy as np
import pandas as pd

from demand_forecast import MAX_LEAD, build_profile
from train_show_model import fast_scorer, features, make_model

ROOMS = {"Downtown Hotel": 124, "Seaside Hotel": 103}

# --- Policy settings (tuned with backtest.py on a replay of 2016) ---------------------------
ALPHA = {"Downtown Hotel": 1.10, "Seaside Hotel": 1.10}  # room budget as a share of free rooms
HOLD = 1.0           # strength of the hold for better-paying future guests (0 = off)
PACE_WEIGHT = 20.0   # how fast we trust this year's booking pace over last year's level (0 = off)
MIN_RATE = 1.0       # requests cheaper than this per night earn nothing: reject
HORIZON_DAYS = 366 + 150  # ledger length: the year plus stays that run past 31 December

TRAIN_FILE = Path(__file__).resolve().parent / "train.csv"   # 2015-2016 bookings only


class Solution:
    model = None    # backtest.py injects a model and a demand profile built from earlier data only
    profile = None

    def __init__(self, commitments: pd.DataFrame, capacity: dict[str, pd.Series]) -> None:
        """Called once, before 2017 starts.

        commitments  2017 arrivals booked before 2017-01-01. The hotel already confirmed them, so
                     decide() is never called for them. Same columns as train.csv, except:
                     - no cancelled column;
                     - arrival_date and booking_date are already pandas Timestamps, not strings.

        capacity     hotel -> rooms free on each night of 2017 (indexed by date): the rooms minus
                     guests who arrived in 2016 and are still in house.
        """
        # Everything we learn comes from train.csv (2015-2016), read once here (about 1 second).
        history = None
        if Solution.model is None or Solution.profile is None:
            history = pd.read_csv(TRAIN_FILE)
        model = Solution.model
        if model is None:
            model = make_model().fit(features(history), 1 - history["cancelled"])
        self._score = fast_scorer(model)

        # Room budget per hotel and night, as arrays indexed by days since the first night.
        self._day0 = min(pd.Timestamp(s.index.min()) for s in capacity.values()).normalize()
        self._budget, self._load = {}, {}
        for hotel, free in capacity.items():
            rooms = np.full(HORIZON_DAYS, float(ROOMS.get(hotel, free.max())))
            idx = (pd.DatetimeIndex(free.index).normalize() - self._day0).days
            ok = (idx >= 0) & (idx < HORIZON_DAYS)
            rooms[idx[ok]] = np.clip(free.to_numpy(dtype=float)[ok], 0, None)
            self._budget[hotel] = rooms * ALPHA.get(hotel, 1.0)
            self._load[hotel] = np.zeros(HORIZON_DAYS)

        # Demand profile from past years: how busy each night usually is, and how late guests book.
        profile = Solution.profile if Solution.profile is not None else build_profile(history)
        self._to_come, self._usual, self._asked = {}, {}, {}
        months = (self._day0 + pd.to_timedelta(np.arange(HORIZON_DAYS), unit="D")).month
        for hotel in self._load:
            self._to_come[hotel] = profile["still_to_come"].get(hotel, np.zeros(MAX_LEAD + 1))
            fallback = profile["hotel_typical"].get(hotel, 0.0)
            self._usual[hotel] = np.array([profile["typical"].get((hotel, m), fallback) for m in months])
            self._asked[hotel] = np.zeros(HORIZON_DAYS)   # expected guests requested so far, accepted or not

        # Typical nightly rate per hotel and arrival month, learned from the requests themselves
        # (so it follows 2017 prices). Seeded with the commitments.
        self._rates = {}
        self.accepted = self.rejected = 0

        if len(commitments):
            p = model.predict_proba(features(commitments))[:, 1]
            for hotel, arrival, nights, rate, prob in zip(
                    commitments["hotel"], commitments["arrival_date"], commitments["nights"],
                    commitments["nightly_rate"], p):
                if hotel not in self._load:
                    continue
                a, b = self._span(arrival, nights)
                self._load[hotel][a:b] += prob
                self._asked[hotel][a:b] += prob
                self._remember_rate(hotel, arrival, rate)

    # ------------------------------------------------------------------------------------
    def _span(self, arrival, nights):
        a = (pd.Timestamp(arrival).normalize() - self._day0).days
        a = min(max(a, 0), HORIZON_DAYS - 1)
        return a, min(a + max(int(nights), 1), HORIZON_DAYS)

    def _remember_rate(self, hotel, arrival, rate):
        if rate is not None and rate == rate and rate >= MIN_RATE:
            bisect.insort(self._rates.setdefault((hotel, pd.Timestamp(arrival).month), []), float(rate))

    def _share_paying_more(self, hotel, arrival, rate):
        """Among requests seen so far for this hotel and arrival month, the share with a higher rate."""
        rates = self._rates.get((hotel, pd.Timestamp(arrival).month), [])
        if len(rates) < 30:
            return 0.0
        return 1.0 - bisect.bisect_right(rates, rate) / len(rates)

    def _hold(self, hotel, a, b, today, share_paying_more):
        """Rooms to keep free on nights a..b for better-paying guests who have not booked yet."""
        days_left = np.clip(np.arange(a, b) - today, 0, MAX_LEAD)
        to_come = self._to_come[hotel][days_left]            # share of a night's guests still to book
        usual = self._usual[hotel][a:b]                      # guests usually wanting a room that night
        if PACE_WEIGHT > 0:
            # Is this night filling faster or slower than usual? Scale the forecast accordingly.
            expected_by_now = usual * (1 - to_come)
            pace = (self._asked[hotel][a:b] + PACE_WEIGHT) / (expected_by_now + PACE_WEIGHT)
            usual = usual * np.clip(pace, 0.5, 2.0)
        return HOLD * usual * to_come * share_paying_more

    def decide(self, booking: dict) -> bool:
        """Called once per 2017 booking request, in booking-date order. `booking` is one row with
        the same columns as commitments, as {column: value}. Return True to confirm the booking,
        False to reject it."""
        try:
            return self._decide(booking)
        except Exception:
            return False  # never crash the evaluation on an odd row

    def _decide(self, booking: dict) -> bool:
        hotel, rate = booking["hotel"], float(booking["nightly_rate"])
        if hotel not in self._load or not rate >= MIN_RATE:
            self.rejected += 1
            return False

        p = self._score(booking)                       # this booking = p of a guest
        a, b = self._span(booking["arrival_date"], booking["nights"])
        today = (pd.Timestamp(booking["booking_date"]).normalize() - self._day0).days
        paying_more = self._share_paying_more(hotel, booking["arrival_date"], rate)
        hold = self._hold(hotel, a, b, today, paying_more)

        self._remember_rate(hotel, booking["arrival_date"], rate)
        self._asked[hotel][a:b] += p                   # count every request, accepted or not

        budget = self._budget[hotel][a:b]
        load = self._load[hotel][a:b]                  # view: updating it updates the ledger

        if np.all(load + p + hold <= budget):
            load += p
            self.accepted += 1
            return True
        self.rejected += 1
        return False
