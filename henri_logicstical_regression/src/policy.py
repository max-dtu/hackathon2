"""Sequential booking control; no realized outcomes are used here."""
import math
from functools import lru_cache

import pandas as pd


def stay_nights(booking):
    start = pd.Timestamp(booking['arrival_date']).normalize()
    return _night_range(start, max(0, int(booking['nights'])))


@lru_cache(maxsize=20000)
def _night_range(start, nights):
    return tuple(pd.date_range(start, periods=nights, freq='D'))


class BookingPolicy:
    def __init__(self, commitments, capacity, predict, mode='expected', margin=0.0, allowance=0,
                 allowance_fraction=0.0, protection=0.0, demand=None, hotel_settings=None,
                 walk_cost=0.0):
        self.capacity = {h: {pd.Timestamp(d).normalize(): float(v) for d, v in s.items()}
                         for h, s in capacity.items()}
        self.predict = predict
        self.mode, self.margin, self.allowance = mode, margin, allowance
        self.allowance_fraction, self.protection = allowance_fraction, protection
        demand = demand.session(commitments) if hasattr(demand, "session") else demand
        self.demand, self.hotel_settings = demand, hotel_settings or {}
        self.walk_cost = walk_cost
        if (protection or walk_cost) and demand is None:
            raise ValueError('Price protection and walk cost require a fitted demand forecast')
        self.room_counts = {h: float(s.max()) for h, s in capacity.items()}
        self.mean = {h: {} for h in capacity}
        self.variance = {h: {} for h in capacity}
        for booking in commitments.to_dict('records'):
            self._add(booking, self._probability(booking))

    def _probability(self, booking):
        if self.mode != 'expected':
            return 1.0
        probability = 1.0 - float(self.predict(booking))
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Invalid arrival probability')
        return probability

    def _add(self, booking, probability):
        hotel = booking['hotel']
        for day in stay_nights(booking):
            self.mean[hotel][day] = self.mean[hotel].get(day, 0) + probability
            self.variance[hotel][day] = self.variance[hotel].get(day, 0) + probability * (1 - probability)

    def decide(self, booking):
        try:
            return self._decide(booking)
        finally:
            if hasattr(self.demand, "observe"):
                self.demand.observe(booking)

    def _decide(self, booking):
        if self.mode == 'reject':
            return False
        if booking['hotel'] not in self.capacity:
            return False
        nights = stay_nights(booking)
        if not len(nights):
            return False
        probability = self._probability(booking)
        hotel = booking['hotel']
        settings = self.hotel_settings.get(hotel, {})
        fraction = settings.get('allowance_fraction', self.allowance_fraction)
        protection = settings.get('protection', self.protection)
        margin = settings.get('margin', self.margin)
        walk_cost = settings.get('walk_cost', self.walk_cost)
        if (protection or walk_cost) and self.demand is None:
            raise ValueError('Price protection and walk cost require a fitted demand forecast')
        allowance = self.allowance + fraction * self.room_counts[hotel]
        value = float(booking['nightly_rate']) * len(nights)
        if walk_cost and self.mode == 'expected':
            return self._decide_by_net_revenue(
                booking, nights, probability, allowance, protection, walk_cost, value)
        if self.mode != 'all':
            for day in nights:
                if day not in self.capacity[hotel]:
                    return False
                mean = self.mean[hotel].get(day, 0) + probability
                variance = self.variance[hotel].get(day, 0) + probability * (1 - probability)
                if mean + margin * math.sqrt(variance) > self.capacity[hotel][day] + allowance:
                    return False
        if protection:
            required = sum(self.demand.reservation_price(
                hotel, day, booking['booking_date'],
                self.capacity[hotel][day] + allowance - self.mean[hotel].get(day, 0),
            ) for day in nights)
            if value < protection * required:
                return False
        self._add(booking, probability)
        return True

    def _decide_by_net_revenue(self, booking, nights, probability, allowance, protection,
                               walk_cost, value):
        """Accept when the stay is worth more than the harm it is expected to cause.

        Instead of a hard occupancy cap, each night is priced. A full night means a
        guest is walked: the hotel loses that guest's revenue and pays 2.5 times it,
        3.5 times in total. The cheapest arrivals are walked first, so the cost uses
        a typical cheap rate, weighted by the chance that the night is full. A night
        that is probably free instead costs what a later guest would likely pay.
        """
        hotel = booking['hotel']
        cost = 0.0
        for day in nights:
            if day not in self.capacity[hotel]:
                return False
            rooms = self.capacity[hotel][day] + allowance
            mean = self.mean[hotel].get(day, 0)
            spread = math.sqrt(
                self.variance[hotel].get(day, 0) + probability * (1 - probability)) + 1e-6
            # Chance that the other expected guests already fill the hotel that night.
            full = 0.5 * math.erfc((rooms - mean - 0.5) / (spread * math.sqrt(2)))
            cost += 3.5 * walk_cost * self.demand.low_rate(hotel, day) * full
            if protection:
                cost += protection * self.demand.reservation_price(
                    hotel, day, booking['booking_date'], rooms - mean)
        if value < cost:
            return False
        self._add(booking, probability)
        return True
