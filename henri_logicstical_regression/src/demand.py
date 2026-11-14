"""Historical future room-night demand, fitted only on matured past stays."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

LEAD_UPPER = (3, 7, 14, 30, 60, 90, 180, 365, float('inf'))


class FutureDemand:
    def __init__(self, curves, cutoff):
        self.curves = curves
        self.cutoff = str(pd.Timestamp(cutoff).date())

    @classmethod
    def fit(cls, data, cutoff, min_dates=20):
        cutoff = pd.Timestamp(cutoff).normalize()
        # Never use outcomes whose stay has not completed by the cutoff.
        history = data[(data.checkout < cutoff) & (data.nights > 0)].copy()
        curves = {}
        for hotel, rows in history.groupby('hotel'):
            calendar = pd.date_range(rows.arrival_date.min(), cutoff - pd.Timedelta(days=1))
            nights, prices, leads = [], [], []
            arrived = rows[(rows.cancelled == 0) & (rows.nightly_rate > 0)]
            for b in arrived.to_dict('records'):
                stay = pd.date_range(b['arrival_date'], periods=int(b['nights']))
                nights.extend(stay)
                prices.extend([float(b['nightly_rate'])] * len(stay))
                leads.extend((stay - b['booking_date']).days)
            observations = pd.DataFrame({'night': pd.to_datetime(nights), 'price': prices, 'lead': leads})
            for month in range(1, 13):
                months = [month]
                fallback = 'month'
                dates = calendar[calendar.month.isin(months)]
                if len(dates) < min_dates:
                    months = [(month - 2) % 12 + 1, month, month % 12 + 1]
                    dates = calendar[calendar.month.isin(months)]
                    fallback = 'adjacent_months'
                if len(dates) < min_dates:
                    dates = calendar
                    fallback = 'hotel'
                sample = observations[observations.night.dt.month.isin(set(dates.month))]
                for band, upper in enumerate(LEAD_UPPER):
                    # A request booked later has a smaller service-night lead time.
                    rates = np.sort(sample.loc[(sample.lead >= 0) & (sample.lead < upper), 'price'].to_numpy())[::-1]
                    curves[(hotel, month, band)] = (rates, len(dates), fallback)
        return cls(curves, cutoff)

    def reservation_price(self, hotel, night, booking_date, free_rooms):
        lead = (pd.Timestamp(night) - pd.Timestamp(booking_date)).days
        if lead <= 0:
            return 0.0
        band = int(np.searchsorted(LEAD_UPPER, lead, side='left'))
        curve = self.curves.get((hotel, pd.Timestamp(night).month, band))
        if curve is None:
            return 0.0
        rates, exposure, _ = curve
        if not len(rates) or not exposure or len(rates) / exposure <= max(0, free_rooms):
            return 0.0
        marginal = min(len(rates) - 1, int(max(0, free_rooms) * exposure))
        return float(rates[marginal])

    def low_rate(self, hotel, night, quantile=0.25):
        """Typical cheap nightly rate: the guests walked first are the cheapest."""
        curve = self.curves.get((hotel, pd.Timestamp(night).month, len(LEAD_UPPER) - 1))
        if curve is None or not len(curve[0]):
            return 0.0
        rates = curve[0]  # sorted from most to least expensive
        return float(rates[min(len(rates) - 1, int((1 - quantile) * len(rates)))])

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
