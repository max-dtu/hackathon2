"""Learn remaining demand above a price from matured historical nights."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from src.demand import FutureDemand
from src.policy import stay_nights

class LearnedDemand:
    @classmethod
    def fit(cls, data, cutoff):
        obj = cls()
        cutoff = pd.Timestamp(cutoff)
        known = data[(data.booking_date < cutoff) & (data.nights > 0)]
        mature = known[known.checkout < cutoff]
        obj.empirical = FutureDemand.fit(data, cutoff)
        obj.prices, obj.history, obj.cache = {}, {}, {}
        x, y = [], []
        for hotel, rows in mature.groupby('hotel'):
            rates = rows.loc[rows.nightly_rate > 0, 'nightly_rate']
            if rates.empty: continue
            eligible = []
            for night in pd.date_range(rows.arrival_date.min(), cutoff-pd.Timedelta(days=1)):
                covering = known[(known.hotel == hotel) & (known.arrival_date <= night) & (known.checkout > night)]
                if not (covering.checkout >= cutoff).any(): eligible.append((night, covering))
            if len(eligible) < 60: continue
            obj.prices[hotel] = np.unique(np.r_[0, rates.quantile(np.linspace(.1, 1, 10))])
            for night, covering in eligible:
                for lead in (1,3,7,14,30,60,90,180,365):
                    date = night-pd.Timedelta(days=lead)
                    if date < rows.arrival_date.min(): continue
                    past = covering[(covering.booking_date < date) & (covering.arrival_date >= date)]
                    counts = [len(past), (past.booking_date >= date-pd.Timedelta(days=7)).sum(), (past.booking_date >= date-pd.Timedelta(days=28)).sum()]
                    future = covering[(covering.booking_date >= date) & (covering.cancelled == 0) & (covering.nightly_rate > 0)]
                    for price in obj.prices[hotel]:
                        x.append([hotel,night.month,night.dayofweek,lead,*counts,price])
                        y.append(int((future.nightly_rate >= price).sum()))
        obj.model = None
        if x and any(y):
            obj.model = CatBoostRegressor(iterations=500,depth=6,learning_rate=.05,loss_function='Poisson',random_seed=42,thread_count=4,verbose=False,allow_writing_files=False,cat_features=[0])
            obj.model.fit(x,y)
        obj.training_rows = len(x)
        return obj

    def session(self, commitments):
        import copy
        result = copy.copy(self)
        result.history, result.cache, result.fallbacks = {}, {}, 0
        for booking in commitments.to_dict('records'): result.observe(booking)
        return result

    def observe(self, booking):
        for night in stay_nights(booking):
            self.history.setdefault((booking['hotel'],night),[]).append((pd.Timestamp(booking['booking_date']).normalize(),pd.Timestamp(booking['arrival_date']).normalize()))

    def low_rate(self,*args,**kwargs):
        return self.empirical.low_rate(*args,**kwargs)

    def reservation_price(self,hotel,night,booking_date,free_rooms):
        night,date = pd.Timestamp(night).normalize(),pd.Timestamp(booking_date).normalize()
        if night <= date: return 0.
        if self.model is None or hotel not in self.prices:
            self.fallbacks += 1
            return self.empirical.reservation_price(hotel,night,date,free_rooms)
        key = hotel,night,date
        if key not in self.cache:
            leads = [(date-booked).days for booked,arrival in self.history.get((hotel,night),[]) if booked < date and arrival >= date]
            counts = [len(leads),sum(v<=7 for v in leads),sum(v<=28 for v in leads)]
            x = [[hotel,night.month,night.dayofweek,(night-date).days,*counts,p] for p in self.prices[hotel]]
            self.cache[key] = np.minimum.accumulate(np.maximum(0,self.model.predict(x,prediction_type='Exponent',thread_count=1)))
        prices = self.prices[hotel][self.cache[key] > max(0,free_rooms)]
        return float(prices[-1]) if len(prices) else 0.
