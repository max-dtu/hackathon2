"""Training and inference share exactly the same booking transformations."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = [
    'days_ahead', 'weekend_nights', 'weekday_nights', 'adults', 'children',
    'babies', 'returning_guest', 'past_cancellations', 'past_stays',
    'amendments', 'waitlist_days', 'nightly_rate', 'special_requests', 'nights',
    'arrival_month_number', 'arrival_weekday', 'arrival_day', 'arrival_week',
    'booking_month', 'booking_weekday',
]
CATEGORICAL = [
    'hotel', 'meal', 'country', 'sales_segment', 'sales_channel', 'room_type',
    'agent', 'company', 'guest_type',
]


def features(bookings):
    frame = pd.DataFrame([bookings]) if isinstance(bookings, dict) else bookings.copy()
    arrival = pd.to_datetime(frame['arrival_date'])
    booked = pd.to_datetime(frame['booking_date'])
    frame['arrival_month_number'] = arrival.dt.month
    frame['arrival_weekday'] = arrival.dt.day_of_week
    frame['arrival_day'] = arrival.dt.day
    frame['arrival_week'] = arrival.dt.isocalendar().week.astype(float)
    frame['booking_month'] = booked.dt.month
    frame['booking_weekday'] = booked.dt.day_of_week
    for name in NUMERIC:
        frame[name] = pd.to_numeric(frame[name], errors='coerce') if name in frame else np.nan
    for name in CATEGORICAL:
        values = frame[name] if name in frame else pd.Series(np.nan, index=frame.index)
        if name in ('agent', 'company'):
            # CSV IDs such as 304.0 and inference IDs such as 304 are identical.
            values = pd.to_numeric(values, errors='coerce').map(
                lambda v: str(float(v)) if pd.notna(v) else np.nan
            )
        frame[name] = values.map(lambda v: str(v) if pd.notna(v) and str(v) else 'Unknown')
    return frame[NUMERIC + CATEGORICAL]


def make_model(kind='logistic'):
    if kind == 'catboost':
        return Pipeline([('classifier', CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05, loss_function='Logloss',
            random_seed=42, thread_count=4, verbose=False, allow_writing_files=False,
            cat_features=CATEGORICAL,
        ))])
    if kind not in ('logistic', 'forest'):
        raise ValueError(f'Unknown model: {kind}')
    numeric = Pipeline([('fill', SimpleImputer(strategy='median', keep_empty_features=True)),
                        ('scale', StandardScaler())])
    transform = ColumnTransformer([
        ('numeric', numeric, NUMERIC),
        ('categorical', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
    ])
    classifier = (LogisticRegression(max_iter=1500, random_state=42)
                  if kind == 'logistic' else RandomForestClassifier(
                      n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=42))
    return Pipeline([('features', transform), ('classifier', classifier)])


class CancellationModel:
    def __init__(self, pipeline, calibrator=None):
        self.pipeline = pipeline
        self.calibrator = calibrator

    def predict_batch(self, bookings):
        if len(bookings) == 0:
            return np.array([], dtype=float)
        classes = list(self.pipeline.classes_)
        if 1 not in classes:
            return np.zeros(len(bookings))
        classifier = self.pipeline.named_steps['classifier']
        kwargs = {'thread_count': 1} if isinstance(classifier, CatBoostClassifier) else {}
        probability = self.pipeline.predict_proba(features(bookings), **kwargs)[:, classes.index(1)]
        if self.calibrator is not None:
            probability = self.calibrator.predict_proba(self._logits(probability))[:, 1]
        return probability

    @staticmethod
    def _logits(probability):
        clipped = np.clip(probability, 1e-6, 1 - 1e-6)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    def fit_calibration(self, bookings):
        """Fit sigmoid scaling on a separate, later historical sample."""
        if self.calibrator is not None:
            raise ValueError('Calibration is already fitted')
        if bookings.cancelled.nunique() != 2:
            raise ValueError('Calibration requires both cancellation classes')
        raw = self.predict_batch(bookings)
        self.calibrator = LogisticRegression(C=1000, random_state=42).fit(
            self._logits(raw), bookings.cancelled)
        return self

    def predict_cancellation(self, booking):
        return float(self.predict_batch(pd.DataFrame([booking]))[0])

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'pipeline': self.pipeline, 'calibrator': self.calibrator}, path)

    @classmethod
    def load(cls, path):
        saved = joblib.load(path)
        return cls(saved['pipeline'], saved.get('calibrator')) if isinstance(saved, dict) else cls(saved)


def read_bookings(path='train.csv'):
    data = pd.read_csv(path, parse_dates=['arrival_date', 'booking_date'])
    data['checkout'] = data['arrival_date'] + pd.to_timedelta(data['nights'], unit='D')
    return data
