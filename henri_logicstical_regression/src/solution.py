"""Your team's solution. At the end of the hackathon you pass this class to evaluate(Solution)
from evaluate.py, which creates one Solution and replays every 2017 booking request through it.

RULES -- Downtown Hotel has 124 rooms, Seaside Hotel 103. A guest who shows up earns
nightly_rate x nights. When a hotel is full at check-in, the cheapest arriving guests are
walked: they earn nothing and cost 2.5 x their stay value. A seated guest keeps the room.
Your result is the net revenue: revenue of seated guests - walk penalties.

Reading any 2017 data yourself is cheating.
"""

import json
from pathlib import Path

import pandas as pd

from src.demand import FutureDemand
from src.policy import BookingPolicy
from src.prediction import CancellationModel


class Solution:
    def __init__(
        self, commitments: pd.DataFrame, capacity: dict[str, pd.Series]
    ) -> None:
        """Called once, before 2017 starts.

        commitments  2017 arrivals booked before 2017-01-01. The hotel already confirmed them, so
                     decide() is never called for them. Same columns as train.csv, except:
                     - no cancelled column;
                     - arrival_date and booking_date are already pandas Timestamps, not strings.

        capacity     hotel -> rooms free on each night of 2017 (indexed by date): the rooms minus
                     guests who arrived in 2016 and are still in house.
        """
        artifacts = Path(__file__).resolve().parent.parent / "artifacts"
        self.model = CancellationModel.load(artifacts / "model.joblib")
        # Per-request inference avoids thread-pool setup for each small prediction.
        classifier = self.model.pipeline.named_steps["classifier"]
        if hasattr(classifier, "n_jobs"):
            classifier.n_jobs = 1
        settings = json.loads((artifacts / "policy.json").read_text())
        demand_path = artifacts / "demand.joblib"
        demand = FutureDemand.load(demand_path) if demand_path.exists() else None
        # Batch inherited predictions once, especially important for random forest.
        inherited_scores = (dict(zip(commitments['booking_id'], self.model.predict_batch(commitments)))
                            if len(commitments) else {})

        def predict(booking):
            cached = inherited_scores.get(booking.get('booking_id'))
            return cached if cached is not None else self.model.predict_cancellation(booking)

        self.policy = BookingPolicy(
            commitments,
            capacity,
            predict,
            mode=settings["mode"],
            margin=settings["margin"],
            allowance=settings["allowance"],
            allowance_fraction=settings.get("allowance_fraction", 0.0),
            protection=settings.get("protection", 0.0),
            demand=demand,
            hotel_settings=settings.get("hotel_settings"),
            walk_cost=settings.get("walk_cost", 0.0),
        )

    def decide(self, booking: dict) -> bool:
        """Called once per 2017 booking request, in booking-date order. `booking` is one row with
        the same columns as commitments, as {column: value}. Return True to confirm the booking,
        False to reject it."""
        return bool(self.policy.decide(booking))
