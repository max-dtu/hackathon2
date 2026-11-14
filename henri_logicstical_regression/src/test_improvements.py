import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import src.test_pipeline as fixtures
from src.demand import FutureDemand
from src.policy import BookingPolicy
from src.prediction import CancellationModel
from src.simulator import ReplayContext
from src.test_pipeline import booking
from src.train import cached_predictions


class DemandTests(unittest.TestCase):
    def history(self):
        b = booking(arrival="2016-06-01", rate=100)
        b["booking_date"] = pd.Timestamp("2016-05-30")
        return pd.DataFrame([b])

    def test_zero_demand_dates_and_sparse_fallback(self):
        forecast = FutureDemand.fit(self.history(), "2016-07-01")
        rates, exposure, fallback = forecast.curves[("Downtown Hotel", 6, 3)]
        self.assertEqual(exposure, 30)
        self.assertEqual(len(rates), 1)
        self.assertEqual(fallback, "month")
        self.assertEqual(
            forecast.curves[("Downtown Hotel", 7, 3)][2], "adjacent_months"
        )
        self.assertEqual(forecast.curves[("Downtown Hotel", 1, 3)][2], "hotel")
        self.assertEqual(
            forecast.reservation_price(
                "Downtown Hotel", "2017-06-01", "2017-05-15", 0.02
            ),
            100,
        )
        self.assertEqual(
            forecast.reservation_price("Downtown Hotel", "2017-06-01", "2017-05-15", 1),
            0,
        )

    def test_per_night_lead_time(self):
        rows = self.history()
        rows["nights"] = 3
        rows["checkout"] = pd.Timestamp("2016-06-04")
        forecast = FutureDemand.fit(rows, "2016-07-01")
        self.assertEqual(len(forecast.curves[("Downtown Hotel", 6, 0)][0]), 1)
        self.assertEqual(len(forecast.curves[("Downtown Hotel", 6, 1)][0]), 3)

    def test_future_outcomes_do_not_change_forecast(self):
        history = self.history()
        future = booking(2, arrival="2016-08-01", rate=9999)
        a = FutureDemand.fit(history, "2016-07-01")
        b = FutureDemand.fit(pd.concat([history, pd.DataFrame([future])]), "2016-07-01")
        self.assertEqual(a.curves.keys(), b.curves.keys())
        for key in a.curves:
            np.testing.assert_array_equal(a.curves[key][0], b.curves[key][0])
            self.assertEqual(a.curves[key][1:], b.curves[key][1:])

    def test_arrival_day_and_unknown_hotel_have_no_price_protection(self):
        forecast = FutureDemand.fit(self.history(), "2016-07-01")
        self.assertEqual(
            forecast.reservation_price("Downtown Hotel", "2016-06-01", "2016-06-01", 0),
            0,
        )
        self.assertEqual(
            forecast.reservation_price("Unknown", "2016-06-01", "2016-05-01", 0), 0
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demand.joblib"
            forecast.save(path)
            self.assertEqual(FutureDemand.load(path).cutoff, "2016-07-01")


class ValuePolicyTests(unittest.TestCase):
    def policy(self, **kwargs):
        capacity = {
            "Downtown Hotel": pd.Series(1, index=pd.date_range("2016-07-01", periods=3))
        }
        return BookingPolicy(pd.DataFrame(), capacity, lambda b: 0.5, **kwargs)

    def demand(self, rates=(150, 150)):
        return FutureDemand(
            {
                ("Downtown Hotel", 7, band): (np.array(rates), 1, "month")
                for band in range(9)
            },
            "2016-07-01",
        )

    def test_protects_scarce_room_for_higher_rate(self):
        policy = self.policy(protection=1, demand=self.demand())
        self.assertFalse(policy.decide(booking(rate=100)))
        later = booking(rate=200)
        later["booking_date"] = pd.Timestamp("2016-06-30")
        self.assertTrue(policy.decide(later))
        self.assertEqual(policy.mean["Downtown Hotel"][pd.Timestamp("2016-07-01")], 0.5)

    def test_quiet_night_and_zero_protection_accept_cheap_booking(self):
        self.assertTrue(
            self.policy(protection=1, demand=self.demand(rates=())).decide(
                booking(rate=20)
            )
        )
        self.assertTrue(self.policy().decide(booking(rate=20)))

    def test_releases_price_protection_near_arrival(self):
        b = booking(rate=20)
        b["booking_date"] = b["arrival_date"]
        self.assertTrue(self.policy(protection=1, demand=self.demand()).decide(b))

    def test_compares_whole_stay_against_all_nightly_prices(self):
        # Two scarce nights require a total of 300.
        self.assertFalse(
            self.policy(protection=1, demand=self.demand()).decide(
                booking(nights=2, rate=149)
            )
        )
        self.assertTrue(
            self.policy(protection=1, demand=self.demand()).decide(
                booking(nights=2, rate=150)
            )
        )

    def test_fractional_allowance_and_hotel_override(self):
        p = self.policy(allowance_fraction=0.5)
        self.assertTrue(all(p.decide(booking(i)) for i in range(3)))
        self.assertFalse(p.decide(booking(4)))
        p = self.policy(
            protection=0,
            demand=self.demand(),
            hotel_settings={"Downtown Hotel": {"protection": 1}},
        )
        self.assertFalse(p.decide(booking(rate=20)))

    def test_missing_forecast_fails_explicitly(self):
        with self.assertRaises(ValueError):
            self.policy(protection=1)


class EvaluationTests(unittest.TestCase):
    def test_controlled_scenario_excludes_inherited_bookings(self):
        old = booking(1)
        new = booking(2)
        new["booking_date"] = pd.Timestamp("2016-07-01")
        context = ReplayContext(
            pd.DataFrame([old, new]), "2016-07-01", "2016-07-03", "controlled"
        )
        result = context.run({2: 0.0})
        self.assertEqual(result["commitments"], 0)
        self.assertEqual(result["initial_in_house"]["Downtown Hotel"], 0)
        self.assertEqual(result["net_revenue"], 100)
        self.assertEqual(
            sum(v["net_revenue"] for v in result["by_hotel"].values()), 100
        )

    def test_policy_never_gets_outcomes(self):
        data = pd.DataFrame([booking()])
        context = ReplayContext(data, "2016-06-01", "2016-07-03")
        for _, row in context.records:
            self.assertNotIn("cancelled", row)
            self.assertNotIn("checkout", row)

    def test_cache_hit_and_invalidation(self):
        fixtures.ModelTests.setUpClass()
        model = fixtures.ModelTests.model
        rows = fixtures.ModelTests.rows
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source, artifact, cache = (
                directory / "data.csv",
                directory / "model.joblib",
                directory / "scores.npz",
            )
            source.write_text("first")
            model.save(artifact)
            _, hit = cached_predictions(rows, source, artifact, cache)
            self.assertFalse(hit)
            with patch.object(
                CancellationModel,
                "load",
                side_effect=AssertionError("cache should avoid inference"),
            ):
                _, hit = cached_predictions(rows, source, artifact, cache)
                self.assertTrue(hit)
            source.write_text("changed")
            _, hit = cached_predictions(rows, source, artifact, cache)
            self.assertFalse(hit)
            with artifact.open("ab") as stream:
                stream.write(b"new model fingerprint")
            _, hit = cached_predictions(rows, source, artifact, cache)
            self.assertFalse(hit)


if __name__ == "__main__":
    unittest.main()
