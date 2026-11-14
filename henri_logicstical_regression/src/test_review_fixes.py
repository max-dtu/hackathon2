import importlib.util
import json
import unittest
from pathlib import Path

import pandas as pd

from src.demand import FutureDemand
from src.policy import BookingPolicy
from src.simulator import ROOMS, ReplayContext, replay
from src.test_pipeline import booking
from src.tune import choose, grid, policy_from


def request(identifier, arrival, nights=1, rate=100, booked="2016-07-01"):
    row = booking(identifier, arrival=arrival, nights=nights, rate=rate)
    row["booking_date"] = pd.Timestamp(booked)
    return row


class BoundaryTests(unittest.TestCase):
    def test_stay_crossing_period_end_is_replayed(self):
        data = pd.DataFrame([request(1, "2016-07-02", nights=5)])
        result = replay(data, "2016-07-01", "2016-07-04", {1: 0.0}, mode="capacity")
        self.assertEqual(result["requests"], 1)
        self.assertEqual(result["accepted_requests"], 1)
        self.assertEqual(result["net_revenue"], 500)

    def test_crossing_stays_compete_for_rooms(self):
        # 124 long stays arrive on the last day and fill the hotel; one more is walked.
        rows = [request(i, "2016-07-03", nights=4, rate=200) for i in range(124)]
        rows.append(request(999, "2016-07-03", nights=1, rate=50))
        scores = {row["booking_id"]: 0.0 for row in rows}
        result = replay(pd.DataFrame(rows), "2016-07-01", "2016-07-04", scores, mode="all")
        self.assertEqual(result["walked"], 1)
        self.assertEqual(result["penalties"], 125)

    def test_arrivals_after_period_are_excluded(self):
        data = pd.DataFrame([request(1, "2016-07-04")])
        context = ReplayContext(data, "2016-07-01", "2016-07-04")
        self.assertEqual(len(context.requests), 0)

    def test_occupancy_uses_period_nights(self):
        data = pd.DataFrame([request(1, "2016-07-01", nights=10)])
        context = ReplayContext(data, "2016-07-01", "2016-07-04")
        self.assertEqual(context.period_nights, 3)
        self.assertEqual(len(context.capacity["Downtown Hotel"]), 10)


class WalkCostTests(unittest.TestCase):
    def setUp(self):
        history = []
        for i, rate in enumerate([40, 60, 80, 100, 120, 140, 160, 180]):
            row = booking(i, arrival="2016-06-10", rate=rate)
            row["booking_date"] = pd.Timestamp("2016-06-01")
            history.append(row)
        self.demand = FutureDemand.fit(pd.DataFrame(history), "2016-07-01")

    def policy(self, commitments=(), rooms=1, **kwargs):
        capacity = {
            "Downtown Hotel": pd.Series(rooms, index=pd.date_range("2016-06-10", periods=5))
        }
        return BookingPolicy(pd.DataFrame(list(commitments)), capacity, lambda b: 0.0,
                             demand=self.demand, **kwargs)

    def test_low_rate_is_a_cheap_historical_rate(self):
        self.assertEqual(self.demand.low_rate("Downtown Hotel", "2017-06-10"), 60)

    def test_requires_demand(self):
        with self.assertRaises(ValueError):
            BookingPolicy(pd.DataFrame([]), {"Downtown Hotel": pd.Series(
                1, index=pd.date_range("2016-06-10", periods=1))}, lambda b: 0.0, walk_cost=1.0)

    def test_empty_hotel_accepts(self):
        p = self.policy(rooms=10, walk_cost=1.0)
        self.assertTrue(p.decide(booking(1, arrival="2016-06-10", rate=30)))

    def test_full_hotel_rejects_cheap_but_accepts_valuable_stay(self):
        # One room, already taken by a certain arrival: a walk is almost sure.
        held = [booking(1, arrival="2016-06-10", rate=100)]
        cheap = self.policy(held, walk_cost=1.0)
        self.assertFalse(cheap.decide(booking(2, arrival="2016-06-10", rate=100)))
        # 3.5 x the cheap rate of 60 is 210: a stay worth more is still accepted,
        # which the plain occupancy cap can never do.
        valuable = self.policy(held, walk_cost=1.0)
        self.assertTrue(valuable.decide(booking(3, arrival="2016-06-10", rate=400)))
        capped = self.policy(held)
        self.assertFalse(capped.decide(booking(3, arrival="2016-06-10", rate=400)))

    def test_default_behaviour_is_unchanged(self):
        a, b = self.policy(rooms=2), self.policy(rooms=2, walk_cost=0.0)
        for i in range(4):
            row = booking(i, arrival="2016-06-10")
            self.assertEqual(a.decide(row), b.decide(row))

    def test_hotel_settings_override_margin_and_walk_cost(self):
        held = [booking(1, arrival="2016-06-10", rate=100)]
        p = self.policy(held, hotel_settings={"Downtown Hotel": {"walk_cost": 1.0}})
        self.assertTrue(p.decide(booking(3, arrival="2016-06-10", rate=400)))
        capacity = {"Downtown Hotel": pd.Series(2, index=pd.date_range("2016-06-10", periods=1))}
        strict = BookingPolicy(pd.DataFrame([]), capacity, lambda b: 0.5,
                               hotel_settings={"Downtown Hotel": {"margin": 3.0}})
        loose = BookingPolicy(pd.DataFrame([]), capacity, lambda b: 0.5)
        rows = [booking(i, arrival="2016-06-10") for i in range(4)]
        self.assertLess(sum(strict.decide(r) for r in rows), sum(loose.decide(r) for r in rows))


class TuningTests(unittest.TestCase):
    def test_grid_covers_margin_and_walk_cost(self):
        configs = grid()
        self.assertTrue(any(c["margin"] > 0 for c in configs))
        self.assertTrue(any(c["walk_cost"] > 0 for c in configs))
        self.assertFalse(any(c["margin"] and c["walk_cost"] for c in configs))

    def test_hotel_settings_need_a_clear_gain(self):
        configs = [dict(allowance_fraction=a, protection=1.0, margin=0.0, walk_cost=0.0)
                   for a in (0.1, 0.2)]
        down, sea = list(ROOMS)
        close = {"p": [{down: 100.0, sea: 100.0}, {down: 100.5, sea: 99.9}]}
        shared, _, hotels = choose(close, configs)
        self.assertEqual((shared, hotels), (1, {}))
        split = {"p": [{down: 100.0, sea: 100.0}, {down: 150.0, sea: 80.0}]}
        shared, _, hotels = choose(split, configs)
        self.assertEqual(shared, 1)
        self.assertEqual(hotels, {sea: configs[0]})
        policy = policy_from("logistic", configs[shared], hotels)
        self.assertEqual(policy["hotel_settings"][sea]["allowance_fraction"], 0.1)
        json.dumps(policy)


class RootEntryPointTests(unittest.TestCase):
    def test_root_solution_exposes_the_working_class(self):
        root = Path(__file__).resolve().parents[2] / "solution.py"
        if not root.exists():
            self.skipTest("Repository-root solution.py is not present")
        spec = importlib.util.spec_from_file_location("root_solution", root)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from src.solution import Solution
        self.assertIs(module.Solution, Solution)


if __name__ == "__main__":
    unittest.main()
