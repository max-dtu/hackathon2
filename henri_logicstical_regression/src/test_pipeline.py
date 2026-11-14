import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.policy import BookingPolicy
from src.prediction import (
    CancellationModel,
    features,
    make_model,
)
from src.simulator import (
    allocate,
    reconstruct_in_house,
    replay,
)


def booking(identifier=1, arrival="2016-07-01", nights=1, rate=100, cancelled=0):
    return dict(
        booking_id=identifier,
        hotel="Downtown Hotel",
        arrival_date=pd.Timestamp(arrival),
        booking_date=pd.Timestamp("2016-06-01"),
        nights=nights,
        nightly_rate=rate,
        cancelled=cancelled,
        checkout=pd.Timestamp(arrival) + pd.Timedelta(days=nights),
    )


class PolicyTests(unittest.TestCase):
    def policy(self, commitments=None, **kwargs):
        capacity = {
            "Downtown Hotel": pd.Series(1, index=pd.date_range("2016-07-01", periods=3))
        }
        return BookingPolicy(
            pd.DataFrame([] if commitments is None else commitments),
            capacity,
            lambda b: 0.5,
            **kwargs,
        )

    def test_updates_immediately(self):
        p = self.policy()
        self.assertTrue(p.decide(booking(1)))
        self.assertTrue(p.decide(booking(2)))
        self.assertFalse(p.decide(booking(3)))

    def test_every_night_checked(self):
        p = self.policy(mode="capacity")
        self.assertTrue(p.decide(booking(arrival="2016-07-02")))
        self.assertFalse(p.decide(booking(nights=2)))

    def test_commitments_and_missing_capacity(self):
        p = self.policy([booking()], mode="capacity")
        self.assertFalse(p.decide(booking()))
        self.assertFalse(p.decide(booking(arrival="2016-07-04")))

    def test_zero_nights(self):
        self.assertFalse(self.policy().decide(booking(nights=0)))

    def test_margin(self):
        p = self.policy(margin=1)
        self.assertTrue(p.decide(booking()))
        self.assertFalse(p.decide(booking()))


class SimulatorTests(unittest.TestCase):
    def test_historical_state_cannot_exceed_capacity(self):
        rows = pd.DataFrame(
            [booking(i, arrival="2016-06-30", nights=3) for i in range(130)]
        )
        state = reconstruct_in_house(rows, pd.Timestamp("2016-07-01"))
        self.assertEqual(len(state["Downtown Hotel"]), 124)

    def test_penalty_is_separate_from_foregone_revenue(self):
        rows = pd.DataFrame(
            [
                dict(booking(), origin="inherited"),
                dict(booking(2, rate=200), origin="new"),
            ]
        )
        result = allocate(rows, rooms={"Downtown Hotel": 1}, audit=True)
        self.assertEqual(result["revenue"], 200)
        self.assertEqual(result["penalties"], 250)
        self.assertEqual(result["foregone_revenue"], 100)
        self.assertEqual(result["by_origin"]["inherited"]["penalties"], 250)
        self.assertEqual(result["walks"][0]["booking_id"], 1)

    def test_replay_hides_outcomes_and_preserves_request_order(self):
        rows = [booking(i, cancelled=0) for i in range(126)]
        for b in rows:
            b["booking_date"] = pd.Timestamp("2016-07-01")
        data = pd.DataFrame(rows)
        seen = []

        class Scores(dict):
            def __getitem__(self, key):
                seen.append(key)
                return super().__getitem__(key)

        scores = Scores({i: 0.0 for i in range(126)})
        result = replay(data, "2016-07-01", "2016-07-03", scores)
        self.assertEqual(seen, list(range(126)))
        self.assertEqual(result["accepted_requests"], 124)
        self.assertEqual(result["net_revenue"], 12400)
        self.assertEqual(result["walked"], 0)

    def test_replay_accounts_for_in_house_guests(self):
        old = [booking(i, arrival="2016-06-30", nights=3) for i in range(124)]
        new = booking(999)
        new["booking_date"] = pd.Timestamp("2016-07-01")
        result = replay(
            pd.DataFrame(old + [new]),
            "2016-07-01",
            "2016-07-04",
            {999: 0.0},
            mode="capacity",
        )
        self.assertEqual(result["accepted_requests"], 0)

    def test_existing_guest_keeps_room(self):
        result = allocate(
            pd.DataFrame(
                [
                    booking(1, nights=2, rate=50),
                    booking(2, arrival="2016-07-02", rate=500),
                ]
            ),
            rooms={"Downtown Hotel": 1},
        )
        self.assertEqual(result["net_revenue"], 100 - 1250)

    def test_cancellation_and_checkout(self):
        result = allocate(
            pd.DataFrame(
                [booking(1), booking(2, cancelled=1), booking(3, arrival="2016-07-02")]
            ),
            rooms={"Downtown Hotel": 1},
        )
        self.assertEqual(result["net_revenue"], 200)
        self.assertEqual(result["walked"], 0)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = pd.DataFrame([booking(i, cancelled=i % 2) for i in range(40)])
        cls.rows["days_ahead"] = np.arange(40)
        cls.pipeline = make_model("logistic").fit(
            features(cls.rows), cls.rows.cancelled
        )
        cls.model = CancellationModel(cls.pipeline)

    def test_missing_unseen_and_timestamp(self):
        b = booking()
        b.update(country="Unseen", children=None, agent=304)
        p = self.model.predict_cancellation(b)
        self.assertTrue(np.isfinite(p) and 0 <= p <= 1)

    def test_batch_and_roundtrip(self):
        batch = self.model.predict_batch(self.rows)
        self.assertAlmostEqual(
            batch[0], self.model.predict_cancellation(self.rows.iloc[0].to_dict())
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            self.model.save(path)
            np.testing.assert_allclose(
                batch, CancellationModel.load(path).predict_batch(self.rows)
            )

    def test_label_and_identifier_not_features(self):
        changed = self.rows.copy()
        changed["cancelled"] = 1 - changed.cancelled
        changed["booking_id"] = -1
        pd.testing.assert_frame_equal(features(self.rows), features(changed))


if __name__ == "__main__":
    unittest.main()
