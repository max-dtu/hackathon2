import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.demand import FutureDemand
from src.prediction import CancellationModel, features, make_model
from src.test_pipeline import booking
from src.train import calibration_partitions, package, settings


class RetainedModelTests(unittest.TestCase):
    def test_calibration_probabilities_survive_reload(self):
        rows = pd.DataFrame([booking(i, cancelled=i % 2) for i in range(120)])
        rows['days_ahead'] = np.arange(len(rows))
        base, later = rows.iloc[:60], rows.iloc[60:]
        model = CancellationModel(make_model('logistic').fit(features(base), base.cancelled))
        model.fit_calibration(later)
        batch = model.predict_batch(later)
        self.assertTrue(np.isfinite(batch).all())
        self.assertTrue(((batch >= 0) & (batch <= 1)).all())
        self.assertAlmostEqual(batch[0], model.predict_cancellation(later.iloc[0].to_dict()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'calibrated.joblib'
            model.save(path)
            restored = CancellationModel.load(path)
            self.assertIsNotNone(restored.calibrator)
            np.testing.assert_allclose(batch, restored.predict_batch(later))
        with self.assertRaises(ValueError):
            model.fit_calibration(later)

    def test_calibration_uses_only_later_matured_requests(self):
        rows = pd.DataFrame([
            dict(booking(1), booking_date=pd.Timestamp('2016-01-01'), checkout=pd.Timestamp('2016-03-20')),
            dict(booking(2), booking_date=pd.Timestamp('2016-04-02'), checkout=pd.Timestamp('2016-05-20')),
            dict(booking(3), booking_date=pd.Timestamp('2016-03-20'), checkout=pd.Timestamp('2016-05-20')),
            dict(booking(4), booking_date=pd.Timestamp('2016-06-20'), checkout=pd.Timestamp('2016-07-01')),
        ])
        base, later = calibration_partitions(rows, '2016-07-01')
        self.assertEqual(list(base.booking_id), [1])
        self.assertEqual(list(later.booking_id), [2])
        self.assertTrue(set(base.booking_id).isdisjoint(later.booking_id))

    def test_forest_probabilities_and_standalone_bundle(self):
        rows = pd.DataFrame([booking(i, cancelled=i % 2) for i in range(60)])
        rows['days_ahead'] = np.arange(len(rows))
        model = CancellationModel(make_model('forest').fit(features(rows), rows.cancelled))
        probe = booking()
        probe.update(children=None, country='unseen', days_ahead=10)
        probability = model.predict_cancellation(probe)
        self.assertTrue(np.isfinite(probability) and 0 <= probability <= 1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = output / 'forest.joblib'
            model.save(path)
            self.assertAlmostEqual(probability, CancellationModel.load(path).predict_cancellation(probe))
            FutureDemand({}, '2017-01-01').save(output / 'demand.joblib')
            config = settings('forest')
            config.update(protection=1.0, allowance_fraction=0.1)
            package(output, path, config, 'candidate.zip')
            extracted = output / 'isolated'
            with zipfile.ZipFile(output / 'candidate.zip') as bundle:
                self.assertIsNone(bundle.testzip())
                bundle.extractall(extracted)
            command = '''
from pathlib import Path
import pandas as pd
from solution import Solution
import src.solution
assert Path(src.solution.__file__).parent.parent == Path.cwd()
dates = pd.date_range('2017-01-01', '2017-12-31')
capacity = {'Downtown Hotel': pd.Series(124, index=dates), 'Seaside Hotel': pd.Series(103, index=dates)}
robot = Solution(pd.DataFrame(), capacity)
booking = dict(hotel='Downtown Hotel', booking_date=pd.Timestamp('2017-01-01'), arrival_date=pd.Timestamp('2017-01-05'), nights=2, nightly_rate=100)
assert type(robot.decide(booking)) is bool
assert robot.decide(dict(booking, arrival_date=pd.Timestamp('2018-01-01'))) is False
'''
            result = subprocess.run([sys.executable, '-c', command], cwd=extracted,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
