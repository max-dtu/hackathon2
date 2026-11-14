import json
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
from src.simulator import ReplayContext
from src.test_pipeline import booking
from src.train import digest, fit_model, package, training_signature
from src.tune import policy_from, validate_artifacts


class CatBoostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = pd.DataFrame([booking(i, cancelled=i % 2) for i in range(240)])
        cls.rows['days_ahead'] = np.arange(len(cls.rows))
        cls.rows['agent'] = 304.0
        cls.rows['booking_date'] = pd.Timestamp('2016-04-01')
        cls.rows['checkout'] = pd.Timestamp('2016-05-01')
        cls.rows.loc[:119, 'booking_date'] = pd.Timestamp('2016-01-01')
        cls.rows.loc[:119, 'checkout'] = pd.Timestamp('2016-03-01')
        cls.models = {name: fit_model(cls.rows, name, '2016-07-01')
                      for name in ('catboost', 'catboost_calibrated')}

    def test_deposit_is_ignored_by_all_models(self):
        absent = self.rows.drop(columns=['deposit'], errors='ignore')
        changed = absent.assign(deposit='Non Refund', deposit_type='Refundable')
        pd.testing.assert_frame_equal(features(absent), features(changed))
        for kind in ('logistic', 'forest', 'catboost'):
            model = CancellationModel(make_model(kind).fit(features(absent), absent.cancelled))
            np.testing.assert_allclose(model.predict_batch(absent), model.predict_batch(changed),
                                       rtol=0, atol=1e-14)

    def test_missing_unseen_and_normalized_ids(self):
        probe = self.rows.iloc[:2].copy()
        probe['children'] = np.nan
        probe['country'] = 'Unseen'
        probe['company'] = None
        for model in self.models.values():
            scores = model.predict_batch(probe)
            self.assertTrue(np.isfinite(scores).all())
            self.assertTrue(((scores >= 0) & (scores <= 1)).all())
            other = probe.assign(agent='304', company=np.nan)
            np.testing.assert_allclose(scores, model.predict_batch(other))
            self.assertAlmostEqual(scores[0], model.predict_cancellation(probe.iloc[0].to_dict()))

    def test_calibration_and_roundtrip(self):
        self.assertIsNone(self.models['catboost'].calibrator)
        self.assertIsNotNone(self.models['catboost_calibrated'].calibrator)
        # Changing outcomes in the later calibration slice cannot change the base classifier.
        changed = self.rows.copy()
        changed.loc[120:, 'cancelled'] = 1 - changed.loc[120:, 'cancelled']
        alternative = fit_model(changed, 'catboost_calibrated', '2016-07-01')
        original = self.models['catboost_calibrated']
        np.testing.assert_allclose(original.pipeline.predict_proba(features(self.rows)),
                                   alternative.pipeline.predict_proba(features(self.rows)))
        with tempfile.TemporaryDirectory() as directory:
            for name, model in self.models.items():
                path = Path(directory) / f'{name}.joblib'
                model.save(path)
                np.testing.assert_allclose(model.predict_batch(self.rows),
                                           CancellationModel.load(path).predict_batch(self.rows))

    def test_bundles_preserve_tuned_policy_and_load_in_fresh_process(self):
        for name, model in self.models.items():
            with self.subTest(model=name), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                model.save(output / 'model.joblib')
                FutureDemand({}, '2017-01-01').save(output / 'demand.joblib')
                shared = dict(allowance_fraction=0.2, protection=1.25, margin=0.0, walk_cost=1.0)
                config = policy_from(name, shared, {'Seaside Hotel': dict(shared, margin=0.5)})
                package(output, output / 'model.joblib', config, 'candidate.zip')
                extracted = output / 'isolated'
                with zipfile.ZipFile(output / 'candidate.zip') as bundle:
                    self.assertIsNone(bundle.testzip())
                    self.assertEqual(json.loads(bundle.read('artifacts/policy.json')), config)
                    bundle.extractall(extracted)
                command = '''
import pandas as pd
from solution import Solution
capacity = {h: pd.Series(n, index=pd.date_range('2017-01-01', periods=30))
            for h, n in [('Downtown Hotel', 124), ('Seaside Hotel', 103)]}
robot = Solution(pd.DataFrame(), capacity)
assert robot.policy.walk_cost == 1.0
assert robot.policy.hotel_settings['Seaside Hotel']['margin'] == 0.5
b = dict(hotel='Downtown Hotel', booking_date=pd.Timestamp('2017-01-01'),
         arrival_date=pd.Timestamp('2017-01-05'), nights=2, nightly_rate=100)
assert type(robot.decide(b)) is bool
assert robot.decide(dict(b, arrival_date=pd.Timestamp('2018-01-01'))) is False
'''
                result = subprocess.run([sys.executable, '-c', command], cwd=extracted,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)


class ProvenanceTests(unittest.TestCase):
    def test_tuner_rejects_stale_or_replaced_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            data = output / 'train.csv'
            data.write_text('historical data')
            signature = training_signature(data)
            (output / 'development.json').write_text(json.dumps(dict(signature=signature)))
            (output / 'model.joblib').write_bytes(b'model')
            (output / 'demand.joblib').write_bytes(b'demand')
            final = dict(training_signature=signature, model='catboost',
                         model_fingerprint=digest(b'model'), demand_fingerprint=digest(b'demand'))
            (output / 'final_manifest.json').write_text(json.dumps(final))
            self.assertEqual(validate_artifacts(data, output, 'catboost'), signature)
            with self.assertRaises(ValueError):
                validate_artifacts(data, output, 'logistic')
            (output / 'model.joblib').write_bytes(b'replaced')
            with self.assertRaises(ValueError):
                validate_artifacts(data, output, 'catboost')
            (output / 'model.joblib').write_bytes(b'model')
            data.write_text('different historical data')
            with self.assertRaises(ValueError):
                validate_artifacts(data, output, 'catboost')

    def test_validation_cohort_includes_crossing_stays(self):
        rows = pd.DataFrame([booking(1, arrival='2016-09-30', nights=5),
                             booking(2, arrival='2016-10-01', nights=1)])
        rows['booking_date'] = pd.Timestamp('2016-07-01')
        context = ReplayContext(rows, '2016-07-01', '2016-10-01')
        self.assertEqual(list(context.requests.booking_id), [1])
        self.assertEqual(context.requests.iloc[0].checkout, pd.Timestamp('2016-10-05'))


if __name__ == '__main__':
    unittest.main()
