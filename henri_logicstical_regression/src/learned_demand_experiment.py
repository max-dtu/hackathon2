"""Learned-demand comparison, shared by main.py and the focused CLI."""
import argparse
import json
from pathlib import Path
import time

import joblib

from src.demand import FutureDemand
from src.learned_demand import LearnedDemand
from src.prediction import read_bookings
from src.simulator import ReplayContext
from src.train import SCHEMA_VERSION, digest, fit_model, package, training_signature

SETTINGS = dict(version=SCHEMA_VERSION, model='logistic', mode='expected', allowance=0,
                allowance_fraction=.2, protection=1.25, margin=0., walk_cost=1.)
PERIODS = (('2016-07-01', '2016-10-01'), ('2016-10-01', '2017-01-01'))


def signature(data_path):
    source = Path(__file__).parent
    return digest(training_signature(data_path), SETTINGS, PERIODS,
                  *( (source / name).read_bytes() for name in
                     ('learned_demand.py', 'learned_demand_experiment.py',
                      'demand.py', 'policy.py', 'simulator.py', 'train.py')))


def run(data, data_path, output, stage='all'):
    """Honor train/evaluate/package stages without overwriting the main submission."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stamp = signature(data_path)
    manifest = output / 'development.json'
    if stage in ('all', 'train'):
        files = {}
        for cutoff, _ in PERIODS:
            started = time.monotonic()
            print(f'Learned demand: training {cutoff}...', flush=True)
            model = fit_model(data, 'logistic', cutoff)
            models = {'model': model, 'baseline': FutureDemand.fit(data, cutoff),
                      'learned': LearnedDemand.fit(data, cutoff)}
            for name, fitted in models.items():
                path = output / f'{cutoff}_{name}.joblib'
                if name == 'model': fitted.save(path)
                else: joblib.dump(fitted, path)
                files[path.name] = digest(path.read_bytes())
            print(f'  {models["learned"].training_rows} examples; {time.monotonic()-started:.1f}s', flush=True)
        manifest.write_text(json.dumps(dict(signature=stamp, files=files), indent=2)+'\n')
    if stage in ('all', 'evaluate', 'package'):
        if not manifest.exists():
            raise ValueError('Train learned-demand artifacts first with main.py --stage train')
        saved = json.loads(manifest.read_text())
        if saved['signature'] != stamp or any(
                digest((output / name).read_bytes()) != fingerprint
                for name, fingerprint in saved['files'].items()):
            raise ValueError('Stale learned-demand artifacts; run main.py --stage train')
    if stage in ('all', 'evaluate'):
        from src.prediction import CancellationModel
        report = dict(note='Previously inspected historical comparisons, not unseen tests.',
                      signature=stamp, policy=SETTINGS, periods={})
        options = {k: v for k, v in SETTINGS.items() if k not in ('version', 'model')}
        for cutoff, end in PERIODS:
            model = CancellationModel.load(output / f'{cutoff}_model.joblib')
            probabilities = dict(zip(data.booking_id, model.predict_batch(data)))
            context = ReplayContext(data, cutoff, end)
            results = {}
            for label, name in [('baseline', 'baseline'), ('learned_demand', 'learned')]:
                demand = joblib.load(output / f'{cutoff}_{name}.joblib')
                results[label] = context.run(probabilities, demand=demand, **options)
            a, b = (results[key]['net_revenue'] for key in ('baseline', 'learned_demand'))
            print(f'Learned-demand comparison {cutoff}: baseline={a:,.2f}; learned={b:,.2f}; delta={b-a:+,.2f}', flush=True)
            report['periods'][cutoff] = results
        (output / 'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    if stage in ('all', 'package'):
        print('Packaging baseline and learned-demand candidates...', flush=True)
        model = fit_model(data, 'logistic', '2017-01-01')
        model.save(output / 'model.joblib')
        for label, forecast in [('baseline', FutureDemand.fit(data, '2017-01-01')),
                                ('learned_demand', LearnedDemand.fit(data, '2017-01-01'))]:
            joblib.dump(forecast, output / 'demand.joblib')
            package(output, output / 'model.joblib', SETTINGS, label+'.zip')
        print(f'Learned-demand candidates: {output.resolve()}', flush=True)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='train.csv')
    parser.add_argument('--output', default='artifacts/learned_demand')
    parser.add_argument('--stage', choices=['all', 'train', 'evaluate', 'package'], default='all')
    args = parser.parse_args()
    run(read_bookings(args.data), args.data, args.output, args.stage)


if __name__ == '__main__':
    cli()
