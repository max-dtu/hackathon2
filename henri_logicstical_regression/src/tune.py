"""Tune booking-policy settings on replays shaped like the official evaluation.

python -m src.tune            # search and report; writes artifacts/tuning.json
python -m src.tune --apply    # also update artifacts/policy.json and submission.zip

The official run replays a whole year: bookings confirmed earlier are inherited and
every later request is decided, including long-lead ones. Each period here does the
same on history. It starts at a different date and runs to the end of the data, with
the cancellation model and demand forecast fitted only on stays completed before
that start. Settings are compared on several periods so one season cannot decide.

The search covers the overbooking allowance, price protection, the uncertainty
margin and the walk-cost rule, and lets each hotel keep its own best settings when
that is clearly better than shared ones.
"""

import argparse
import itertools
import json
import os
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from src.demand import FutureDemand
from src.prediction import read_bookings
from src.simulator import ROOMS, ReplayContext
from src.train import SCHEMA_VERSION, digest, fit_model, package, training_signature

STARTS = ("2016-01-01", "2016-04-01", "2016-07-01")
ALLOWANCES = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
PROTECTIONS = (1.0, 1.25, 1.5, 2.0)
# (margin, walk_cost): the margin only applies to the occupancy cap, which the
# walk-cost rule replaces.
RULES = ((0.0, 0.0), (0.5, 0.0), (0.0, 1.0))
KEYS = ("allowance_fraction", "protection", "margin", "walk_cost")
# A hotel keeps its own settings only if they beat shared ones by this much.
HOTEL_GAIN = 0.01


def grid():
    return [
        dict(zip(KEYS, (fraction, protection, margin, walk)))
        for fraction, protection, (margin, walk) in itertools.product(
            ALLOWANCES, PROTECTIONS, RULES
        )
    ]


def hotel_nets(context, probabilities, demand, configs):
    rows = []
    for config in configs:
        result = context.run(probabilities, demand=demand, **config)
        rows.append({h: t["net_revenue"] for h, t in result["by_hotel"].items()})
    return rows


def prepare(data, model, start):
    start = pd.Timestamp(start)
    scores = fit_model(data, model, start).predict_batch(data)
    probabilities = dict(zip(data.booking_id, scores))
    end = data.arrival_date.max() + pd.Timedelta(days=1)
    return ReplayContext(data, start, end), probabilities, FutureDemand.fit(data, start)


def search(data, model, configs, jobs):
    """Return {period: [{hotel: net revenue} for each config]}."""
    table = {}
    for start in STARTS:
        context, probabilities, demand = prepare(data, model, start)
        chunks = [configs[i::jobs] for i in range(jobs)]
        parts = Parallel(n_jobs=jobs)(
            delayed(hotel_nets)(context, probabilities, demand, chunk)
            for chunk in chunks if chunk
        )
        rows = [None] * len(configs)
        for offset, part in enumerate(parts):
            rows[offset::jobs] = part
        table[start] = rows
        print(f"Replayed {len(configs)} settings from {start}: "
              f"{len(context.commitments)} inherited, {len(context.requests)} requests",
              flush=True)
    return table


def relative(table, hotels):
    """Average over periods of net revenue relative to that period's best setting."""
    count = len(next(iter(table.values())))
    scores = [0.0] * count
    for rows in table.values():
        nets = [sum(row[h] for h in hotels) for row in rows]
        best = max(nets)
        for i, net in enumerate(nets):
            scores[i] += net / best / len(table) if best > 0 else 0.0
    return scores


def choose(table, configs):
    hotels = list(ROOMS)
    shared_scores = relative(table, hotels)
    shared = max(range(len(configs)), key=shared_scores.__getitem__)
    hotel_settings = {}
    for hotel in hotels:
        scores = relative(table, [hotel])
        own = max(range(len(configs)), key=scores.__getitem__)
        if scores[own] - scores[shared] > HOTEL_GAIN:
            hotel_settings[hotel] = configs[own]
    return shared, shared_scores, hotel_settings


def policy_from(model, shared, hotel_settings):
    policy = dict(version=SCHEMA_VERSION, model=model, mode="expected", allowance=0, **shared)
    if hotel_settings:
        policy["hotel_settings"] = hotel_settings
    return policy


def net_for(rows, index, hotel_indexes):
    return sum(rows[hotel_indexes.get(h, index)][h] for h in ROOMS)


def describe(config):
    rule = "walk-cost rule" if config["walk_cost"] else f"margin {config['margin']:g}"
    return (f"allowance {config['allowance_fraction']:.0%}, "
            f"price protection {config['protection']:g}, {rule}")


def validate_artifacts(data_path, output, model):
    """Do not mix newly fitted tuning models with stale packaged artifacts."""
    signature = training_signature(data_path)
    development = json.loads((output / 'development.json').read_text())
    final_path = output / 'final_manifest.json'
    if not final_path.exists():
        raise ValueError('Missing final provenance. Run main.py before tuning.')
    final = json.loads(final_path.read_text())
    if (development['signature'] != signature
            or final['training_signature'] != signature
            or final['model'] != model
            or final['model_fingerprint'] != digest((output / 'model.joblib').read_bytes())
            or final['demand_fingerprint'] != digest((output / 'demand.joblib').read_bytes())):
        raise ValueError('Stale training artifacts. Run main.py before tuning.')
    return signature


def run(args):
    output = Path(args.output)
    current = json.loads((output / "policy.json").read_text())
    model = current["model"]
    signature = validate_artifacts(args.data, output, model)
    data = read_bookings(args.data)
    configs = grid()
    before = {key: current.get(key, 0.0) for key in KEYS}
    if before not in configs:
        configs.append(before)
    table = search(data, model, configs, args.jobs)
    shared, scores, hotel_settings = choose(table, configs)
    hotel_indexes = {h: configs.index(c) for h, c in hotel_settings.items()}
    start_index = configs.index(before)

    print("\nBest shared settings (score = average share of each period's best):")
    for i in sorted(range(len(configs)), key=scores.__getitem__, reverse=True)[:8]:
        nets = "  ".join(f"{sum(table[s][i].values()):>11,.0f}" for s in STARTS)
        print(f"  {scores[i]:.4f}  {nets}  {describe(configs[i])}")
    print(f"\nCurrent policy.json: {describe(before)}")
    print(f"Selected shared:     {describe(configs[shared])}")
    for hotel, config in hotel_settings.items():
        print(f"Selected for {hotel}: {describe(config)}")
    if not hotel_settings:
        print("Hotel-specific settings were not clearly better than shared ones.")
    print("\nNet revenue by period (current -> selected):")
    report = {}
    for start in STARTS:
        old = sum(table[start][start_index].values())
        new = net_for(table[start], shared, hotel_indexes)
        report[start] = dict(current=old, selected=new)
        print(f"  from {start}: {old:,.0f} -> {new:,.0f} ({new / old - 1:+.1%})")

    policy = policy_from(model, configs[shared], hotel_settings)
    (output / "tuning.json").write_text(json.dumps(dict(
        training_signature=signature, model=model,
        diagnostic_note='Overlapping, inspected historical periods; not an unseen test.',
        periods=list(STARTS), current=before, selected=policy, net_revenue=report,
        settings=[dict(config, score=scores[i],
                       net_revenue={s: table[s][i] for s in STARTS})
                  for i, config in enumerate(configs)],
    ), indent=2) + "\n")
    print(f"\nSaved every result to {output / 'tuning.json'}")
    if args.apply:
        if validate_artifacts(args.data, output, model) != signature:
            raise ValueError('Training inputs changed during tuning. Run tuning again.')
        (output / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
        package(output, output / "model.joblib", policy, "submission.zip")
        print(f"Updated {output / 'policy.json'} and {output / 'submission.zip'}")
    else:
        print("Run again with --apply to update policy.json and submission.zip.")


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="train.csv")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--apply", action="store_true",
                        help="Write the selected settings to policy.json and submission.zip")
    run(parser.parse_args())


if __name__ == "__main__":
    cli()
