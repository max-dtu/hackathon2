"""Train probabilities, evaluate policies, and package the selected submission.

python train.py --stage train      # probability models and historical demand only
python train.py --stage evaluate   # compare policies without retraining
python train.py --stage package    # train final selected model and package
python train.py                    # both stages
"""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import catboost
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.demand import FutureDemand
from src.prediction import (
    CancellationModel,
    features,
    make_model,
    read_bookings,
)
from src.simulator import ReplayContext

JULY = pd.Timestamp("2016-07-01")
OCTOBER = pd.Timestamp("2016-10-01")
ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 6
MODELS = ('logistic', 'logistic_calibrated', 'forest', 'forest_calibrated',
          'catboost', 'catboost_calibrated')
POLICIES = {
    'early': (0.10, 0.0),
    'balanced': (0.10, 1.0),
    'future_focused': (0.10, 1.5),
    'conservative': (0.05, 1.0),
    'aggressive': (0.15, 1.0),
}
ALLOWANCES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.0)


def digest(*values):
    h = hashlib.sha256()
    for value in values:
        h.update(value if isinstance(value, bytes) else str(value).encode())
        h.update(b"\0")
    return h.hexdigest()


def training_signature(data_path):
    return digest(
        Path(data_path).read_bytes(),
        (ROOT / "prediction.py").read_bytes(),
        sklearn.__version__,
        catboost.__version__,
        SCHEMA_VERSION,
        f"{MODELS};validation=2016-07-01;calibration=prior-3-months",
    )


def evaluation_signature(args, output):
    manifest = (output / "development.json").read_bytes()
    return digest(
        training_signature(args.data),
        manifest,
        *(
            (ROOT / name).read_bytes()
            for name in ("policy.py", "demand.py", "simulator.py", "train.py")
        ),
    )


def metrics(actual, probability):
    return dict(
        rows=len(actual),
        log_loss=float(log_loss(actual, probability, labels=[0, 1])),
        brier=float(brier_score_loss(actual, probability)),
        auc=float(roc_auc_score(actual, probability))
        if len(set(actual)) == 2
        else None,
    )


def calibration(actual, probability):
    groups = pd.DataFrame({"actual": np.asarray(actual), "predicted": probability})
    groups["band"] = pd.cut(
        groups.predicted, np.linspace(0, 1, 11), include_lowest=True
    )
    return [
        dict(
            band=str(band),
            rows=len(g),
            predicted=float(g.predicted.mean()),
            actual=float(g.actual.mean()),
        )
        for band, g in groups.groupby("band", observed=True)
    ]


def probability_report(data, probabilities):
    result = metrics(data.cancelled, probabilities)
    result["calibration"] = calibration(data.cancelled, probabilities)
    result["hotels"] = {}
    result["lead_bands"] = {}
    for column, groups in [
        ("hotels", data.hotel),
        (
            "lead_bands",
            pd.cut(
                data.days_ahead,
                [-1, 7, 30, 90, float("inf")],
                labels=["0-7", "8-30", "31-90", "91+"],
            ),
        ),
    ]:
        for group in groups.dropna().unique():
            mask = (groups == group).to_numpy()
            result[column][str(group)] = metrics(
                data.loc[mask, "cancelled"], probabilities[mask]
            )
            result[column][str(group)]["calibration"] = calibration(
                data.loc[mask, "cancelled"], probabilities[mask]
            )
    return result


def calibration_partitions(data, cutoff):
    cutoff = pd.Timestamp(cutoff)
    calibration_start = cutoff - pd.DateOffset(months=3)
    base = data[data.checkout < calibration_start]
    later = data[(data.booking_date >= calibration_start) & (data.checkout < cutoff)]
    return base, later


def fit_model(data, name, cutoff):
    if name not in MODELS:
        raise ValueError(f'Unknown model: {name}')
    if name.endswith('_calibrated'):
        history, calibrate = calibration_partitions(data, cutoff)
        if len(calibrate) < 100 or calibrate.cancelled.nunique() != 2:
            raise ValueError('Not enough chronological calibration examples')
        kind = name.removesuffix('_calibrated')
        model = CancellationModel(make_model(kind).fit(features(history), history.cancelled))
        return model.fit_calibration(calibrate)
    history = data[data.checkout < cutoff]
    return CancellationModel(make_model(name).fit(features(history), history.cancelled))


def train_development(data, args, output):
    names = list(MODELS)
    for name in names:
        print(f"Training {name}...", flush=True)
        fit_model(data, name, JULY).save(output / f"{name}_development.joblib")
    manifest = dict(
        signature=training_signature(args.data),
        models=names,
        cutoff=str(JULY.date()),
        schema_version=SCHEMA_VERSION,
        model_fingerprints={
            name: digest((output / f"{name}_development.joblib").read_bytes())
            for name in names
        },
    )
    (output / "development.json").write_text(json.dumps(manifest, indent=2) + "\n")


def cached_predictions(data, data_path, model_path, cache_path):
    fingerprint = digest(
        Path(data_path).read_bytes(),
        model_path.read_bytes(),
        (ROOT / "prediction.py").read_bytes(),
        sklearn.__version__,
        catboost.__version__,
    )
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as saved:
            if str(saved["fingerprint"]) == fingerprint and np.array_equal(
                saved["ids"], data.booking_id.to_numpy()
            ):
                return saved["probabilities"].copy(), True
    probabilities = CancellationModel.load(model_path).predict_batch(data)
    np.savez_compressed(
        cache_path,
        fingerprint=np.array(fingerprint),
        ids=data.booking_id.to_numpy(),
        probabilities=probabilities,
    )
    return probabilities, False


def forecast(data, cutoff, path):
    result = FutureDemand.fit(data, cutoff)
    result.save(path)
    return result


def options(settings, demand):
    return {
        **{
            key: value
            for key, value in settings.items()
            if key not in ("model", "version")
        },
        "demand": demand,
    }


def settings(model):
    return dict(
        version=SCHEMA_VERSION,
        model=model,
        mode="expected",
        margin=0.0,
        allowance=0,
        allowance_fraction=0.0,
        protection=0.0,
    )


def annotate(result, baseline):
    result["change_vs_inherited_only"] = {
        key: result[key] - baseline[key]
        for key in ("revenue", "penalties", "net_revenue", "walked")
    }
    return result


def print_score(name, result):
    inherited = result["by_origin"].get("inherited", {}).get("penalties", 0)
    new = result["by_origin"].get("new", {}).get("penalties", 0)
    delta = result.get("change_vs_inherited_only", {}).get("net_revenue", 0)
    print(
        f"{name}: net={result['net_revenue']:,.2f}; delta vs reject-new={delta:+,.2f}; "
        f"penalties={result['penalties']:,.2f} (inherited={inherited:,.2f}, new={new:,.2f}); "
        f"accepted/rejected={result['accepted_requests']}/{result['rejected_requests']}; "
        f"occupancy={result['occupancy']:.1%}",
        flush=True,
    )


def package(output, model_path, policy, name):
    with zipfile.ZipFile(output / name, "w", zipfile.ZIP_DEFLATED) as bundle:
        # Preserve the src package used by imports and pickled demand artifacts.
        for source in ("__init__.py", "solution.py", "policy.py", "prediction.py", "demand.py", "learned_demand.py"):
            bundle.write(ROOT / source, f"src/{source}")
        bundle.writestr("solution.py", "from src.solution import Solution\n")
        bundle.write(ROOT.parent / "requirements.txt", "requirements.txt")
        bundle.write(model_path, "artifacts/model.joblib")
        bundle.write(output / "demand.joblib", "artifacts/demand.joblib")
        bundle.writestr("artifacts/policy.json", json.dumps(policy, indent=2) + "\n")


def evaluate(data, args, output):
    manifest_path = output / "development.json"
    if not manifest_path.exists():
        raise ValueError("Run python train.py --stage train first")
    manifest = json.loads(manifest_path.read_text())
    if manifest["signature"] != training_signature(args.data):
        raise ValueError(
            "Training inputs/config changed. Run python train.py --stage train again."
        )
    for name, fingerprint in manifest["model_fingerprints"].items():
        if digest((output / f"{name}_development.joblib").read_bytes()) != fingerprint:
            raise ValueError(
                f"Development model changed: {name}. Run --stage train again."
            )
    context = ReplayContext(data, JULY, OCTOBER)
    validation = context.requests
    report = dict(
        schema_version=SCHEMA_VERSION,
        selection_period="July–September 2016",
        models={},
        validation_policies={},
        scenarios={},
        diagnostic_note="October–December is previously inspected, not a pristine holdout.",
    )
    training = data[data.checkout < JULY]
    rates = training.groupby("hotel").cancelled.mean()
    report["historical_rate_baseline"] = metrics(
        validation.cancelled,
        validation.hotel.map(rates).fillna(training.cancelled.mean()).to_numpy(),
    )
    probabilities = {}
    for name in manifest["models"]:
        scores, hit = cached_predictions(
            data,
            args.data,
            output / f"{name}_development.joblib",
            output / f"{name}_predictions.npz",
        )
        probabilities[name] = dict(zip(data.booking_id, scores))
        entry = probability_report(validation, scores[validation.index.to_numpy()])
        entry["cache_hit"] = hit
        report["models"][name] = entry
        print(
            f"Cancellation model: {name}; Brier={entry['brier']:.4f}; AUC={entry['auc']:.4f}",
            flush=True,
        )
    demand = forecast(data, JULY, output / "demand_development.joblib")
    first = next(iter(probabilities))
    baseline = context.run(probabilities[first], mode="reject")
    report["validation_policies"]["reject_new"] = annotate(baseline, baseline)
    capacity_config = settings(first)
    capacity_config["mode"] = "capacity"
    report["validation_policies"]["capacity"] = annotate(
        context.run(probabilities[first], **options(capacity_config, demand)), baseline
    )
    candidates = []
    specs = [(fraction, protection) for fraction in ALLOWANCES for protection in (0.0, 1.0)]
    specs.extend(spec for spec in POLICIES.values() if spec not in specs)
    for model in MODELS:
        for fraction, protection in specs:
            config = settings(model)
            config.update(allowance_fraction=fraction, protection=protection)
            name = f"{model}_extra_{fraction}_price_{protection}"
            result = annotate(
                context.run(probabilities[model], **options(config, demand)), baseline,
            )
            report["validation_policies"][name] = result
            if args.verbose:
                print_score(name, result)
            candidates.append((name, config, result))
    portfolio = {}
    for model in MODELS:
        model_candidates = [c for c in candidates if c[1]['model'] == model]
        for label, (fraction, protection) in POLICIES.items():
            candidate = next(c for c in model_candidates if c[1]['allowance_fraction'] == fraction
                             and c[1]['protection'] == protection)
            portfolio[f'{model}_{label}'] = dict(settings=candidate[1], validation=candidate[2],
                                                bundle=f'candidates/{model}_{label}.zip')
        local_best = max(model_candidates, key=lambda c: c[2]['net_revenue'])
        if not any(entry['settings'] == local_best[1] for entry in portfolio.values()):
            portfolio[f'{model}_local_best'] = dict(settings=local_best[1], validation=local_best[2],
                                                  bundle=f'candidates/{model}_local_best.zip')
    report['portfolio'] = portfolio
    benchmark = next(
        c
        for c in candidates
        if c[1]["model"] == "logistic" and c[1]["allowance_fraction"] == 0.10 and c[1]["protection"] == 1.0
    )
    best = max([benchmark] + candidates, key=lambda c: c[2]["net_revenue"])
    print(
        "\nTop validation policies (net revenue already includes relocation expense):",
        flush=True,
    )
    for _, config, result in sorted(
        candidates, key=lambda c: c[2]["net_revenue"], reverse=True
    )[:3]:
        extra_penalty = result["penalties"] - baseline["penalties"]
        print(
            f"  {config['model']}: extra expected demand {config['allowance_fraction']:.0%}, price factor {config['protection']:g}: "
            f"net={result['net_revenue']:,.2f}; extra penalties={extra_penalty:,.2f}; "
            f"accepted={result['accepted_requests']}",
            flush=True,
        )
    best_fraction = best[1]["allowance_fraction"]
    report["search"] = dict(
        allowances=list(ALLOWANCES),
        protection=[0, 1],
        winner_at_upper_bound=best_fraction == max(ALLOWANCES),
    )
    if report["search"]["winner_at_upper_bound"]:
        print(
            "Search warning: winner is at the largest allowance tested; optimum is not bracketed.",
            flush=True,
        )
    selected_name, selected, selected_result = best
    report["selected"] = selected
    report["selected_name"] = selected_name
    report["improvement_vs_current_benchmark"] = (
        selected_result["net_revenue"] - benchmark[2]["net_revenue"]
    )
    print(
        f"Local selection: {selected['model']}; extra expected demand {selected['allowance_fraction']:.0%}; price protection {'on' if selected['protection'] else 'off'}",
        flush=True,
    )
    print(f"Retained {len(portfolio)} named candidates across {len(MODELS)} models for official comparison.", flush=True)
    # Diagnostic comparisons never change the selected settings.
    for period, start, end in [
        ("validation", JULY, OCTOBER),
        ("later_diagnostic", OCTOBER, pd.Timestamp("2017-01-01")),
    ]:
        for scenario in ("inherited", "controlled"):
            ctx = (
                context
                if period == "validation" and scenario == "inherited"
                else ReplayContext(data, start, end, scenario)
            )
            zero = ctx.run(probabilities[first], mode="reject")
            results = {"reject_new": annotate(zero, zero)}
            configs = {
                "capacity": {**benchmark[1], "mode": "capacity"},
                "current_benchmark": benchmark[1],
                "selected": selected,
            }
            for label, config in configs.items():
                result = annotate(
                    ctx.run(probabilities[config["model"]], **options(config, demand)),
                    zero,
                )
                results[label] = result
                if args.verbose:
                    print_score(f"{period}/{scenario}/{label}", result)
            results["comparisons"] = {
                label: {
                    key: results["selected"][key] - results[label][key]
                    for key in ("net_revenue", "revenue", "penalties")
                }
                for label in ("capacity", "current_benchmark")
            }
            report["scenarios"][f"{period}/{scenario}"] = results
            if scenario == "inherited":
                chosen, prior = results["selected"], results["current_benchmark"]
                print(
                    f"{period}: previous={prior['net_revenue']:,.2f}; selected={chosen['net_revenue']:,.2f}; "
                    f"gain={chosen['net_revenue'] - prior['net_revenue']:+,.2f}; "
                    f"extra penalties={chosen['penalties'] - prior['penalties']:+,.2f}",
                    flush=True,
                )
    # Write metrics before final training, so a failed packaging step does not lose evaluation.
    (output / "portfolio.json").write_text(json.dumps(portfolio, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    (output / "benchmark_policy.json").write_text(
        json.dumps(benchmark[1], indent=2) + "\n"
    )
    (output / "selection_manifest.json").write_text(
        json.dumps(
            {
                "signature": evaluation_signature(args, output),
                "selection": selected,
                "benchmark": benchmark[1],
                "portfolio": portfolio,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Saved evaluation to {output / 'metrics.json'}; package with --stage package",
        flush=True,
    )


def finalize(data, args, output):
    selected = json.loads((output / "selection.json").read_text())
    benchmark = json.loads((output / "benchmark_policy.json").read_text())
    manifest = json.loads((output / "selection_manifest.json").read_text())
    portfolio = json.loads((output / "portfolio.json").read_text())
    if (
        manifest["signature"] != evaluation_signature(args, output)
        or manifest["selection"] != selected
        or manifest["benchmark"] != benchmark
        or manifest["portfolio"] != portfolio
    ):
        raise ValueError(
            "Selection inputs changed. Run --stage evaluate before packaging."
        )
    print("Fitting final historical artifacts...", flush=True)
    final_cutoff = pd.Timestamp("2017-01-01")
    model_paths = {}
    for model in MODELS:
        path = output / f'{model}_final.joblib'
        fit_model(data, model, final_cutoff).save(path)
        model_paths[model] = path
    forecast(data, final_cutoff, output / "demand.joblib")
    shutil.copyfile(model_paths[selected['model']], output / 'model.joblib')
    (output / "policy.json").write_text(json.dumps(selected, indent=2) + "\n")
    (output / 'final_manifest.json').write_text(json.dumps(dict(
        training_signature=training_signature(args.data), model=selected['model'],
        model_fingerprint=digest((output / 'model.joblib').read_bytes()),
        demand_fingerprint=digest((output / 'demand.joblib').read_bytes()),
    ), indent=2) + '\n')
    package(output, model_paths[selected['model']], selected, "submission.zip")
    shutil.copyfile(model_paths[benchmark['model']], output / 'benchmark_model.joblib')
    package(output, model_paths[benchmark['model']], benchmark, 'benchmark_submission.zip')
    (output / 'candidates').mkdir(exist_ok=True)
    for entry in portfolio.values():
        config = entry['settings']
        package(output, model_paths[config['model']], config, entry['bundle'])
    print(f"Retained candidate bundles: {output / 'candidates'} (index: portfolio.json)", flush=True)
    print(
        f"Saved results, selected submission, and benchmark fallback to {output.resolve()}",
        flush=True,
    )


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data = read_bookings(args.data)
    if args.stage in ("all", "train"):
        train_development(data, args, output)
    if args.stage in ("all", "evaluate"):
        evaluate(data, args, output)
    if args.stage in ("all", "package"):
        finalize(data, args, output)
    from src.learned_demand_experiment import run as run_learned_demand
    run_learned_demand(data, args.data, output / "learned_demand", args.stage)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="train.csv")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every policy and diagnostic scenario",
    )
    parser.add_argument(
        "--stage", choices=["all", "train", "evaluate", "package"], default="all"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    cli()
