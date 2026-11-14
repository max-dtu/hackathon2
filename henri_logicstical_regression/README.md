# Hotel booking robot

Six model variants (**logistic regression, random forest and CatBoost, each ordinary or calibrated**) feed the same booking policy, which checks
expected room demand and whether scarce rooms are worth saving for future guests.
The objective is accommodated revenue minus relocation expense.

## Official evaluation kit

Use [EVALUATION.md](EVALUATION.md) for the final-window checklist. The comparison runner
uses frozen bundles, fresh processes, timeouts and incremental scoreboards:

```sh
python3 evaluation_runner.py --list
python3 evaluation_runner.py --preflight --all
python3 evaluation_runner.py --evaluator /path/to/kit/evaluate.py
```

Only the official evaluator reads the 2017 dataset. The default comparison has a
15-minute budget, leaving time to verify and submit the winning ZIP.

## Commands

With the project virtual environment active:

```sh
python3 main.py                    # train, compare policies, package winner
python3 main.py --stage evaluate   # compare policies using saved predictions
python3 main.py --stage package    # rebuild submission from the last selection
python3 main.py --verbose          # show every experiment and diagnostic
python3 -m src.audit_penalties          # explain the relocation costs
python3 -m src.tune                     # tune policy settings on year-shaped replays
python3 -m src.tune --apply             # ...and write them to policy.json + submission.zip
python3 -m unittest src.test_pipeline src.test_improvements src.test_portfolio src.test_review_fixes src.test_catboost -v     # check correctness
```

For a fresh environment, run `uv venv --python 3.14` and
`uv pip install --python .venv/bin/python -r requirements.txt` first. Without activating
it, replace `python3` with `.venv/bin/python`.

Default output shows probability metrics for all six variants, the top three policies, the local selection, and revenue
comparisons with the previous policy. Detailed results remain in `artifacts/metrics.json`.
Use `--output DIRECTORY` to keep an experiment separate. Re-running overwrites its outputs.

## Clean baseline and CatBoost experiment

Keep this experiment separate from the existing tuned submission. Run these commands
from this project folder, in order:

```sh
python3 main.py --output artifacts/clean_catboost
python3 -m src.tune --output artifacts/clean_catboost --jobs 4 --apply
```

The first command retrains all six variants **without deposit**, compares the same
policies for each, and packages the initial comparison winner and retained candidates.
CatBoost starts with 500 iterations, depth 6, learning rate 0.05, Logloss, seed 42,
and four CPU training threads. Inference uses one thread. No validation-based early
stopping or hyperparameter search is used. Both calibrated variants and ordinary
variants keep their existing chronological training rules.

The second command tunes only the comparison winner, using the existing three-period
search, and replaces this experiment's `policy.json` and `submission.zip`. The original
comparison remains in `selection.json` and `metrics.json`; the tuned result is recorded
in `tuning.json`. These overlapping, inspected periods are diagnostics, not an unseen
test. Do not run packaging again after tuning unless intentionally restoring the initial
comparison winner.

Use the isolated experiment's extracted `submission.zip` for evaluation. The
repository-root entry point still loads the original `artifacts/` directory;
those older models need retraining before use with the deposit-free source.

Tuning checks the current training signature and the final model/demand fingerprints
in `final_manifest.json`. Older outputs must be rebuilt with `main.py` before tuning;
this prevents applying fresh tuning results to stale, deposit-based models. Model and
prediction signatures include the CatBoost version. Amendments and waitlist days remain
inputs under the assumption they are valid at booking time.

## What is tested

All six variants use the same validation period and the same policy search. The two policy controls are:

- **Extra expected demand:** 0%, 5%, 10%, 15%, 20%, 30%, 40%, 60%, 80%, 100% above room capacity.
  This changes how much predicted arrival demand can be accepted; it is not a percentage
  of reservation count and does not alter cancellation probabilities.
- **Price protection:** on or off. When on, a booking must justify using rooms that
  historical demand suggests could sell for more later.

The local winner maximizes validation net revenue, including relocation costs. The
10%-allowance/price-protection logistic policy is the benchmark and wins ties. Search
extends to 100% extra expected demand; a warning appears if the winner is at that boundary.
Calibrated variants use sigmoid scaling fitted on a separate chronological slice:
for July validation, train the base classifier on stays completed before April, then
calibrate on requests booked April–June and completed before July. For final January
artifacts, use the equivalent October–December calibration slice. Ordinary models train
on all stays completed before their cutoff. Thus comparisons include the cost of
reserving training data for calibration; calibration is not fitted on validation labels.

## Final policy tuning (`src.tune`)

The July–September comparison above only contains requests booked and completed
within three months. The official run replays a whole year, where most arrivals,
including long-lead ones, go through `decide()`. `python3 -m src.tune` therefore
replays three periods that start in January, April and July 2016 and run to the end
of the data, each with a model and demand forecast fitted only on earlier stays.

It searches the overbooking allowance (10–40%), price protection (1–2), the
uncertainty margin (0 or 0.5 standard deviations) and the walk-cost rule, and scores
every setting by its average share of each period's best net revenue. A hotel keeps
its own settings only if they beat the shared ones by more than 1%.

**Walk-cost rule** (`walk_cost` in `policy.json`, off by default): instead of a hard
occupancy cap, a booking is accepted when its stay value exceeds the harm it is
expected to cause. Each night is priced at 3.5 × a typical cheap rate × the chance
the hotel is full that night (a walked guest loses their revenue and costs 2.5 times
it, and the cheapest arrivals are walked first), plus the price-protection amount.

`--apply` writes the result to `artifacts/policy.json` and rebuilds
`artifacts/submission.zip` with the already-trained final model. Do not run
`main.py --stage package` afterwards: it would restore the earlier selection.
The repository-root `solution.py` re-exports this folder's `Solution`, so
`from solution import Solution` works from the repository root too.

## Retained candidates for official evaluation

Packaging preserves five named policies for **each of the six model variants**:

- `early`: 10% extra expected demand, price protection off.
- `balanced`: 10% extra expected demand, price factor 1.
- `future_focused`: 10% extra expected demand, stronger price factor 1.5.
- `conservative`: 5% extra expected demand, price factor 1.
- `aggressive`: 15% extra expected demand, price factor 1.

If a model's local winner differs from all five, it is also retained as `local_best`.
Nothing is eliminated just because it loses local validation. All candidates share
the same tested code; only the fitted model and policy settings differ.

Find named archives under `artifacts/candidates/`, for example
`logistic_balanced.zip`, `forest_early.zip`, `logistic_calibrated_balanced.zip`,
and `forest_calibrated_local_best.zip`.
`artifacts/portfolio.json` records each archive's settings and local validation result.
The default `artifacts/submission.zip` remains the local winner, and
`artifacts/benchmark_submission.zip` remains the benchmark fallback. Select among
retained candidates using the organizers' evaluator in the final comparison window.
Extract each archive into a separate directory and evaluate it in a fresh Python process
to avoid reusing another candidate's imported modules or artifacts.

## How it works

1. Predict cancellation probability from supplied booking attributes. Arrival probability
   is one minus that score. Numeric values are imputed/scaled and categories are one-hot
   encoded for logistic regression and random forest; CatBoost uses native categorical
   features and numeric missing values. Deposit is excluded from all models because
   it is unavailable in evaluation. IDs/target/year numbers are excluded; dates become seasonal features.
   Agent/company identifiers remain categorical. Amendments and waitlist information
   are included as requested; confirm that they are available at decision time.
2. Check expected demand on **every night** of a stay. Acceptances update commitments
   immediately. No hidden cancellation outcomes are supplied to this policy.
3. If price protection is on, compare whole-stay price with the sum of historical
   reservation prices for its nights. Demand curves use past positive-price arrivals,
   including zero-demand dates, grouped by hotel/month/remaining lead time. Sparse
   months pool adjacent months, then hotel history. Same-day nights have zero protection.
   These are empirical heuristics, not a guarantee that late guests pay more.

## Evaluation and artifacts

Models and demand forecasts use stays completed before July 2016. Compare models on
requests booked July–September that arrive before October 1, including stays that
finish later. Probability reports use exactly the replay request cohort. October–December is a later
**diagnostic**, not an unseen test; it never chooses the winner. Model fingerprints and
prediction caches prevent accidentally reusing incompatible training outputs.

The primary scenario includes already-confirmed reservations. An empty-start diagnostic
excludes earlier bookings to isolate the new policy's behavior. Each compares rejecting
new requests, physical-capacity control, the previous policy, and the winner. Detailed
outputs include revenue, relocation expense, foregone revenue, acceptance counts,
occupancy, and hotel breakdowns. Compare **total penalty differences**: a new booking
can displace an inherited guest, so the guest's origin does not identify who caused it.

Final packaging trains each model once on stays completed before January 2017, then reuses it across its policy bundles:

- `artifacts/submission.zip`: selected robot and historical artifacts.
- `artifacts/benchmark_submission.zip`: previous-policy fallback.

All bundles load `from solution import Solution`, with the interface
`Solution(commitments, capacity).decide(booking) -> bool`. The archives preserve the
`src/` package and store artifacts at the extracted root. Supporting Python
modules and pinned dependencies are included. Confirm the organizers allow these files.
Never read 2017 evaluation data. Load joblib files only from trusted sources.

## Simulator limitations

One booking occupies one room. Checkouts occur before arrivals; existing guests keep
their rooms. Same-day arrivals are ranked by whole-stay value, with cheaper overflow
guests earning zero and costing 2.5 times stay value. Ties preserve CSV order. A cancelled
label means no arrival; separate no-show behavior is unconfirmed.

Historical in-house guests are reconstructed by replaying earlier arrivals under an
accept-all assumption. Prior revenue/costs are excluded, but their rooms are occupied.
Every stay arriving inside the period is replayed and scored in full, including
stays that run past its end; the room calendar extends to the last checkout. Arrivals
after the period are not replayed. These choices can materially affect
results and must be checked against the official evaluator, which is not yet available.

Relocation `penalties` are an expense. `foregone_revenue` is the separate stay value not
earned, already absent from revenue; it is not subtracted twice. Supplied rates, including
zero/negative rates, enter allocation unchanged. Zero-night bookings and nights missing
from the capacity calendar are rejected by capacity-based policies.

## Learned demand in main.py

`python3 main.py` now also trains, evaluates, and packages the learned room-price
protection challenger. Its stages follow `--stage train`, `--stage evaluate`, and
`--stage package`; evaluate reuses saved models and rejects stale inputs.

The console prints paired baseline/learned revenue for July–September and
October–December. Detailed results are in `artifacts/learned_demand/comparison.json`.
Candidates are `artifacts/learned_demand/baseline.zip` and
`artifacts/learned_demand/learned_demand.zip`. `--output DIRECTORY` places these
under `DIRECTORY/learned_demand/`.

This comparison uses logistic cancellation probabilities and the tuned 20%
allowance, 1.25 price protection, and walk-cost rule for both forecasts. Its totals
therefore differ from the cancellation-model search above. The challenger remains
a named candidate; `submission.zip` is still the cancellation-policy search winner.
Both periods have already been inspected and are historical diagnostics.

To rerun only the demand experiment, use
`python3 -m src.learned_demand_experiment` with the same optional stage argument.
Older demand experiment outputs need a fresh training run to create the provenance
manifest required by the integrated workflow.
