"""Robustness checks for solution.py: odd inputs the judges' 2017 data might contain.

Usage: .venv/bin/python check_solution.py
Builds a mock "2017" from 2016 bookings shifted 52 weeks (outcomes removed) -- only for
exercising the code path and timing, not for scoring (the model has seen 2016).
"""
import datetime as dt
import io
import math
import subprocess
import sys
import tempfile
import time
import warnings
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from solution import Solution, _key  # noqa: E402

ROOMS = {"Downtown Hotel": 124, "Seaside Hotel": 103}
DAYS = pd.date_range("2017-01-01", "2017-12-31")
results = []


def check(name, cond, detail=""):
    results.append(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def mock_2017():
    d = pd.read_csv(HERE / "train.csv", parse_dates=["arrival_date", "booking_date"])
    d = d[d.arrival_date.dt.year == 2016].copy()
    for c in ("arrival_date", "booking_date"):
        d[c] += pd.Timedelta(days=364)
    d = d[d.arrival_date.dt.year == 2017].drop(columns=["cancelled", "deposit"])
    commit = d[d.booking_date < "2017-01-01"]
    req = d[d.booking_date >= "2017-01-01"].sort_values(["booking_date", "booking_id"])
    return commit, req


def capacity(index=DAYS):
    return {h: pd.Series(r, index=index) for h, r in ROOMS.items()}


commit, req = mock_2017()
records = req.to_dict("records")

# 1. Full run the way evaluate.py calls it --------------------------------------------------
err = io.StringIO()
with redirect_stderr(err):
    t0 = time.time()
    sol = Solution(commit, capacity())
    setup = time.time() - t0
    t0 = time.time()
    out = [sol.decide(r) for r in records]
    run = time.time() - t0
check("setup completes", True, f"{setup:.1f}s")
check("every decision is a bool", all(isinstance(o, bool) for o in out))
check("decisions are fast", run / len(out) < 0.01,
      f"{len(out):,} requests in {run:.0f}s, {1000 * run / len(out):.1f} ms each")
check("accepts a sensible share", 0.2 < np.mean(out) < 0.9, f"{np.mean(out):.0%} accepted")
check("no fallback used on normal data", "falling back" not in err.getvalue(), err.getvalue().strip())

# 2. Capacity indexed by datetime.date, with a gap and a NaN --------------------------------
first = records[:3000]
base = Solution(commit, capacity())
ref = [base.decide(r) for r in first]
alt = Solution(commit, capacity(pd.Index([x.date() for x in DAYS])))
check("capacity indexed by datetime.date gives identical decisions",
      [alt.decide(r) for r in first] == ref)
gappy = {h: s.drop(s.index[10:12]).astype(float) for h, s in capacity().items()}
gappy["Downtown Hotel"].iloc[50] = np.nan
err = io.StringIO()
with redirect_stderr(err):
    g = Solution(commit, gappy)
    ok = all(isinstance(g.decide(r), bool) for r in first)
check("capacity with missing days and a NaN still works", ok and "falling back" not in err.getvalue())

# 3. Category formats: agent 9 / 9.0 / "9" / "9.0"; None vs NaN ---------------------------
b = dict(records[0])
b["agent"] = 9.0
probs = []
for v in (9, 9.0, "9", "9.0", np.int64(9)):
    probs.append(base.model.predict_one({**b, "agent": v}))
check("agent 9 / 9.0 / '9' / '9.0' predict the same", max(probs) - min(probs) < 1e-12,
      f"p={probs[0]:.3f}")
p_none = base.model.predict_one({**b, "company": None})
p_nan = base.model.predict_one({**b, "company": np.nan})
check("company None == NaN", abs(p_none - p_nan) < 1e-12)
check("_key canonical forms", _key("PRT") == "PRT" and _key(None) == "nan" and _key(" 12 ") == "12")

# 4. Odd single bookings: must return a bool and never raise -------------------------------
s = Solution(commit, capacity())
odd = {
    "unknown hotel": {**b, "hotel": "Mountain Hotel"},
    "unseen country": {**b, "country": "XYZ"},
    "unseen agent": {**b, "agent": 99999},
    "nights missing": {k: v for k, v in b.items() if k != "nights"},
    "nights NaN": {**b, "nights": float("nan")},
    "nights 0": {**b, "nights": 0, "weekend_nights": 0, "weekday_nights": 0},
    "column missing": {k: v for k, v in b.items() if k != "company"},
    "arrival as string": {**b, "arrival_date": "2017-06-15"},
    "arrival in 2018": {**b, "arrival_date": pd.Timestamp("2018-01-03")},
    "stay runs past Dec 31": {**b, "arrival_date": pd.Timestamp("2017-12-30"), "nights": 5},
    "rate NaN": {**b, "nightly_rate": float("nan")},
    "rate negative": {**b, "nightly_rate": -10.0},
    "rate as string": {**b, "nightly_rate": "120.5"},
    "children None": {**b, "children": None},
    "extra deposit column": {**b, "deposit": "No Deposit"},
    "arrival before 2017": {**b, "arrival_date": pd.Timestamp("2016-12-30"), "nights": 4},
}
for name, bk in odd.items():
    try:
        r = s.decide(bk)
        check(f"odd booking: {name}", isinstance(r, bool), str(r))
    except Exception as exc:  # pragma: no cover
        check(f"odd booking: {name}", False, repr(exc))
e = s.explain({**b, "nightly_rate": float("nan")})
check("NaN or $0 rate is rejected", not e["accept"] and not s.explain({**b, "nightly_rate": 0.0})["accept"])

# 5. Empty commitments ----------------------------------------------------------------------
try:
    s0 = Solution(pd.DataFrame(), capacity())
    check("empty commitments DataFrame", isinstance(s0.decide(b), bool))
except Exception as exc:
    check("empty commitments DataFrame", False, repr(exc))

# 6. Imported from another working directory -------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    code = ("import sys, pandas as pd; sys.path.insert(0, %r); from solution import Solution;"
            "d=pd.date_range('2017-01-01','2017-12-31');"
            "s=Solution(pd.DataFrame(), {'Downtown Hotel': pd.Series(124, index=d)}); print('ok')"
            % str(HERE))
    proc = subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=tmp,
                          capture_output=True, text=True)
    check("works when run from another folder", proc.stdout.strip() == "ok", proc.stderr.strip()[-200:])

# 7. A decision that blows up uses the fallback instead of crashing --------------------------
s = Solution(commit, capacity())
s.model.predict_one = lambda b: 1 / 0  # simulate an internal failure
err = io.StringIO()
with redirect_stderr(err):
    r = s.decide(records[0])
check("internal error -> fallback rule, no crash", isinstance(r, bool) and "falling back" in err.getvalue())

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
