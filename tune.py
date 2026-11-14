"""Grid search: .venv/bin/python tune.py <policy_module> '<json grid>'
Score = sum of net over sim.TUNE_ON scenarios (>=12 months of training); 2016_full shown as stress test."""
import itertools, sys, functools, json
import pandas as pd
import sim, model as M
import importlib

modname = sys.argv[1]
grid = json.loads(sys.argv[2])
pol = importlib.import_module(modname)
df = sim.load()
_cache = {}
_orig_fit = M.ShowModel.fit


def cached_fit(self, train):
    key = (len(train), str(pd.to_datetime(train.arrival_date).max()), str(pd.to_datetime(train.arrival_date).min()))
    if key not in _cache:
        _cache[key] = _orig_fit(self, train)
    return _cache[key]


M.ShowModel.fit = cached_fit
keys = list(grid)
names = list(sim.SCENARIOS)
res = []
for vals in itertools.product(*grid.values()):
    kw = dict(zip(keys, vals))
    cls = functools.partial(pol.Policy, **kw)
    row = dict(kw)
    for nm in names:
        s, e = sim.SCENARIOS[nm]
        row[nm] = sim.run(cls, df, s, e, verbose=False)["net"]
    row["score"] = sum(row[nm] for nm in sim.TUNE_ON)
    res.append(row)
    print({k: row[k] for k in keys}, f"score={row['score']:,.0f}", {nm: f"{row[nm]:,.0f}" for nm in names}, flush=True)
r = pd.DataFrame(res)
pd.set_option("display.width", 250)
print(r.sort_values("score", ascending=False).head(12).to_string())
