"""Fast ablation & grid search harness for policy4 across all 4 scenarios.

Caches ShowModel fits and batch predictions per (scenario, blend_nocountry, recency_decay)
so evaluating each policy configuration across all 4 scenarios takes ~1-2 seconds.

Usage:
  .venv/bin/python tune4.py stage1   # test blend_nocountry & recency_decay
  .venv/bin/python tune4.py stage2   # test demand_recency, pmin, gamma_p, w_max, walk_floor, bump_mult
  .venv/bin/python tune4.py stage3   # joint per-hotel (K, alpha, shift) tuning
"""
import functools
import itertools
import sys
import time

import numpy as np
import pandas as pd

import model as M
import policy4
import sim

DF = sim.load()
SCEN_NAMES = list(sim.SCENARIOS)
SCEN_SPLITS = {nm: sim.make_scenario(DF, *sim.SCENARIOS[nm]) for nm in SCEN_NAMES}
REQ_RECS = {
    nm: SCEN_SPLITS[nm][2].drop(columns=["cancelled"]).to_dict("records")
    for nm in SCEN_NAMES
}
REQ_CLEAN = {
    nm: SCEN_SPLITS[nm][2].drop(columns=["cancelled"])
    for nm in SCEN_NAMES
}
COM_CLEAN = {
    nm: SCEN_SPLITS[nm][1].drop(columns=["cancelled"])
    for nm in SCEN_NAMES
}

_FIT_CACHE = {}
_PRED_CACHE = {}
_ORIG_FIT = M.ShowModel.fit
_ORIG_PRED = M.ShowModel.predict


def _cached_fit(self, train):
    need_m2 = self.blend_nocountry > 0
    key = (len(train), str(train.arrival_date.max()), self.recency_decay,
           self.params1.get("max_iter"), self.params1.get("learning_rate"))
    if key not in _FIT_CACHE:
        orig_b = self.blend_nocountry
        self.blend_nocountry = max(orig_b, 0.35)  # always fit both m1 and m2 once
        fitted = _ORIG_FIT(self, train)
        self.blend_nocountry = orig_b
        _FIT_CACHE[key] = (fitted.m1, fitted.m2)
    m1, m2 = _FIT_CACHE[key]
    self.m1 = m1
    self.m2 = m2 if need_m2 else None
    self._cache_key = (key, need_m2, self.blend_nocountry)
    return self


def _cached_pred(self, df):
    if len(df) == 0:
        return np.array([], dtype=float)
    pkey = (getattr(self, "_cache_key", id(self)), len(df), int(df.booking_id.iloc[0]), int(df.booking_id.iloc[-1]))
    if pkey not in _PRED_CACHE:
        _PRED_CACHE[pkey] = _ORIG_PRED(self, df)
    return _PRED_CACHE[pkey]


_DEMAND_CACHE = {}
_ORIG_BUILD_DEMAND = policy4.Policy._build_demand


def _cached_build_demand(self, train):
    key = (len(train), str(train.arrival_date.max()), float(self.p["demand_recency"]))
    if key not in _DEMAND_CACHE:
        _ORIG_BUILD_DEMAND(self, train)
        _DEMAND_CACHE[key] = self.demand
    else:
        self.demand = _DEMAND_CACHE[key]


policy4.Policy._build_demand = _cached_build_demand
M.ShowModel.fit = _cached_fit
M.ShowModel.predict = _cached_pred


def eval_cfg(kw: dict) -> dict:
    cls = functools.partial(policy4.Policy, **kw)
    row = dict(kw)
    for nm in SCEN_NAMES:
        tr, com, req, cap = SCEN_SPLITS[nm]
        s, e = sim.SCENARIOS[nm]
        sol = cls(COM_CLEAN[nm], {h: c.copy() for h, c in cap.items()}, train=tr)
        sol.prepare(REQ_CLEAN[nm])
        acc = np.array([bool(sol.decide(r)) for r in REQ_RECS[nm]], dtype=bool)
        res = sim.score(pd.concat([com, req[acc]]), cap, s, e)
        row[nm] = res["net"]
        row[f"walk_{nm}"] = res["walked"]
    row["tune_sum"] = sum(row[nm] for nm in sim.TUNE_ON)
    return row


def run_configs(configs: list[dict], sort_by="tune_sum"):
    rows = []
    t0 = time.time()
    for i, kw in enumerate(configs):
        r = eval_cfg(kw)
        rows.append(r)
        print(
            f"[{i+1}/{len(configs)}] full={r['2016_full']:>9,.0f} H2={r['2016_H2']:>9,.0f} "
            f"Sep={r['2016_Sep']:>9,.0f} Q4={r['2016_Q4']:>8,.0f} sum={r['tune_sum']:>9,.0f} | {kw}",
            flush=True,
        )
    print(f"Done {len(configs)} configs in {time.time() - t0:.1f}s")
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    print("\n=== TOP BY tune_sum ===")
    print(res.sort_values("tune_sum", ascending=False).head(12).to_string(index=False))
    print("\n=== TOP BY 2016_full ===")
    print(res.sort_values("2016_full", ascending=False).head(12).to_string(index=False))
    return res


def stage1():
    """Ablate ShowModel blend_nocountry and recency_decay."""
    cfgs = []
    for blend in [0.0, 0.25, 0.35, 0.50]:
        for dec in [0.0, 0.002, 0.004]:
            cfgs.append(dict(blend_nocountry=blend, recency_decay=dec, K=1.20, alpha=2.0, shift=0.0))
    run_configs(cfgs)


def stage2():
    """Ablate individual policy refinements starting from Stage 1 winner."""
    base = dict(blend_nocountry=0.35, recency_decay=0.004, K=1.20, alpha=2.0, shift=0.0)
    groups = [
        ("demand_recency", [{"demand_recency": dr} for dr in [0.5, 1.5, 4.0]]),
        ("pmin", [{"pmin": pm} for pm in [0.03, 0.05, 0.10]]),
        ("gamma_p", [{"gamma_p": gp} for gp in [0.10, 0.20, 0.35]]),
        ("w_max", [{"w_max": wm} for wm in [0.15, 0.30, 0.50]]),
        ("walk_floor", [{"walk_floor": wf, "K_soft": ks} for wf, ks in [(30.0, 0.95), (60.0, 0.95), (30.0, 1.05), (60.0, 1.05), (90.0, 1.05)]]),
        ("bump_mult", [{"bump_mult": bm, "K_bump": kb} for bm, kb in [(1.5, 1.35), (2.0, 1.35), (2.5, 1.40)]]),
        ("adapt", [{"adapt": True}]),
    ]
    cfgs = [dict(base)]
    tags = ["base"]
    for gname, opts in groups:
        for opt in opts:
            cfgs.append({**base, **opt})
            tags.append(gname)
    res = run_configs(cfgs)
    res["group"] = tags
    return base, res


def stage3(base=None):
    """Per-hotel tuning of (K_dt, K_ss, alpha_dt, alpha_ss)."""
    if base is None:
        base = dict(blend_nocountry=0.35, recency_decay=0.004)
    cfgs = []
    for k_dt, k_ss in itertools.product([1.18, 1.24, 1.30], [1.12, 1.18, 1.24]):
        for a_dt, a_ss in itertools.product([1.6, 2.0, 2.5], [1.6, 2.0, 2.5]):
            cfgs.append({**base, "K_dt": k_dt, "K_ss": k_ss, "alpha_dt": a_dt, "alpha_ss": a_ss})
    res = run_configs(cfgs)
    res["combo_rank"] = res["tune_sum"].rank() + res["2016_full"].rank()
    print("\n=== TOP BY 2016_full + tune_sum rank ===")
    print(res.sort_values("combo_rank", ascending=False).head(10).to_string(index=False))
    return res


def stage23():
    """Run Stage 2 ablations, combine generalizing improvements, then run Stage 3 per-hotel grid."""
    base, s2 = stage2()
    b_sum = float(s2.loc[0, "tune_sum"])
    b_full = float(s2.loc[0, "2016_full"])
    winner = dict(base)
    param_keys = {"demand_recency", "pmin", "gamma_p", "w_max", "walk_floor", "K_soft", "bump_mult", "K_bump", "adapt"}
    for gname, grp in s2.iloc[1:].groupby("group", sort=False):
        # Keep refinements that improve both tune_sum and 2016_full (or don't hurt either by > 0.2%)
        ok = grp[(grp["tune_sum"] >= b_sum * 0.998) & (grp["2016_full"] >= b_full * 0.998) &
                 ((grp["tune_sum"] > b_sum) | (grp["2016_full"] > b_full))]
        if len(ok):
            best = ok.assign(cr=ok["tune_sum"].rank() + ok["2016_full"].rank()).sort_values(
                ["cr", "tune_sum"], ascending=False
            ).iloc[0]
            picked = {k: (bool(best[k]) if isinstance(best[k], (bool, np.bool_)) else float(best[k]))
                      for k in param_keys if k in best and pd.notna(best[k])}
            # Only apply keys explicitly tested in that group
            if gname == "demand_recency":
                winner["demand_recency"] = picked["demand_recency"]
            elif gname == "pmin":
                winner["pmin"] = picked["pmin"]
            elif gname == "gamma_p":
                winner["gamma_p"] = picked["gamma_p"]
            elif gname == "w_max":
                winner["w_max"] = picked["w_max"]
            elif gname == "walk_floor":
                winner["walk_floor"] = picked["walk_floor"]
                winner["K_soft"] = picked["K_soft"]
            elif gname == "bump_mult":
                winner["bump_mult"] = picked["bump_mult"]
                winner["K_bump"] = picked["K_bump"]
            elif gname == "adapt":
                winner["adapt"] = picked["adapt"]
    print(f"\n>>> Stage 2 combined winner candidate: {winner}", flush=True)
    comb_res = eval_cfg(winner)
    print(f">>> Stage 2 combined score: full={comb_res['2016_full']:,.0f} (base {b_full:,.0f}) "
          f"tune_sum={comb_res['tune_sum']:,.0f} (base {b_sum:,.0f})", flush=True)
    stage3_base = dict(winner)
    stage3_base.pop("K", None)
    stage3_base.pop("alpha", None)
    stage3(stage3_base)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stage1"
    globals()[cmd]()
