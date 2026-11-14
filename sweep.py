"""Tune the decision rule's knobs. Usage: .venv/bin/python sweep.py START END"""
import sys, warnings
warnings.filterwarnings("ignore")
from simulate import Scenario, load, run
from solution import ShowUpModel, Solution

start, end = sys.argv[1], sys.argv[2]
sc = Scenario(load(), start, end)
model = ShowUpModel(sc.history.drop(columns=["deposit"], errors="ignore"))
print(f"{start}..{end}", flush=True)
for walk in sys.argv[3:] or ("simple", "cheapest"):
    for shift in (-0.6, -0.4, -0.2, 0.0, 0.2):
        r = run(Solution, sc, model=model, p_shift=shift, walk_model=walk)
        print(f"walk={walk:9} p_shift={shift:+.1f}  net={r['net']:>10,}  accepted={r['accepted']:>6}  "
              f"walked={r['walked']:>5}  {r['seconds']}s", flush=True)
