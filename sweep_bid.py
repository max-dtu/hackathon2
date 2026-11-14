"""Tune bid_scale (saving rooms for later, better bookings). Usage: sweep_bid.py START END"""
import sys, warnings
warnings.filterwarnings("ignore")
from simulate import Scenario, load, run
from solution import ShowUpModel, Solution

start, end = sys.argv[1], sys.argv[2]
sc = Scenario(load(), start, end)
model = ShowUpModel(sc.history.drop(columns=["deposit"], errors="ignore"))
print(f"{start}..{end}", flush=True)
for scale in (0.0, 0.25, 0.5, 0.75, 1.0):
    r = run(Solution, sc, model=model, bid_scale=scale)
    print(f"bid_scale={scale:.2f}  net={r['net']:>10,}  accepted={r['accepted']:>6}  "
          f"walked={r['walked']:>5}  {r['seconds']}s", flush=True)
