"""Explain relocation costs without retraining or accessing 2017 evaluation data."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.demand import FutureDemand
from src.prediction import CancellationModel, read_bookings
from src.simulator import (
    ROOMS,
    allocate,
    reconstruct_in_house,
    replay,
)
from src.train import options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2016-07-01")
    parser.add_argument("--end", default="2016-10-01")
    args = parser.parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    data = read_bookings()
    selected = Path("artifacts/selection.json")
    settings = json.loads(
        (selected if selected.exists() else Path("artifacts/policy.json")).read_text()
    )
    model = CancellationModel.load(f"artifacts/{settings['model']}_development.joblib")
    scores = dict(zip(data.booking_id, model.predict_batch(data)))
    departures = reconstruct_in_house(data, start)
    inherited = data[
        (data.arrival_date >= start)
        & (data.arrival_date < end)
        & (data.nights > 0)
        & (data.booking_date < start)
    ].copy()
    inherited["origin"] = "inherited"
    before = allocate(inherited, departures)
    demand = FutureDemand.load("artifacts/demand_development.joblib")
    after = replay(data, start, end, scores, audit=True, **options(settings, demand))
    print(
        "Starting state (raw historical overlap -> reconstructed accommodated guests):"
    )
    for hotel, rooms in ROOMS.items():
        raw = data[
            (data.hotel == hotel)
            & (data.arrival_date < start)
            & (data.checkout > start)
            & (data.cancelled == 0)
        ]
        print(f"  {hotel}: {len(raw)} -> {len(departures[hotel])}; rooms={rooms}")
    for title, result in [
        ("Inherited only: reject every new request", before),
        ("Selected policy", after),
    ]:
        print(
            f"\n{title}: revenue={result['revenue']:,.2f}; penalties={result['penalties']:,.2f}; "
            f"net={result['net_revenue']:,.2f}; walked={result['walked']}; "
            f"foregone revenue={result['foregone_revenue']:,.2f}"
        )
        print(json.dumps(result["by_origin"], indent=2))
    print(
        f"\nPolicy change versus inherited only: revenue "
        f"{after['revenue'] - before['revenue']:+,.2f}; penalties "
        f"{after['penalties'] - before['penalties']:+,.2f}; net "
        f"{after['net_revenue'] - before['net_revenue']:+,.2f}"
    )
    print("\nLargest five relocation expenses:")
    for row in sorted(after["walks"], key=lambda b: b["penalty"], reverse=True)[:5]:
        print(row)


if __name__ == "__main__":
    main()
