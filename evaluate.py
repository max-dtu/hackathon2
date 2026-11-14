"""Net revenue one team's solution earns on the 2017 hold-out.

    from evaluate import evaluate
    evaluate(Solution)

Reads train.csv and test.csv (from make_splits.py) from the current directory.
The contract and the rules the teams see are in solution.py.
"""
import numpy as np
import pandas as pd

ROOMS = {"Downtown Hotel": 124, "Seaside Hotel": 103}
WALK_PENALTY = 2.5
RATE_CAP = 1000.0  # exclude one outlier booking that has nightly_rate 5400
NIGHTS = pd.date_range("2017-01-01", "2017-12-31")


def stay(first_night, nights):
    """The nights of a stay, as indices into NIGHTS."""
    return slice(max(0, first_night), min(len(NIGHTS), first_night + nights))


def initial_conditions(train, requests):
    """What every team starts 2017 with: the bookings already confirmed before 2017-01-01,
    and the rooms left free each night by 2016 guests still in house."""
    commitments = requests[requests.booking_date < NIGHTS[0]].reset_index(drop=True)
    in_house = train[(train.cancelled == 0)
                     & (train.arrival_date + pd.to_timedelta(train.nights, unit="D") > NIGHTS[0])]
    capacity = {}
    for hotel, rooms in ROOMS.items():
        occupied = np.zeros(len(NIGHTS))
        for guest in in_house[in_house.hotel == hotel].itertuples():
            occupied[stay((guest.arrival_date - NIGHTS[0]).days, guest.nights)] += 1
        # The smaller scenario hotel cannot hold more 2016 guests than it has rooms.
        capacity[hotel] = pd.Series(np.maximum(rooms - occupied, 0), index=NIGHTS)
    return commitments, capacity


def confirmed_bookings(Solution, requests, commitments, capacity):
    """Replay 2017 through the solution and return the booking_ids it confirmed."""
    solution = Solution(commitments.copy(), {hotel: free.copy() for hotel, free in capacity.items()})
    confirmed = set(commitments.booking_id)
    new_requests = requests[requests.booking_date >= NIGHTS[0]]
    for booking in new_requests.sort_values(["booking_date", "booking_id"]).to_dict("records"):
        decision = solution.decide(booking)
        if not isinstance(decision, (bool, np.bool_)):
            raise TypeError(f"decide() must return a bool, got {type(decision).__name__}")
        if decision:
            confirmed.add(booking["booking_id"])
    return confirmed


def net_revenue(test, confirmed, capacity):
    shows = test[test.booking_id.isin(confirmed) & (test.cancelled == 0)]
    net = 0.0
    for hotel, free in capacity.items():
        occupied = np.zeros(len(NIGHTS))
        arrivals = shows[shows.hotel == hotel].sort_values(["day", "value"], ascending=[True, False])
        for guest in arrivals.itertuples():
            # Checking the arrival night is enough: free rooms only grow as 2016 guests leave.
            if occupied[guest.day] < free.iloc[guest.day]:
                occupied[stay(guest.day, guest.nights)] += 1
                net += guest.value
            else:
                net -= WALK_PENALTY * guest.value
    return net


def evaluate(Solution):
    """Print the net revenue the Solution class earns on the 2017 hold-out."""
    train = pd.read_csv("train.csv", parse_dates=["arrival_date", "booking_date"])
    test = pd.read_csv("test.csv", parse_dates=["arrival_date", "booking_date"])
    requests = test.drop(columns="cancelled")
    commitments, capacity = initial_conditions(train, requests)
    confirmed = confirmed_bookings(Solution, requests, commitments, capacity)

    test["day"] = (test.arrival_date - NIGHTS[0]).dt.days
    test["value"] = test.nightly_rate.clip(0, RATE_CAP) * test.nights
    print(f"net revenue EUR {net_revenue(test, confirmed, capacity):,.0f}")
