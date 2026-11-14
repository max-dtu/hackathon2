"""Provisional replay. Confirm event ordering and walking rules with evaluate.py."""

import heapq

import pandas as pd

from src.policy import BookingPolicy

ROOMS = {"Downtown Hotel": 124, "Seaside Hotel": 103}


def allocate(accepted, initial_departures=None, rooms=None, audit=False):
    rooms = ROOMS if rooms is None else rooms
    departures = {h: list((initial_departures or {}).get(h, [])) for h in rooms}
    for values in departures.values():
        heapq.heapify(values)
    revenue = penalties = occupied_nights = 0.0
    seated = walked = 0
    by_origin = {}
    by_hotel = {
        h: dict(
            revenue=0.0,
            penalties=0.0,
            seated=0,
            walked=0,
            occupied_nights=0.0,
            foregone_revenue=0.0,
        )
        for h in rooms
    }
    walks = []
    # "Cheapest" provisionally means whole stay value; ties preserve source order.
    arrivals = accepted.copy()
    arrivals["value"] = arrivals["nightly_rate"] * arrivals["nights"]
    arrivals = arrivals.sort_values(
        ["arrival_date", "value"], ascending=[True, False], kind="stable"
    )
    for b in arrivals.to_dict("records"):
        if int(b["cancelled"]) or int(b["nights"]) <= 0:
            continue
        origin = b.get("origin", "unspecified")
        totals = by_origin.setdefault(
            origin,
            dict(revenue=0.0, penalties=0.0, walked=0, seated=0, foregone_revenue=0.0),
        )
        hotel_totals = by_hotel[b["hotel"]]
        queue = departures[b["hotel"]]
        while queue and queue[0] <= b["arrival_date"]:
            heapq.heappop(queue)
        if len(queue) < rooms[b["hotel"]]:
            revenue += b["value"]
            occupied_nights += b["nights"]
            seated += 1
            totals["revenue"] += b["value"]
            totals["seated"] += 1
            hotel_totals["revenue"] += b["value"]
            hotel_totals["seated"] += 1
            hotel_totals["occupied_nights"] += b["nights"]
            heapq.heappush(queue, b["checkout"])
        else:
            penalties += 2.5 * b["value"]
            walked += 1
            totals["penalties"] += 2.5 * b["value"]
            totals["foregone_revenue"] += b["value"]
            totals["walked"] += 1
            hotel_totals["penalties"] += 2.5 * b["value"]
            hotel_totals["foregone_revenue"] += b["value"]
            hotel_totals["walked"] += 1
            if audit:
                walks.append(
                    dict(
                        booking_id=b["booking_id"],
                        origin=origin,
                        hotel=b["hotel"],
                        arrival_date=str(b["arrival_date"].date()),
                        stay_value=float(b["value"]),
                        penalty=float(2.5 * b["value"]),
                    )
                )
    result = dict(
        net_revenue=revenue - penalties,
        revenue=revenue,
        penalties=penalties,
        seated=seated,
        walked=walked,
        occupied_nights=occupied_nights,
        by_origin=by_origin,
        by_hotel=by_hotel,
        foregone_revenue=sum(v["foregone_revenue"] for v in by_origin.values()),
    )
    if audit:
        result["walks"] = walks
        result["departures"] = departures
    for totals in by_hotel.values():
        totals["net_revenue"] = totals["revenue"] - totals["penalties"]
    return result


def reconstruct_in_house(data, start):
    """Seat pre-period arrivals under the same rules, rather than overfill rooms.

    Historical data does not say which guests were walked. Assume all pre-period
    requests were accepted, then simulate actual room allocation. Prior penalties
    do not enter this period's score.
    """
    past = data[(data.arrival_date < start) & (data.nights > 0)]
    state = allocate(past, audit=True)
    return {
        h: [date for date in dates if date > start]
        for h, dates in state["departures"].items()
    }


class ReplayContext:
    """Prepare immutable historical inputs once for many policy comparisons."""

    def __init__(self, data, start, end, scenario="inherited"):
        if scenario not in ("inherited", "controlled"):
            raise ValueError("Unknown replay scenario")
        self.start, self.end = pd.Timestamp(start), pd.Timestamp(end)
        self.data = data
        # Every stay arriving inside the period competes for rooms, including
        # stays that run past its end, so the calendar extends to the last checkout.
        cohort = data[
            (data.arrival_date >= self.start)
            & (data.arrival_date < self.end)
            & (data.nights > 0)
        ]
        last_night = max(
            self.end, cohort.checkout.max() if len(cohort) else self.end
        ) - pd.Timedelta(days=1)
        calendar = pd.date_range(self.start, last_night)
        self.period_nights = int((calendar < self.end).sum())
        self.capacity = {
            h: pd.Series(n, index=calendar, dtype=float) for h, n in ROOMS.items()
        }
        self.departures = (
            reconstruct_in_house(data, self.start)
            if scenario == "inherited"
            else {h: [] for h in ROOMS}
        )
        for hotel, dates in self.departures.items():
            for checkout in dates:
                self.capacity[hotel].loc[calendar[calendar < checkout]] -= 1
        self.commitments = (
            cohort[cohort.booking_date < self.start]
            if scenario == "inherited"
            else cohort.iloc[:0]
        )
        requests = cohort[
            (cohort.booking_date >= self.start) & (cohort.booking_date < self.end)
        ]
        self.requests = requests.sort_values("booking_date", kind="stable")
        # Neither true outcomes nor derived outcome dates are visible to policies.
        self.records = list(
            zip(
                self.requests.index,
                self.requests.drop(columns=["cancelled", "checkout"]).to_dict(
                    "records"
                ),
            )
        )

    def run(
        self,
        probabilities,
        mode="expected",
        margin=0.0,
        allowance=0,
        audit=False,
        allowance_fraction=0.0,
        protection=0.0,
        demand=None,
        hotel_settings=None,
        walk_cost=0.0,
    ):
        policy = BookingPolicy(
            self.commitments.drop(columns=["cancelled", "checkout"]),
            self.capacity,
            lambda b: probabilities[b["booking_id"]],
            mode=mode,
            margin=margin,
            allowance=allowance,
            allowance_fraction=allowance_fraction,
            protection=protection,
            demand=demand,
            hotel_settings=hotel_settings,
            walk_cost=walk_cost,
        )
        chosen = list(self.commitments.index)
        accepted_counts = {h: 0 for h in ROOMS}
        for index, booking in self.records:
            if policy.decide(booking):
                chosen.append(index)
                accepted_counts[booking["hotel"]] += 1
        accepted = self.data.loc[chosen].copy()
        accepted["origin"] = accepted.booking_date.map(
            lambda date: "inherited" if date < self.start else "new"
        )
        result = allocate(accepted, self.departures, audit=audit)
        if audit:
            result.pop("departures")
        result["initial_in_house"] = {
            h: len(dates) for h, dates in self.departures.items()
        }
        result["minimum_initial_free_rooms"] = {
            h: float(s.min()) for h, s in self.capacity.items()
        }
        result.update(
            accepted_requests=len(chosen) - len(self.commitments),
            requests=len(self.requests),
            rejected_requests=len(self.requests) - len(chosen) + len(self.commitments),
            commitments=len(self.commitments),
        )
        # Occupancy compares all sold nights with the rooms free inside the period.
        available = sum(
            series.iloc[: self.period_nights].sum() for series in self.capacity.values()
        )
        result["occupancy"] = (
            result["occupied_nights"] / available if available else 0.0
        )
        for hotel, totals in result["by_hotel"].items():
            count = int((self.requests.hotel == hotel).sum())
            totals.update(
                accepted_requests=accepted_counts[hotel],
                rejected_requests=count - accepted_counts[hotel],
            )
            available = self.capacity[hotel].iloc[: self.period_nights].sum()
            totals["occupancy"] = (
                totals["occupied_nights"] / available if available else 0.0
            )
        return result


def replay(
    data,
    start,
    end,
    probabilities,
    mode="expected",
    margin=0.0,
    allowance=0,
    audit=False,
    scenario="inherited",
    **kwargs,
):
    return ReplayContext(data, start, end, scenario).run(
        probabilities,
        mode=mode,
        margin=margin,
        allowance=allowance,
        audit=audit,
        **kwargs,
    )
