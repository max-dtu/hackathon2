"""What demand usually looks like, learned from past bookings (used to hold rooms for guests still to come)."""
import numpy as np
import pandas as pd

MAX_LEAD = 400


def build_profile(df: pd.DataFrame) -> dict:
    """df: past bookings with a `cancelled` column. Counts only guests who really showed up."""
    df = df.copy()
    df["arrival_date"] = pd.to_datetime(df["arrival_date"])
    df["booking_date"] = pd.to_datetime(df["booking_date"])
    shows = df[df["cancelled"] == 0]
    first, last = df["arrival_date"].min(), df["arrival_date"].max()

    # one row per guest per night stayed
    rn = shows.loc[shows.index.repeat(shows["nights"]), ["hotel", "arrival_date", "booking_date"]]
    rn["night"] = rn["arrival_date"] + pd.to_timedelta(rn.groupby(level=0).cumcount(), unit="D")
    rn["lead"] = (rn["night"] - rn["booking_date"]).dt.days.clip(0, MAX_LEAD)
    rn = rn[(rn["night"] >= first) & (rn["night"] <= last)]

    profile = {"still_to_come": {}, "typical": {}, "hotel_typical": {}}
    for hotel, g in rn.groupby("hotel"):
        # still_to_come[d] = share of a night's guests who book fewer than d days before it
        counts = np.bincount(g["lead"], minlength=MAX_LEAD + 1)
        profile["still_to_come"][hotel] = np.concatenate([[0.0], np.cumsum(counts)[:-1]]) / counts.sum()
        per_night = g.groupby("night").size()
        profile["hotel_typical"][hotel] = float(per_night.mean())
        # guests wanting a room per night, by month: the most recent year we have for that month
        by_ym = per_night.groupby([per_night.index.year, per_night.index.month]).mean()
        for (year, month), v in by_ym.sort_index().items():
            profile["typical"][(hotel, int(month))] = float(v)   # later years overwrite earlier
    return profile
