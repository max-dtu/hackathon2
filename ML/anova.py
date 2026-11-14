"""One-way ANOVA: which ways of grouping bookings really separate stayers from cancellers?

For each grouping (deposit type, sales segment, booking period, ...) a one-way ANOVA asks:
do the groups differ in their stay rate by more than chance would produce? Two numbers matter:

  strength  eta squared: the share of all variation in "stayed or not" that the grouping
            explains. With 78,000 bookings every p-value is tiny, so strength is what counts.
  drift     how far each group's stay rate moved between the first and second half of the
            data. A grouping that is strong but drifts will mislead the model next year.

A grouping is worth building decisions on when it is strong AND steady.

    python ML/anova.py            prints the tables, saves a plot to ML/plots/
    python ML/anova.py --show     also opens the plot in a window
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import f_oneway

import common
from booking_model import booking_period
from common import plt

FACTORS = ["deposit", "sales_segment", "booking_period", "country", "agent", "sales_channel",
           "guest_type", "room_type", "meal", "hotel", "special_requests", "returning_guest",
           "arrival_month", "arrival_weekday"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["stays"] = 1 - d["cancelled"]
    d["booking_period"] = d["days_ahead"].map(booking_period)
    d["arrival_weekday"] = d["arrival_date"].dt.day_name()
    d["special_requests"] = d["special_requests"].clip(upper=3)
    for col in ("agent", "country"):                      # keep big groups, pool the rest
        top = d[col].value_counts()
        d[col] = d[col].where(d[col].isin(top[top >= 200].index), "other").astype(str)
    return d


def anova(d: pd.DataFrame, factor: str, target: str) -> dict:
    groups = [g[target].to_numpy(float) for _, g in d.groupby(factor) if len(g) >= 30]
    f, p = f_oneway(*groups)
    allv = np.concatenate(groups)
    between = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
    return {"groups": len(groups), "F": f, "p": p, "eta2": between / ((allv - allv.mean()) ** 2).sum()}


def drift(d: pd.DataFrame, factor: str) -> float:
    """Average move in a group's stay rate from the first to the second half (weighted by size)."""
    half = d["arrival_date"] < d["arrival_date"].quantile(0.5)
    a = d[half].groupby(factor)["stays"].agg(["mean", "size"])
    b = d[~half].groupby(factor)["stays"].agg(["mean", "size"])
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    j = j[(j["size_a"] >= 30) & (j["size_b"] >= 30)]
    return float(np.average((j["mean_b"] - j["mean_a"]).abs(), weights=j["size_a"] + j["size_b"]))


def plot(t: pd.DataFrame):
    t = t.sort_values("eta2")
    names = [n.replace("_", " ") for n in t.index]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    for ax, col, title, fmt in ((axes[0], "eta2", "Strength: share of stay / cancel explained", "{:.1%}"),
                                (axes[1], "drift", "Drift: move in stay rate, first vs second half", "{:.1%}")):
        ax.barh(names, t[col], color=common.SERIES[0] if col == "eta2" else common.NEUTRAL, height=0.62)
        for i, v in enumerate(t[col]):
            ax.text(v + t[col].max() * 0.015, i, fmt.format(v), va="center", color=common.INK_2, fontsize=9)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlim(0, t[col].max() * 1.16)
        ax.set_xticks([])
        ax.grid(False)
    fig.suptitle("One-way ANOVA: which groupings separate stayers from cancellers, and do they hold?",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    return fig


def main() -> None:
    common.style()
    d = prepare(common.load())
    rows = {}
    for f in FACTORS:
        rows[f] = anova(d, f, "stays") | {"drift": drift(d, f), "eta2_rate": anova(d[d["nightly_rate"] > 0], f, "nightly_rate")["eta2"]}
    t = pd.DataFrame(rows).T.sort_values("eta2", ascending=False)
    print("One-way ANOVA, outcome = guest stayed (1) or cancelled (0):\n")
    print(f"  {'grouping':<18}{'groups':>7}{'F':>10}{'p-value':>10}{'strength':>10}{'drift':>8}{'explains rate':>15}")
    for name, r in t.iterrows():
        print(f"  {name:<18}{int(r.groups):>7}{r.F:>10,.0f}{r.p:>10.1e}{r.eta2:>10.1%}{r.drift:>8.1%}{r.eta2_rate:>15.1%}")
    print("\n  strength = eta squared for stay/cancel; drift = move in group stay rates between halves;")
    print("  explains rate = eta squared when the outcome is the nightly rate instead.\n")
    common.save(plot(t), "anova_groupings")


if __name__ == "__main__":
    main()
