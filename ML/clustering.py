"""Clustering: which kinds of guests are there, and which kinds stay?

K-means groups bookings that look alike (lead time, length of stay, rate, party size,
deposit, sales segment...). Whether the guest cancelled is NOT shown to the algorithm; it is
only used afterwards, to see how reliable each group turned out to be.

    python ML/clustering.py            prints the groups, saves plots to ML/plots/
    python ML/clustering.py --show     also opens each plot in a window
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import common
from common import plt

K_RANGE = range(3, 9)
PROFILE = {"lead_days": "Days booked ahead", "nights": "Nights", "nightly_rate": "Nightly rate",
           "guests": "Guests", "special_requests": "Special requests", "amendments": "Amendments",
           "returning_guest": "Returning guest", "weekend_share": "Weekend share"}


def features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (readable columns for profiling, numeric matrix for k-means)."""
    d = pd.DataFrame({
        "lead_days": df["days_ahead"], "nights": df["nights"], "nightly_rate": df["nightly_rate"],
        "guests": df["adults"] + df["children"].fillna(0) + df["babies"],
        "special_requests": df["special_requests"], "amendments": df["amendments"],
        "returning_guest": df["returning_guest"],
        "weekend_share": df["weekend_nights"] / df["nights"].clip(lower=1),
    }, index=df.index)
    X = d.copy()
    X["lead_days"] = np.log1p(X["lead_days"])                 # 1 vs 10 days matters more than 300 vs 310
    X["nights"] = X["nights"].clip(upper=14)
    X["nightly_rate"] = X["nightly_rate"].clip(0, 400)
    X["amendments"] = X["amendments"].clip(upper=5)
    segment = df["sales_segment"].where(df["sales_segment"].isin(
        ["Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate"]), "Other")
    X = pd.concat([X, pd.get_dummies(segment, prefix="seg", dtype=float),
                   pd.get_dummies(df["deposit"], prefix="dep", dtype=float),
                   pd.get_dummies(df["guest_type"], prefix="type", dtype=float),
                   pd.get_dummies(df["hotel"], prefix="hotel", dtype=float)], axis=1)
    return d, X


def describe(df: pd.DataFrame, d: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """One row per group, most reliable first, lettered A, B, C..."""
    g = df.assign(cluster=labels, stays=1 - df["cancelled"], lead_days=d["lead_days"]).groupby("cluster")
    t = g.agg(bookings=("stays", "size"), stay_rate=("stays", "mean"), nightly_rate=("nightly_rate", "mean"),
              lead_days=("lead_days", "median"), nights=("nights", "mean"),
              segment=("sales_segment", lambda s: s.mode()[0]),
              deposit=("deposit", lambda s: s.mode()[0]),
              hotel=("hotel", lambda s: s.mode()[0]))
    t = t.sort_values("stay_rate", ascending=False)
    t["letter"] = [chr(65 + i) for i in range(len(t))]
    t["name"] = [f"{r.segment}, {r.lead_days:.0f} days ahead"
                 + (f", {r.deposit.lower()}" if r.deposit != "No Deposit" else "") for r in t.itertuples()]
    return t


def plot_choose_k(scores: dict, k: int):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ks = list(scores)
    ax.plot(ks, [scores[i] for i in ks], color=common.SERIES[0], marker="o", ms=6)
    ax.scatter([k], [scores[k]], s=160, facecolors="none", edgecolors=common.INK, linewidths=1.5, zorder=3)
    ax.annotate(f"chosen: {k} groups", (k, scores[k]), xytext=(0, 14), textcoords="offset points",
                ha="center", color=common.INK, fontsize=9)
    ax.set(xlabel="Number of groups", ylabel="Silhouette score", xticks=ks,
           ylim=(min(scores.values()) * 0.9, max(scores.values()) * 1.12),
           title="How many groups? Higher = groups are more clearly apart")
    ax.grid(axis="x", visible=False)
    return fig


def plot_segments(t: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5.6))
    size = (t["bookings"] / t["bookings"].max() * 2600).clip(lower=260)   # keep tiny groups readable
    ax.scatter(t["nightly_rate"], t["stay_rate"] * 100, s=size,
               color=common.SERIES[0], alpha=0.85, edgecolors=common.SURFACE, linewidths=2, zorder=2)
    for r in t.itertuples():
        ax.text(r.nightly_rate, r.stay_rate * 100, r.letter, ha="center", va="center", color="white",
                fontweight="bold", fontsize=10, zorder=3)
    legend = "\n".join(f"{r.letter}   {r.name}  ({r.bookings:,} bookings)" for r in t.itertuples())
    ax.text(1.02, 1.0, "Groups, most reliable first\n\n" + legend, transform=ax.transAxes, va="top",
            color=common.INK_2, fontsize=9, linespacing=1.6)
    pad = (t["nightly_rate"].max() - t["nightly_rate"].min()) * 0.18
    ax.set(xlabel="Average nightly rate", ylabel="Guests who showed up (%)", ylim=(-6, 108),
           xlim=(t["nightly_rate"].min() - pad, t["nightly_rate"].max() + pad),
           title="Which guest groups stay, and what they pay\nBigger circle = more bookings")
    return fig


def plot_profile(d: pd.DataFrame, labels: np.ndarray, t: pd.DataFrame):
    means = d.groupby(labels).mean().loc[t.index, list(PROFILE)]
    z = (means - d[list(PROFILE)].mean()) / d[list(PROFILE)].std()
    fig, ax = plt.subplots(figsize=(10, 0.62 * len(t) + 2.2))
    lim = max(float(np.abs(z.to_numpy()).max()), 1.0)
    ax.imshow(z.to_numpy(), cmap=LinearSegmentedColormap.from_list("div", common.DIVERGING),
              vmin=-lim, vmax=lim, aspect="auto")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            v = means.iloc[i, j]
            ax.text(j, i, f"{v:.0f}" if abs(v) >= 10 else f"{v:.1f}", ha="center", va="center",
                    color="white" if abs(z.iloc[i, j]) > lim * 0.6 else common.INK, fontsize=9)
    ax.set_xticks(range(len(PROFILE)), list(PROFILE.values()), fontsize=9)
    ax.set_yticks(range(len(t)), [f"{r.letter}  ({r.stay_rate:.0%} stay)" for r in t.itertuples()])
    ax.xaxis.tick_top()
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, z.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, z.shape[0]), minor=True)
    ax.grid(which="minor", color=common.SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    fig.suptitle("What makes each group different", x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.text(0.01, 0.02, "Numbers are each group's average.  Blue = below the overall average,  "
             "red = above,  grey = typical.", color=common.INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return fig


def plot_map(Xs: np.ndarray, labels: np.ndarray, t: pd.DataFrame):
    """Every booking squeezed onto two axes (PCA). One panel per group, the rest in grey."""
    idx = np.random.default_rng(0).choice(len(Xs), min(5000, len(Xs)), replace=False)
    xy = PCA(n_components=2, random_state=0).fit_transform(Xs)[idx]
    lab = labels[idx]
    cols = min(len(t), 4)
    rows = int(np.ceil(len(t) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.1 * rows), sharex=True, sharey=True, squeeze=False)
    for ax, r in zip(axes.flat, t.itertuples()):
        ax.scatter(xy[:, 0], xy[:, 1], s=4, color=common.NEUTRAL, alpha=0.35, linewidths=0)
        m = lab == r.Index
        ax.scatter(xy[m, 0], xy[m, 1], s=5, color=common.SERIES[0], alpha=0.7, linewidths=0)
        ax.set_title(f"{r.letter}  {r.stay_rate:.0%} stay", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    for ax in axes.flat[len(t):]:
        ax.set_visible(False)
    fig.suptitle("Where each group sits among all bookings (similar bookings are close together)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    return fig


def main() -> None:
    common.style()
    df = common.load()
    d, X = features(df)
    Xs = StandardScaler().fit_transform(X)

    sample = np.random.default_rng(0).choice(len(Xs), 8000, replace=False)
    scores = {}
    print("Choosing the number of groups (silhouette score, higher is better):")
    for k in K_RANGE:
        lab = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(Xs)
        scores[k] = silhouette_score(Xs[sample], lab[sample])
        print(f"  {k} groups: {scores[k]:.3f}", flush=True)
    k = max(scores, key=scores.get)
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xs)
    t = describe(df, d, labels)

    print(f"\n{k} groups, most reliable first:")
    print(f"  {'':<3}{'group':<44}{'bookings':>9}{'stay rate':>11}{'avg rate':>10}{'nights':>8}  hotel")
    for r in t.itertuples():
        print(f"  {r.letter:<3}{r.name:<44}{r.bookings:>9,}{r.stay_rate:>11.0%}{r.nightly_rate:>10.0f}"
              f"{r.nights:>8.1f}  {r.hotel}")
    print()
    common.save(plot_choose_k(scores, k), "clustering_choose_k")
    common.save(plot_segments(t), "clustering_segments")
    common.save(plot_profile(d, labels, t), "clustering_profile")
    common.save(plot_map(Xs, labels, t), "clustering_map")


if __name__ == "__main__":
    main()
