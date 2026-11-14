"""Regression: what nightly rate should a booking fetch?

Predicts the nightly rate from the hotel, season, lead time, room type and party size.
This is the "what is this room worth" side of the problem: a request far below the going
rate for its night is a room sold too cheaply.

    python ML/regression.py            prints the scores, saves plots to ML/plots/
    python ML/regression.py --show     also opens each plot in a window
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

import common
from common import plt

CAT = ["hotel", "month", "room_type", "meal", "sales_segment", "sales_channel", "guest_type", "deposit"]
NUM = ["days_ahead", "nights", "weekend_nights", "adults", "children", "babies",
       "special_requests", "returning_guest", "day_of_year", "day_of_week"]
TARGET = "nightly_rate"


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df[TARGET] > 0) & (df[TARGET] < 1000)].copy()   # drop free stays and typos
    df["month"] = df["arrival_date"].dt.month_name()
    df["day_of_year"] = df["arrival_date"].dt.dayofyear
    df["day_of_week"] = df["arrival_date"].dt.dayofweek
    return df


def models() -> dict:
    onehot = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT),
                                ("num", make_pipeline(SimpleImputer(), StandardScaler()), NUM)])

    def ordinal():
        return ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT),
            ("num", SimpleImputer(), NUM)])

    return {
        "Linear (ridge)": make_pipeline(onehot, Ridge(alpha=1.0)),
        "Decision tree": make_pipeline(ordinal(), DecisionTreeRegressor(max_leaf_nodes=64, min_samples_leaf=50, random_state=0)),
        "Random forest": make_pipeline(ordinal(), RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=0)),
        "Gradient boosting": make_pipeline(ordinal(), HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=0)),
    }


def plot_models(scores: dict):
    names = list(scores)[::-1]
    mae = [scores[n]["mae"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(names, mae, height=0.6,
            color=[common.NEUTRAL if n.startswith("Baseline") else common.SERIES[0] for n in names])
    for i, v in enumerate(mae):
        ax.text(v + max(mae) * 0.012, i, f"{v:.1f}", va="center", color=common.INK_2, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Average error per night (lower is better)", xlim=(0, max(mae) * 1.1),
           title="How far off each model's predicted nightly rate is")
    return fig


def plot_scatter(y, pred, name):
    idx = np.random.default_rng(0).choice(len(y), min(6000, len(y)), replace=False)
    hi = float(np.quantile(y, 0.995))
    fig, ax = plt.subplots(figsize=(6, 5.8))
    ax.scatter(y[idx], pred[idx], s=9, color=common.SERIES[0], alpha=0.22, linewidths=0)
    ax.plot([0, hi], [0, hi], color=common.INK_2, lw=1, ls="--")
    ax.text(hi * 0.97, hi * 0.9, "perfect prediction", color=common.INK_2, fontsize=9, ha="right")
    ax.set(xlim=(0, hi), ylim=(0, hi), xlabel="Actual nightly rate", ylabel="Predicted nightly rate",
           title=f"Predicted vs actual rate, {name.lower()}\nEach dot is one unseen booking")
    return fig


def plot_months(test, pred):
    d = test.assign(pred=pred, m=test["arrival_date"].dt.to_period("M").dt.to_timestamp())
    hotels = sorted(d["hotel"].unique())
    fig, axes = plt.subplots(1, len(hotels), figsize=(10, 4), sharey=True)
    for ax, hotel in zip(axes, hotels):
        g = d[d["hotel"] == hotel].groupby("m")[[TARGET, "pred"]].mean()
        ax.plot(g.index, g[TARGET], color=common.SERIES[0], marker="o", ms=5, label="Actual")
        ax.plot(g.index, g["pred"], color=common.SERIES[1], marker="o", ms=5, label="Predicted")
        ax.set_title(hotel, fontsize=11)
        ax.set_xticks(g.index, [m.strftime("%b") for m in g.index])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Average nightly rate")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc="upper right")
    fig.suptitle("Does the model follow the season? Average rate by arrival month, 2016",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    return fig


def plot_importance(model, X, y, name):
    idx = np.random.default_rng(0).choice(len(X), min(5000, len(X)), replace=False)
    r = permutation_importance(model, X.iloc[idx], y[idx], n_repeats=3, random_state=0,
                               scoring="neg_mean_absolute_error")
    order = np.argsort(r.importances_mean)[-10:]
    vals = r.importances_mean[order]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh([X.columns[j].replace("_", " ") for j in order], vals, color=common.SERIES[0], height=0.62)
    for i, v in enumerate(vals):
        ax.text(v + vals.max() * 0.012, i, f"{v:.1f}", va="center", color=common.INK_2, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Extra error per night when this input is scrambled", xlim=(0, vals.max() * 1.1),
           title=f"What drives the nightly rate ({name.lower()})")
    return fig


def main() -> None:
    common.style()
    train, test = common.time_split(prepare(common.load()))
    Xtr, ytr = train[CAT + NUM], train[TARGET].to_numpy()
    Xte, yte = test[CAT + NUM], test[TARGET].to_numpy()
    print(f"Train: {len(train):,} bookings arriving before {common.SPLIT}   Test: {len(test):,} after   "
          f"(average rate {yte.mean():.0f})\n")
    print(f"  {'model':<34}{'MAE':>8}{'RMSE':>8}{'R2':>8}")

    baseline = test["hotel"].map(train.groupby("hotel")[TARGET].mean()).to_numpy()
    preds = {"Baseline: hotel's average rate": baseline}
    fitted = {}
    for name, model in models().items():
        fitted[name] = model.fit(Xtr, ytr)
        preds[name] = model.predict(Xte)
    scores = {}
    for name, p in preds.items():
        scores[name] = {"mae": mean_absolute_error(yte, p)}
        print(f"  {name:<34}{scores[name]['mae']:>8.1f}{root_mean_squared_error(yte, p):>8.1f}"
              f"{r2_score(yte, p):>8.3f}")
    print("\n  MAE = average error per night. R2: 1.0 = perfect, 0 = no better than the average.\n")

    best = min(fitted, key=lambda n: scores[n]["mae"])
    common.save(plot_models(scores), "regression_models")
    common.save(plot_scatter(yte, preds[best], best), "regression_predicted_vs_actual")
    common.save(plot_months(test, preds[best]), "regression_by_month")
    common.save(plot_importance(fitted[best], Xte, yte, best), "regression_importance")


if __name__ == "__main__":
    main()
