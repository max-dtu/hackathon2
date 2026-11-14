"""Classification: will this guest show up, or cancel?

Trains the five stay models from booking_model.py on arrivals before July 2016 and tests
them on arrivals from July 2016 onwards (bookings the models have never seen).

    python ML/classification.py            prints the scores, saves plots to ML/plots/
    python ML/classification.py --show     also opens each plot in a window
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score, roc_curve

import common
from booking_model import FEATURES, MODEL_KINDS, StayModel, _clean
from common import plt

LABELS = {"segments": "Segment table", "tree": "Decision tree", "logistic": "Logistic regression",
          "forest": "Random forest", "boosting": "Gradient boosting",
          "combined": "Combined (table + boosting)"}


def plot_roc(y, preds):
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0, 1], [0, 1], color=common.NEUTRAL, lw=1, ls="--", label="Guessing (AUC 0.50)")
    for color, (kind, p) in zip(common.SERIES, preds.items()):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=color, label=f"{LABELS[kind]} (AUC {roc_auc_score(y, p):.2f})")
    ax.set(xlabel="Cancellers wrongly predicted to stay (false positive rate)",
           ylabel="Stayers correctly predicted to stay (true positive rate)",
           xlim=(0, 1), ylim=(0, 1.005), title="How well each model separates stayers from cancellers")
    ax.legend(loc="lower right")
    return fig


def plot_calibration(y, preds):
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0, 1], [0, 1], color=common.NEUTRAL, lw=1, ls="--", label="Perfectly honest")
    for color, (kind, p) in zip(common.SERIES, preds.items()):
        bins = np.minimum((p * 10).astype(int), 9)          # ten equal-width probability bands
        xs = [p[bins == b].mean() for b in range(10) if (bins == b).sum() >= 50]
        ys = [y[bins == b].mean() for b in range(10) if (bins == b).sum() >= 50]
        ax.plot(xs, ys, color=color, marker="o", ms=5, label=LABELS[kind])
    ax.set(xlabel="Stay chance the model predicted", ylabel="Share of those guests who really stayed",
           xlim=(0, 1), ylim=(0, 1.005),
           title="Can the predicted chances be trusted?\nBelow the dashed line = the model is too optimistic")
    ax.legend(loc="upper left")
    return fig


def plot_importance(model, test, y):
    """How much the AUC drops when one input is scrambled: the inputs the model leans on."""
    X = model._matrix(_clean(test))
    base = roc_auc_score(y, model._proba(X))
    rng = np.random.default_rng(0)
    drops = []
    for j in range(X.shape[1]):
        Xp = X.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        drops.append(base - roc_auc_score(y, model._proba(Xp)))
    order = np.argsort(drops)[-12:]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([FEATURES[j].replace("_", " ") for j in order], [drops[j] for j in order],
            color=common.SERIES[0], height=0.62)
    for i, j in enumerate(order):
        ax.text(drops[j] + max(drops) * 0.012, i, f"{drops[j]:.3f}", va="center", color=common.INK_2, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Loss of accuracy (AUC) when this input is scrambled", xlim=(0, max(drops) * 1.12),
           title=f"What the {LABELS[model.kind].lower()} model relies on most")
    return fig


def main() -> None:
    common.style()
    train, test = common.time_split(common.load())
    y = 1 - test["cancelled"].to_numpy()                      # 1 = the guest stayed
    print(f"Train: {len(train):,} bookings arriving before {common.SPLIT}   "
          f"Test: {len(test):,} arriving after   ({y.mean():.1%} of test guests stayed)\n")
    print(f"  {'model':<29}{'AUC':>7}{'accuracy':>10}{'log loss':>10}{'Brier':>8}{'avg predicted stay':>20}")
    print(f"  {'always say stays':<29}{0.5:>7.3f}{max(y.mean(), 1 - y.mean()):>10.1%}")
    preds, models = {}, {}
    for kind in MODEL_KINDS:
        models[kind] = StayModel(kind).fit(train)
        p = preds[kind] = models[kind].predict(test)
        print(f"  {LABELS[kind]:<29}{roc_auc_score(y, p):>7.3f}{accuracy_score(y, p >= 0.5):>10.1%}"
              f"{log_loss(y, p):>10.3f}{brier_score_loss(y, p):>8.3f}{p.mean():>20.1%}", flush=True)
    print("\n  AUC: 1.0 = perfect ranking, 0.5 = guessing. Log loss and Brier: lower is better.\n")

    best = max(preds, key=lambda k: roc_auc_score(y, preds[k]))
    common.save(plot_roc(y, preds), "classification_roc")
    common.save(plot_calibration(y, preds), "classification_calibration")
    common.save(plot_importance(models[best], test, y), "classification_importance")


if __name__ == "__main__":
    main()
