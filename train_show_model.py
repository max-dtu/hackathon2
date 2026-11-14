"""Show-up probability model for the hotel hackathon.

Usage:
    python train_show_model.py train.csv     # validation report: trained on earlier, tested on later periods

solution.py trains the same model itself from train.csv when it starts (about 1 second):
    model = make_model().fit(features(df), 1 - df["cancelled"])
    p = model.predict_proba(features(commitments))[:, 1]   # all commitments at once
    score = fast_scorer(model)
    p = score(booking)                                      # in decide(): ~0.01 ms per booking
"""
import math
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM = ["lead_log", "nights_c", "adults", "children", "returning_guest", "past_cancellations",
       "past_stays", "amendments", "rate_log", "special_requests", "has_company", "has_agent",
       "waitlist_days"]
CAT = ["hotel", "meal", "sales_segment", "sales_channel", "room_type", "guest_type"]
# Deliberately NOT used:
#   deposit      - will not be available in the evaluation, so the model must not rely on it;
#   booking_id, arrival_year / week / date - 2017 is unseen, and they let the model memorise
#                  2015-16 events;
#   country, agent / company IDs - higher AUC, but worse totals on later periods in our tests.


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Same function for train.csv, commitments and a single booking wrapped in a DataFrame."""
    def col(name):  # a column missing from the evaluation data counts as empty, never a crash
        return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)

    X = pd.DataFrame(index=df.index)
    X["lead_log"] = np.log1p(pd.to_numeric(col("days_ahead")).fillna(0).clip(lower=0))
    X["rate_log"] = np.log1p(pd.to_numeric(col("nightly_rate")).fillna(0).clip(0, 500))
    X["nights_c"] = pd.to_numeric(col("nights")).fillna(1).clip(upper=10)
    X["has_company"] = col("company").notna().astype(int)
    X["has_agent"] = col("agent").notna().astype(int)
    for c in ["adults", "children", "returning_guest", "past_cancellations", "past_stays",
              "amendments", "special_requests", "waitlist_days"]:
        X[c] = pd.to_numeric(col(c)).fillna(0)
    for c in CAT:
        X[c] = col(c).astype(str)
    return X[NUM + CAT]


def fast_scorer(model):
    """Turn the fitted pipeline into a plain-Python function booking_dict -> show-up probability.
    Gives the same number as model.predict_proba but about 1000x faster for one booking."""
    pre, lr = model[0], model[-1]
    names = list(pre.get_feature_names_out())
    coef = dict(zip(names, lr.coef_[0]))
    sc = pre.named_transformers_["num"]
    num = [(c, sc.mean_[i], sc.scale_[i], coef[f"num__{c}"]) for i, c in enumerate(NUM)]
    cat = {}
    for c in CAT:
        prefix = f"cat__{c}_"
        known = {n[len(prefix):]: coef[n] for n in names if n.startswith(prefix)}
        other = known.pop("infrequent_sklearn", 0.0)  # rare or never-seen categories
        cat[c] = (known, other)
    b0 = float(lr.intercept_[0])

    def val(x):
        return 0.0 if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)

    def present(x):
        return 0.0 if x is None or (isinstance(x, float) and math.isnan(x)) else 1.0

    def score(b: dict) -> float:
        x = {c: val(b.get(c)) for c in ["adults", "children", "returning_guest", "past_cancellations",
                                        "past_stays", "amendments", "special_requests", "waitlist_days"]}
        x["lead_log"] = math.log1p(max(val(b.get("days_ahead")), 0.0))
        x["rate_log"] = math.log1p(min(max(val(b.get("nightly_rate")), 0.0), 500.0))
        x["nights_c"] = min(val(b.get("nights", 1)), 10.0)
        x["has_company"] = present(b.get("company"))
        x["has_agent"] = present(b.get("agent"))
        z = b0 + sum((x[c] - mu) / sd * w for c, mu, sd, w in num)
        for c, (known, other) in cat.items():
            v = b.get(c)
            z += known.get("nan" if v is None else str(v), other)
        return 1.0 / (1.0 + math.exp(-z))

    return score


def make_model():
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        # unseen or rare categories in 2017 fall into an "infrequent" bucket instead of crashing
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), CAT),
    ])
    return make_pipeline(pre, LogisticRegression(C=0.3, max_iter=3000))


def report(name, valid, p):
    y = 1 - valid["cancelled"].values
    print(f"\n{name}: AUC {roc_auc_score(y, p):.3f} | log loss {log_loss(y, p):.3f} | "
          f"mean predicted {p.mean():.3f} vs actual {y.mean():.3f}")
    # What the policy really uses: predicted vs actual number of arrivals per hotel and day
    day = pd.DataFrame({"hotel": valid["hotel"].values, "day": valid["arrival_date"].values,
                        "pred": p, "actual": y}).groupby(["hotel", "day"]).sum()
    err = day["actual"] - day["pred"]
    print(pd.DataFrame({"avg arrivals/day": day["actual"].groupby("hotel").mean(),
                        "avg abs error": err.abs().groupby("hotel").mean(),
                        "bias (actual - predicted)": err.groupby("hotel").mean()}).round(2))
    # Calibration: within each band of predicted probability, does reality match?
    band = pd.cut(p, [0, .1, .3, .5, .7, .8, .9, 1.0], include_lowest=True)
    print(pd.DataFrame({"pred": p, "actual": y}).groupby(band, observed=True)
          .agg(predicted=("pred", "mean"), actual=("actual", "mean"), n=("actual", "size")).round(3))


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1], parse_dates=["arrival_date", "booking_date"])
    y = 1 - df["cancelled"]

    # 1) Validate by TIME, never by random split (group bookings cancel together and would leak)
    for cut, end in [("2016-01-01", "2016-07-01"), ("2016-07-01", "2017-01-01")]:
        tr = df["arrival_date"] < cut
        va = (df["arrival_date"] >= cut) & (df["arrival_date"] < end)
        m = make_model().fit(features(df[tr]), y[tr])
        report(f"train arrivals < {cut}, validate up to {end}", df[va], m.predict_proba(features(df[va]))[:, 1])

    # 2) Final model: fit on all of 2015-2016 (solution.py does exactly this when it starts)
    final = make_model().fit(features(df), y)

    # 3) The story for the judges: what pushes show-up probability up or down
    names = final[0].get_feature_names_out()
    coef = pd.Series(final[-1].coef_[0], index=names).sort_values()
    print("\nStrongest signals of cancellation:\n", coef.head(8).round(2).to_string())
    print("\nStrongest signals of showing up:\n", coef.tail(8).round(2).to_string())
