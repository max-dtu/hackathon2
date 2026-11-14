"""Show-up probability model: P(not cancelled | booking features).

Supports an optional blend of a full-feature HistGradientBoostingClassifier and a shift-resistant
no-country HistGradientBoostingClassifier, plus optional exponential recency sample weighting.
Single-row prediction reuses the fitted ColumnTransformer category mapping and invokes
_predict_iterations directly so online decide() calls match batch predict() bit-for-bit at ~1-2 ms/row.
"""
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

CAT = ["hotel", "meal", "country", "sales_segment", "sales_channel", "room_type", "deposit", "guest_type", "agent", "company"]
NUM = [
    "days_ahead", "weekend_nights", "weekday_nights", "nights", "adults", "children", "babies", "returning_guest",
    "past_cancellations", "past_stays", "amendments", "waitlist_days", "nightly_rate", "special_requests", "value",
]


def _num(v):
    try:
        f = float(v)
        return f if not math.isnan(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _key(v):
    """Canonical string for a categorical value (handles NaN / float ids like 240.0)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "nan"
    return str(v)


class _HGBMember:
    """Single HistGradientBoostingClassifier with manual category encoding and fast single-row inference."""

    def __init__(self, num_cols, cat_cols, min_count=20, **hgb_params):
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.min_count = min_count
        self.hgb_params = dict(hgb_params)

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        n_num, n_cat = len(self.num_cols), len(self.cat_cols)
        X = np.empty((len(df), n_num + n_cat), dtype=float)
        for j, c in enumerate(self.num_cols):
            if c == "value":
                X[:, j] = pd.to_numeric(df["nightly_rate"], errors="coerce").to_numpy() * pd.to_numeric(df["nights"], errors="coerce").to_numpy()
            else:
                X[:, j] = pd.to_numeric(df[c], errors="coerce").to_numpy()
        for k, c in enumerate(self.cat_cols):
            m = self.codes[c]
            other = len(m)
            X[:, n_num + k] = [m.get(_key(v), other) for v in df[c].tolist()]
        return X

    def fit(self, train: pd.DataFrame, y: np.ndarray, sample_weight=None):
        self.codes = {}
        for c in self.cat_cols:
            vc = pd.Series([_key(v) for v in train[c].tolist()]).value_counts()
            keep = sorted(vc[vc >= self.min_count].index.tolist())[:250]
            self.codes[c] = {v: i for i, v in enumerate(keep)}
        mask = [False] * len(self.num_cols) + [True] * len(self.cat_cols)
        self.clf = HistGradientBoostingClassifier(
            categorical_features=mask if self.cat_cols else None,
            **self.hgb_params,
        )
        self.clf.fit(self._matrix(train), y, sample_weight=sample_weight)
        self._init_fast_predict()
        return self

    def _init_fast_predict(self):
        n_cat, n_num = len(self.cat_cols), len(self.num_cols)
        self._row_buf = np.empty((1, n_cat + n_num), dtype=float)
        self._raw_buf = np.zeros((1, 1), dtype=self.clf._baseline_prediction.dtype, order="F")
        self._cat_maps = []
        if self.clf._preprocessor is not None:
            enc = self.clf._preprocessor.named_transformers_["encoder"]
            for c, cats in zip(self.cat_cols, enc.categories_):
                orig_map = self.codes[c]
                ord_lookup = {float(v): float(idx) for idx, v in enumerate(cats)}
                other_ord = ord_lookup.get(float(len(orig_map)), np.nan)
                new_map = {k: ord_lookup.get(float(v), other_ord) for k, v in orig_map.items()}
                self._cat_maps.append((new_map, other_ord))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(self._matrix(df))[:, 1]

    def predict_one(self, b: dict) -> float:
        n_cat = len(self._cat_maps)
        x = self._row_buf
        for k, c in enumerate(self.cat_cols):
            if n_cat:
                mp, oth = self._cat_maps[k]
                x[0, k] = mp.get(_key(b.get(c)), oth)
            else:
                mp = self.codes[c]
                x[0, len(self.num_cols) + k] = mp.get(_key(b.get(c)), len(mp))
        num_off = n_cat if n_cat else 0
        for j, c in enumerate(self.num_cols):
            x[0, num_off + j] = _num(b.get("nightly_rate")) * _num(b.get("nights")) if c == "value" else _num(b.get(c))
        raw = self._raw_buf
        raw[0, 0] = self.clf._baseline_prediction[0, 0]
        self.clf._predict_iterations(x, self.clf._predictors, raw, False, 1)
        return 1.0 / (1.0 + math.exp(-float(raw[0, 0])))


class ShowModel:
    """Show-up probability classifier blending full-feature and shift-resistant models."""

    def __init__(self, min_count=20, blend_nocountry=0.35, recency_decay=0.0, num_cols=None, cat_cols=None, **kw):
        self.min_count = min_count
        self.blend_nocountry = float(blend_nocountry)
        self.recency_decay = float(recency_decay)
        self.num_cols = list(NUM if num_cols is None else num_cols)
        self.cat_cols = list(CAT if cat_cols is None else cat_cols)
        params1 = dict(max_iter=200, learning_rate=0.07, max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.0,
                       random_state=0)
        params1.update(kw)
        self.params1 = params1
        self.params2 = dict(max_iter=150, learning_rate=0.07, max_leaf_nodes=31, min_samples_leaf=80, l2_regularization=2.0,
                            random_state=1)

    def fit(self, train: pd.DataFrame):
        y = 1 - train["cancelled"].to_numpy()
        sw = None
        if self.recency_decay > 0:
            arr = pd.to_datetime(train["arrival_date"])
            days_ago = (arr.max() - arr).dt.days.to_numpy(dtype=float)
            sw = np.exp(-self.recency_decay * days_ago)
        self.m1 = _HGBMember(self.num_cols, self.cat_cols, self.min_count, **self.params1).fit(train, y, sample_weight=sw)
        self.m2 = None
        if self.blend_nocountry > 0 and "country" in self.cat_cols:
            nocnt_cats = [c for c in self.cat_cols if c != "country"]
            self.m2 = _HGBMember(self.num_cols, nocnt_cats, self.min_count, **self.params2).fit(train, y, sample_weight=sw)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        p1 = self.m1.predict(df)
        if self.m2 is None:
            return p1
        w = self.blend_nocountry
        return (1.0 - w) * p1 + w * self.m2.predict(df)

    def predict_one(self, b: dict) -> float:
        p1 = self.m1.predict_one(b)
        if self.m2 is None:
            return p1
        w = self.blend_nocountry
        return (1.0 - w) * p1 + w * self.m2.predict_one(b)


if __name__ == "__main__":
    import time
    import sim
    from sklearn.metrics import log_loss, roc_auc_score

    df = sim.load()
    for name, (s, e) in sim.SCENARIOS.items():
        tr, com, req, cap = sim.make_scenario(df, s, e)
        m = ShowModel().fit(tr)
        for part_name, part in [("commitments", com), ("requests", req)]:
            p = np.clip(m.predict(part), 1e-4, 1 - 1e-4)
            y = 1 - part.cancelled.to_numpy()
            print(f"{name:10s} {part_name:12s} n={len(part):6d} show={y.mean():.3f} pred={p.mean():.3f} "
                  f"AUC={roc_auc_score(y, p):.3f} LL={log_loss(y, p):.3f}")
        sub = req.head(500)
        pb = m.predict(sub)
        rows = sub.drop(columns=["cancelled"]).to_dict("records")
        t0 = time.time()
        po = np.array([m.predict_one(r) for r in rows])
        ms = (time.time() - t0) / len(rows) * 1000
        print(f"  fast predict_one: {ms:.2f} ms/row, maxdiff={np.abs(pb - po).max():.2e}")
