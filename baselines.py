"""Trivial reference policies for sim.py."""
import pandas as pd


class _Base:
    def __init__(self, commitments, capacity, train=None):
        self.commitments, self.capacity = commitments, capacity


class AcceptAll(_Base):
    def decide(self, b):
        return True


class RejectAll(_Base):
    def decide(self, b):
        return False


class RejectNonRefund(_Base):
    def decide(self, b):
        return b["deposit"] != "Non Refund"


if __name__ == "__main__":
    import sim

    df = sim.load()
    for cls in [AcceptAll, RejectAll, RejectNonRefund]:
        print(cls.__name__)
        for s, e in sim.SCENARIOS.values():
            sim.run(cls, df, s, e)
    # Oracle (cheating upper-ish bound): accept exactly the requests that will show up.
    print("Oracle(shows only)")
    for s, e in sim.SCENARIOS.values():
        tr, com, req, cap = sim.make_scenario(df, s, e)
        r = sim.score(pd.concat([com, req[req.cancelled == 0]]), cap, s, e)
        print(f"  {s}..{e}: net={r['net']:>12,.0f}  rev={r['revenue']:>12,.0f}  pen={r['penalty']:>10,.0f} walked={r['walked']}")
    # Greedy oracle: among requests that show, take highest nightly rate first if room on all nights
    # (commitments are forced in). A rough "how good could it get" reference.
    import numpy as np

    print("GreedyOracle(shows, by rate, capacity-feasible)")
    for s, e in sim.SCENARIOS.values():
        tr, com, req, cap = sim.make_scenario(df, s, e)
        start = pd.Timestamp(s)
        keep = []
        for h in sim.ROOMS:
            c = cap[h].to_numpy().astype(float)
            c = np.concatenate([c, np.full(100, sim.ROOMS[h])])
            for _, r in com[(com.hotel == h) & (com.cancelled == 0)].iterrows():
                a = (r.arrival_date - start).days
                c[a : a + r.nights] -= 1
            q = req[(req.hotel == h) & (req.cancelled == 0)].sort_values("nightly_rate", ascending=False)
            for idx, r in q.iterrows():
                a = (r.arrival_date - start).days
                if (c[a : a + r.nights] >= 1).all():
                    c[a : a + r.nights] -= 1
                    keep.append(idx)
        r = sim.score(pd.concat([com, req.loc[keep]]), cap, s, e)
        print(f"  {s}..{e}: net={r['net']:>12,.0f}  rev={r['revenue']:>12,.0f}  pen={r['penalty']:>10,.0f} walked={r['walked']} acc={len(keep)}")
