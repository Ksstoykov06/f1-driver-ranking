"""Turns tables into the integer indices and padded ranking arrays the models need."""
import numpy as np, pandas as pd
from .config import *

NO_RACE = {("2021", "belgium")}   # Spa 2021: no racing laps behind anything but the safety car

def add_reasons(res, collisions_as_errors=False):
    res = res.copy()
    r = res.reason.fillna("")
    err = set(SOLO_ERROR) | (COLLISION if collisions_as_errors else set())
    res["started"] = ~res.pos_text.isin(["DNS", "DNQ", "DNP", "EX"])
    res["error"] = res.started & r.isin(err)
    res["finished"] = res.finish.notna() & (r == "")
    res["no_race"] = [(str(s), g) in NO_RACE for s, g in zip(res.season, res.gp)]
    return res

def wet_labels(quali, race):
    """Lap-derived labels. Pipeline B borrows these in the pilot; pre-2018 needs a hand-built list."""
    wr = race.groupby("race_id").wet.first().rename("wet_r")
    wq = quali.groupby("race_id").wet.first().rename("wet_q")
    return pd.concat([wq, wr], axis=1).fillna(False).astype(bool)

class Design:
    def __init__(self, res, wet, holdout=()):
        self.res = res = res.copy()
        res = res.merge(wet, left_on="race_id", right_index=True, how="left")
        res[["wet_q", "wet_r"]] = res[["wet_q", "wet_r"]].fillna(False).astype(bool)
        res["ts"] = res.team + "|" + res.season.astype(str)
        res["dts"] = res.driver + "|" + res.ts
        res["era"] = np.select([res.season < 2022, res.season < 2026], [0, 1], 2)
        res["prog"] = res["round"] / res.groupby("season")["round"].transform("max") - 0.5
        res["age_c"] = (res.age - 28) / 10
        self.drivers = sorted(res.driver.unique())
        self.ts = sorted(res.ts.unique()); self.dts = sorted(res.dts.unique())
        self.circuits = sorted(res.circuit.unique())
        ix = lambda col, keys: res[col].map({k: i for i, k in enumerate(keys)}).values
        res["d"], res["t"], res["dt"], res["c"] = ix("driver", self.drivers), ix("ts", self.ts), ix("dts", self.dts), ix("circuit", self.circuits)
        self.dts_driver = np.array([self.drivers.index(k.split("|")[0]) for k in self.dts])
        self.res = res
        self.holdout = set(holdout)
        self.train = res[~res.race_id.isin(self.holdout)]
        self.names = res.drop_duplicates("driver").set_index("driver").name.to_dict()
        self.n = dict(d=len(self.drivers), t=len(self.ts), dt=len(self.dts), c=len(self.circuits))
        self.wet_pairs = res.drop_duplicates("race_id")[["wet_q", "wet_r"]].values.astype(float)

def rank_arrays(df, order_col):
    """One row per race; drivers sorted best→worst, right-padded. Returns row indices into df."""
    groups = [g.sort_values(order_col).index.values for _, g in df.groupby("race_id")]
    groups = [g for g in groups if len(g) >= 2]
    width = max(map(len, groups))
    idx = np.zeros((len(groups), width), int); mask = np.zeros((len(groups), width), bool)
    for i, g in enumerate(groups):
        idx[i, : len(g)] = g; mask[i, : len(g)] = True
    return idx, mask

def grid_norm(g):
    return (np.nan_to_num(g.astype(float), nan=20.0) - 10.5) / 9.5
