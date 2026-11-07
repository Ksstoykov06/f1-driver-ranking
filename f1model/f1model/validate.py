"""Does each pipeline predict what it has not seen? Held-out races + a known-answer test."""
import numpy as np, pandas as pd
from scipy.special import expit
from scipy.stats import norm, spearmanr
from .design import Design
from . import pipeline as P

def holdout_races(res, k=5):
    return res[res["round"] % k == 0].race_id.unique().tolist()

def _pairs(ds, col):
    te = ds.res[ds.res.race_id.isin(ds.holdout) & ds.res[col].notna() & (~ds.res.no_race)]
    if col == "finish":
        te = te[te.finished]
    out = []
    for (rid, team), g in te.groupby(["race_id", "team"]):
        if len(g) == 2:
            a, b = g.iloc[0], g.iloc[1]
            out.append((a, b, float(a[col] < b[col])))
    return out

def _h2h(ds, a, b, col):
    tr = ds.train[(ds.train.season == a.season) & (ds.train.team == a.team)]
    tr = tr[tr.driver.isin([a.driver, b.driver])]
    if col == "finish":
        tr = tr[tr.finished & ~tr.no_race]
    p = tr.pivot_table(index="race_id", columns="driver", values=col).dropna()
    if a.driver not in p or b.driver not in p:
        return 0.5
    w = (p[a.driver] < p[b.driver]).sum()
    return (w + 1) / (len(p) + 2)

def _score(y, p):
    y, p = np.array(y), np.clip(np.array(p), 1e-3, 1 - 1e-3)
    return dict(n=len(y), accuracy=float(((p > .5) == (y == 1)).mean()), log_loss=float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()))

def _gridn(x):
    return (np.nan_to_num(float(x), nan=20.0) - 10.5) / 9.5

def predict_pair(s, a, b, col, kind):
    """Probability the first driver of a teammate pair beats the second."""
    wet = float(a.wet_q if col == "quali_pos" else a.wet_r)
    key = "q" if col == "quali_pos" else "r"
    f = 0.3 if kind == "A" else 1.0
    def mu(x):
        age = f * (s["age1"] * x.age_c + s["age2"] * x.age_c ** 2)
        return (s[f"skill_{key}"][:, x.d] + s[f"wet_{key}"][:, x.d] * wet + s[f"dev_{key}"][:, x["dt"]] + age)
    if kind == "B":
        d = mu(a) - mu(b)
        if key == "r":
            d = d + (s["grid_era"][:, a.era] + s["grid_c"][:, a.c]) * (_gridn(a.grid) - _gridn(b.grid))
        return float(expit(d).mean())
    d = mu(a) - mu(b)
    if key == "q":
        sd = np.sqrt(2) * (s["sig_q"] + s["sig_q_wet"] * wet)
        return float(norm.cdf(d / sd).mean())
    c = (s["craft"][:, a.d] - s["craft"][:, b.d]) + (s["craft_wet"][:, a.d] - s["craft_wet"][:, b.d]) * wet
    grid = (s["grid_era"][:, a.era] + s["grid_c"][:, a.c]) * (_gridn(a.grid) - _gridn(b.grid))
    noise = np.random.default_rng(0).normal(size=len(d)) * np.sqrt(2) * (s["sig_r"] + s["sig_r_wet"] * wet)
    return float(expit(c + grid + s["b_pace"] * (d + noise)).mean())

def teammate_test(ds, samples, kinds=("A", "B")):
    rows = []
    for col, label in [("quali_pos", "qualifying"), ("finish", "race")]:
        pairs = _pairs(ds, col)
        y = [p[2] for p in pairs]
        rows.append(dict(session=label, model="coin flip", **_score(y, [.5] * len(y))))
        rows.append(dict(session=label, model="season-to-date head-to-head", **_score(y, [_h2h(ds, a, b, col) for a, b, _ in pairs])))
        if col == "finish":
            rows.append(dict(session=label, model="better grid position", **_score(y, [0.72 if (_gridn(a.grid) < _gridn(b.grid)) else 0.28 for a, b, _ in pairs])))
        for k in kinds:
            s = {x: np.asarray(v) for x, v in samples[k].items()}
            rows.append(dict(session=label, model=f"pipeline {k}", **_score(y, [predict_pair(s, a, b, col, k) for a, b, _ in pairs])))
    return pd.DataFrame(rows)

def _centre(x, groups):
    return x - pd.Series(x).groupby(groups).transform("mean").values

def _sim_order(rng, df, u):
    return (-(u + rng.gumbel(size=len(df)))).argsort().argsort() + 1

def synthetic_results(ds, rng):
    """Known-answer test for pipeline B: invent true coefficients, generate orders on the real calendar."""
    n, r = ds.n, ds.res.copy()
    tr = dict(skill_q=rng.normal(0, .6, n["d"]), skill_r=rng.normal(0, .4, n["d"]), sigma=np.exp(rng.normal(-1.3, .4, n["d"])))
    car_q, car_r = rng.normal(0, 1.5, n["t"]), rng.normal(0, 1, n["t"])
    z = rng.normal(size=n["dt"]); dq = _centre(tr["sigma"][ds.dts_driver] * z, ds.dts_driver)
    for rid, g in r[r.quali_pos.notna()].groupby("race_id"):
        u = tr["skill_q"][g.d] + car_q[g.t] + dq[g.dt]
        r.loc[g.index, "quali_pos"] = _sim_order(rng, g, u)
    for rid, g in r[r.finished].groupby("race_id"):
        gr = (np.nan_to_num(g.grid.values.astype(float), nan=20.) - 10.5) / 9.5
        u = tr["skill_r"][g.d] + car_r[g.t] + dq[g.dt] - 1.3 * gr
        r.loc[g.index, "finish"] = _sim_order(rng, g, u)
    return r, tr

def synthetic_laps(ds, qp, rp, rng):
    """Known-answer test for pipeline A: invent true % pace gaps with realistic noise."""
    n = ds.n
    tr = dict(skill_q=rng.normal(0, .15, n["d"]), skill_r=rng.normal(0, .15, n["d"]), sigma=np.exp(rng.normal(-1.3, .4, n["d"])))
    car_q, car_r = rng.normal(0, .6, n["t"]), rng.normal(0, .7, n["t"])
    dq = _centre(0.3 * tr["sigma"][ds.dts_driver] * rng.normal(size=n["dt"]), ds.dts_driver)
    def make(pace, skill, car, sig):
        m = pace[["race_id", "driver"]].merge(ds.res[["race_id", "driver", "d", "t", "dt"]], on=["race_id", "driver"])
        out = pace.copy()
        out["gap"] = -(skill[m.d] + car[m.t] + dq[m.dt]) + rng.normal(0, sig, len(m))
        return out
    return make(qp, tr["skill_q"], car_q, .15), make(rp, tr["skill_r"], car_r, .25), tr

def recovery(samples, truth, ds, eligible):
    rows = []
    for k in ["skill_q", "skill_r"]:
        x = np.asarray(samples[k]); m = x.mean(0)[eligible]; t = truth[k][eligible]
        lo, hi = np.quantile(x, .05, 0)[eligible], np.quantile(x, .95, 0)[eligible]
        # skills are only identified relative to the pool, so centre both before checking coverage
        off = (m - t).mean()
        rows.append(dict(coef=k, spearman=spearmanr(m, t)[0], coverage_90=float(((t + off >= lo) & (t + off <= hi)).mean())))
    return rows
