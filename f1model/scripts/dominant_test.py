"""How much of a dominant car's advantage gets credited to its driver?
Truth = C50 posterior means. A sim with and without a planted car advantage, same random numbers.
Any change in a driver's score between the two is car credit landing on the driver.
   python scripts/dominant_test.py SEED BUMP"""
import os, sys, pickle, json
os.environ["F1_RES"] = "results_v2_50.pkl"
sys.path.insert(0, ".")
import numpyro; numpyro.set_host_device_count(2)
import numpy as np, pandas as pd
from f1model.config import PROCESSED, ROOT
from f1model.v2_inputs import apply_team_orders, build_wet
from f1model.v2_model import *
from f1model import v2_score as VS
OUT = ROOT / "outputs_v2"; seed, bump = int(sys.argv[1]), float(sys.argv[2])
PLANT = {"mercedes": [1954, 1955], "lotus": [1962, 1963, 1964, 1965], "maserati": [1957], "ferrari": [2001, 2002, 2004], "mclaren": [1984, 1988, 1989], "williams": [1992, 1993, 1996], "red-bull": [2011, 2023], "mercedes-2": [2015, 2016, 2020]}
res = apply_team_orders(pd.read_pickle(PROCESSED / "results_v2_50.pkl"), True); wet = build_wet("")
data = Data(res, wet, 1950); n = data.n
st = pickle.load(open(OUT / "ckpt/C50.pkl", "rb")); M = {k: np.concatenate([c[k] for c in st["chunks"]], axis=1).mean((0, 1)) for k in st["chunks"][0]}
ix = {k: i for i, k in enumerate(data.ts)}; pl = []
for t, ys in PLANT.items():
    for y in ys:
        k = f"{t.split('-2')[0]}|{y}"
        if k in ix: pl.append(ix[k])
        else: print("not found", k)
pl = np.array(pl); cq, cr = M["car_q"].copy(), M["car_r"].copy(); cq[pl] += bump; cr[pl] += bump
sc = float(M["r_scale"]); r = data.res.copy(); rng = np.random.default_rng(seed)
def fun(g, key):
    w = g.wet_q.values if key == "q" else g.wet_r.values; s_ = 1.0 if key == "q" else sc
    v = M[f"skill_{key}"][g.d] + M[f"wet_{key}"][g.d] * w + (cq if key == "q" else cr)[g.t] + s_ * (M["form"][g.dsi] + M["fit"][g["dt"]] + M["ad_t"][g.d] * g.x_t.values + M["ad_r"][g.d] * g.x_r.values)
    return v
gridpos = {}
for rid, g in r[r.quali_pos.notna() & (r.session == "gp")].sort_values("race_id").groupby("race_id"):
    o = (-(fun(g, "q") + rng.gumbel(size=len(g)))).argsort().argsort() + 1; r.loc[g.index, "quali_pos"] = o; r.loc[g.index, "grid"] = o
for key, g in r[r.finished].sort_values("race_id").groupby(["race_id", "session"]):
    gr = (np.nan_to_num(g.grid.values.astype(float), nan=20.) - 10.5) / 9.5
    r.loc[g.index, "finish"] = (-(fun(g, "r") + M["chg_r"][g.d] * g.chg_r.values - 1.3 * gr + rng.gumbel(size=len(g)))).argsort().argsort() + 1
d2 = Data(r.drop(columns=["wet_q", "wet_r", "chg_r", "ts", "dteam", "ds", "d", "t", "dt", "c", "dsi", "era", "decade", "prog", "x_t", "x_r", "sprint"]), wet, 1950)
s = fit(model_b, inputs_b(d2), 500, 400, tree=7)
g, sco = VS.season_scores(s, d2, "B", None, draws=100, weekends=20); pc = VS.peak_career(g, sco, d2)
tag = f"s{seed}_b{bump:g}"; D = OUT / "dominant"; D.mkdir(exist_ok=True); pc.to_csv(D / f"{tag}_scores.csv", index=False)
ix2 = {k: i for i, k in enumerate(d2.ts)}; car = (s["car_q"].mean((0, 1)) + s["car_r"].mean((0, 1))) / 2
pd.DataFrame(dict(ts=d2.ts, car=car, planted=[k in {data.ts[i] for i in pl} for k in d2.ts])).to_csv(D / f"{tag}_cars.csv", index=False)
c = pd.DataFrame(dict(driver=d2.drivers, skill_q=s["skill_q"].mean((0, 1)), skill_r=s["skill_r"].mean((0, 1)))); c.to_csv(D / f"{tag}_skill.csv", index=False); print("done", tag, flush=True)
