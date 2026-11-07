"""Customer-works car link test. Args: on|off. Same held-out races and seed; scores every pair across teams, not just teammates."""
import os, sys, json
mode = sys.argv[1]
if mode == "on": os.environ["F1_LINK"] = "1"
if mode == "qt": os.environ["F1_QT"] = "0.5"
if mode == "age": os.environ["F1_AGE"] = "1"
if mode.startswith("rho"): os.environ["F1_CAR_RHO"] = "0." + mode[3:]
if mode.startswith("form"): os.environ["F1_FORM"] = "0." + mode[4:]
if mode == "cart3": os.environ["F1_CAR_T"] = "3"
if mode == "cartest": os.environ["F1_CAR_T"] = "est"
if mode == "gap": os.environ["F1_GAP"] = "0.5"
os.environ["F1_RES"] = "results_v2_50.pkl"
sys.path.insert(0, ".")
import numpyro; numpyro.set_host_device_count(2)
import numpy as np, pandas as pd
from scipy.special import expit
import numpyro.diagnostics as nd
from f1model.config import PROCESSED, ROOT
from f1model.v2_inputs import apply_team_orders, build_wet
from f1model.v2_model import Data, model_b, inputs_b, fit
OUT = ROOT / "outputs_v2" / ("gap" if mode not in ("on", "off") else "link"); OUT.mkdir(exist_ok=True)
SUF = "_fast" if os.environ.get("LINK_FAST") else os.environ.get("LINK_SUF", "")
W, N = (20, 20) if os.environ.get("LINK_FAST") else (int(os.environ.get("LINK_W", 300)), 300)
def diag(s):
    rh = {k: float(np.nanmax(nd.gelman_rubin(v))) for k, v in s.items() if v.ndim >= 2 and v.shape[1] > 3 and not k.startswith("sig_x")}
    w = max(rh, key=rh.get); return dict(max_rhat=rh[w], worst=w)
res = apply_team_orders(pd.read_pickle(PROCESSED / "results_v2_50.pkl"), True)
if os.environ.get("TM_DROP"):   # remove teammate y from the races he shared a team with x, then refit
    x_, y_ = os.environ["TM_DROP"].split(",")
    k_ = set(zip(*res[res.driver == x_][["race_id", "team"]].values.T))
    m_ = (res.driver == y_) & pd.Series(list(zip(res.race_id, res.team)), index=res.index).isin(k_)
    print("dropped rows", int(m_.sum()), "gp", int((m_ & (res.session == "gp")).sum()), flush=True); res = res[~m_].copy()
wet = build_wet(""); full = Data(res, wet, 1950)
hold = full.res[full.res["round"] % 5 == 0].race_id.unique().tolist()
FULL = bool(os.environ.get("LINK_FULL"))
data = Data(res, wet, 1950, holdout=[] if FULL else hold)
if mode == "on": print("links", len(data.n["link_child"]), flush=True)
s = fit(model_b, inputs_b(data), W, N, tree=int(os.environ.get("LINK_T", 7)), ckpt=None)
f = {k: v.reshape((-1,) + v.shape[2:]) for k, v in s.items()}; f = {k: v[::3] for k, v in f.items()}; g_ = None
ND = len(f["r_scale"]); print("draws", ND, {k: np.round(f[k].mean(0), 3).tolist() + np.round(f[k].std(0), 3).tolist() for k in ("link_q", "link_r", "gap_b", "gap_s", "age_1", "age_2", "qt_b", "qt_s", "qt_wet", "car_q_nu", "car_r_nu", "car_q_sd", "car_r_sd", "car_q_rho", "car_r_rho") if k in f}, flush=True)
CF = "_fullcars" if FULL else "_cars"
pd.DataFrame(dict(ts=data.ts, car_q=f["car_q"].mean(0), car_r=f["car_r"].mean(0))).to_csv(OUT / f"{mode}{SUF}{CF}.csv", index=False)
import pickle; pickle.dump(dict(ts=list(data.ts), car_q=f["car_q"].astype("float32"), car_r=f["car_r"].astype("float32")), open(OUT / f"{mode}{SUF}{CF}_draws.pkl", "wb"))
d = diag(s); print(d, flush=True)
if FULL:
    from f1model import v2_score as VS
    g_, sc_ = VS.season_scores(s, data, "B", None, draws=int(os.environ.get("SC_DRAWS", 120))); VS.peak_career(g_, sc_, data).to_csv(OUT / f"{mode}{SUF}_fullscores.csv", index=False)
    g_.drop(columns=["dts", "w"]).assign(score=sc_.mean(0)).to_csv(OUT / f"{mode}{SUF}_fullseasons.csv", index=False)
    if os.environ.get("SAVE_DRAWS"): pickle.dump(dict(g=g_.drop(columns=["dts", "w"]), sc=sc_.astype("float32"), names=data.names), open(OUT / f"{mode}{SUF}_scoredraws.pkl", "wb"))
    sys.exit()
te = data.res[data.res.race_id.isin(hold) & (data.res.session == "gp")]; train_ts = data.train.groupby("ts").size()
rows = []
for col, key, lab in [("quali_pos", "q", "qualifying"), ("finish", "r", "race")]:
    t = te[te[col].notna()]
    if col == "finish": t = t[t.finished & ~t.no_race]
    kap = np.c_[f["kappa_" + key], np.ones(ND)]
    sc = 1.0 if key == "q" else f["r_scale"][:, None]
    for rid, g in t.groupby("race_id"):
        if len(g) < 2: continue
        w = g["wet_" + key].values.astype(float); pr = g["prog"].values
        m = f["skill_" + key][:, g.d] + f["car_" + key][:, g.t] + f["drift_" + key][:, g.t] * pr + (f["wet_" + key][:, g.d] + f["wetcar_" + key][:, g.t]) * w
        m = m + sc * (f["form"][:, g.dsi] + f["fit"][:, g["dt"]] + f["ad_t"][:, g.d] * g.x_t.values + f["ad_r"][:, g.d] * g.x_r.values)
        m = m * kap[:, g.decade.values[0]][:, None]
        if key == "r":
            gr = (np.nan_to_num(g.grid.values.astype(float), nan=20.) - 10.5) / 9.5
            m = m + (f["grid_era"][:, g.era.values[0]][:, None] + f["grid_c"][:, g.c])* gr + (f["chg_r"][:, g.d] + .5 * f["wetcar_r"][:, g.t]) * g.chg_r.values
        v = g[col].values; tm = g.team.values; ts = g.ts.values
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                p = expit(m[:, i] - m[:, j]).mean(); y = float(v[i] < v[j])
                cust = ("~" in tm[i]) or ("~" in tm[j])
                rows.append(dict(same=tm[i] == tm[j], session=lab, season=int(g.season.values[0]), y=y, p=p, cust=cust, sparse=min(train_ts.get(ts[i], 0), train_ts.get(ts[j], 0)) < 8))
o = pd.DataFrame(rows); o.to_pickle(OUT / f"{mode}{SUF}_pairs.pkl")
print(len(o), o.groupby("session").apply(lambda x: -(x.y * np.log(x.p.clip(.001, .999)) + (1 - x.y) * np.log(1 - x.p.clip(.001, .999))).mean()).round(4).to_dict(), flush=True)
