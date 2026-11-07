import os
#!/usr/bin/env python
"""Version 2 runner. Each stage saves its results and is skipped if they already exist, so it can be restarted.
   python run_v2.py A18|B18|B80 [--fast]        fit and score one model
   python run_v2.py holdout A18|B18|B80         teammate predictions in unseen races
   python run_v2.py synth A18|B18|B80           known-answer test
   python run_v2.py sens                        change assumptions
"""
import sys, json, warnings, pickle
import numpyro; numpyro.set_host_device_count(2)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, numpyro.diagnostics as nd
from scipy.special import expit
from scipy.stats import norm, spearmanr
from f1model.config import PROCESSED, ROOT
from f1model.v2_inputs import apply_team_orders, build_wet, build_pace
from f1model.v2_model import *
from f1model import v2_score as VS

if any(a.startswith("C50") for a in sys.argv[1:]): os.environ["F1_RES"] = "results_v2_50.pkl"
OUT = ROOT / "outputs_v2"; OUT.mkdir(exist_ok=True); CK = OUT / "ckpt"; CK.mkdir(exist_ok=True)
args = [a for a in sys.argv[1:] if not a.startswith("--")]; FAST = "--fast" in sys.argv
SET = {"A18": dict(kind="A", start=2018, warm=5000, samp=20000, tree=8, seg=20), "B18": dict(kind="B", start=2018, warm=1500, samp=1500, tree=9, seg=3), "B80": dict(kind="B", start=1966, warm=1000, samp=1000, tree=9, seg=5), "B80s12": dict(kind="B", start=1966, warm=1000, samp=1000, tree=9, seg=5), "B80s13": dict(kind="B", start=1966, warm=1000, samp=1000, tree=9, seg=5), "C66": dict(kind="B", start=1966, warm=600, samp=600, tree=8, seg=2), "C61": dict(kind="B", start=1961, warm=600, samp=600, tree=8, seg=2), "B80m": dict(kind="B", start=1966, warm=1000, samp=1000, tree=9, seg=5), "C50": dict(kind="B", start=1950, warm=1000, samp=1000, tree=9, seg=5), "B80ctl": dict(kind="B", start=1980, warm=500, samp=500, tree=9, seg=2)}

def load(orders=True, tag=""):
    res = apply_team_orders(pd.read_pickle(PROCESSED / os.environ.get("F1_RES", "results_v2.pkl")), orders)
    return res, build_wet(tag), pd.read_pickle(PROCESSED / f"pace_v2{tag}.pkl")

def inputs(cfg, data, pace):
    return (model_a, inputs_a(data, pace)) if cfg["kind"] == "A" else (model_b, inputs_b(data))

def spread_of(pace, data):
    d = pace[(pace.kind == "q") & (pace.cond == "dry") & pace.lap_sd.notna() & (pace.n >= 3)]
    g = d.groupby("driver").lap_sd.agg(["median", "size"]); o = d.lap_sd.median()
    sh = (g["median"] * g["size"] + o * 10) / (g["size"] + 10)
    return np.array([sh.get(x, o) for x in data.drivers])

def diag(s):
    rh, es = {}, {}
    for k, v in s.items():
        if v.ndim >= 2 and v.shape[1] > 3 and not k.startswith("sig_x"):
            rh[k] = float(np.nanmax(nd.gelman_rubin(v))); es[k] = float(np.nanmin(nd.effective_sample_size(v)))
    w = max(rh, key=rh.get); e = min(es, key=es.get)
    return dict(max_rhat=rh[w], worst=w, min_ess=es[e], min_ess_param=e, key={k: round(rh[k], 3) for k in ["skill_q", "skill_r", "form", "ad_t", "ad_r", "wet_r", "car_r"] if k in rh})

def score(name, s, data, kind, pace, tag="", simk={}):
    spr = spread_of(pace, data) if kind == "A" else None
    g, sc = VS.season_scores(s, data, kind, spr, **simk.get("season", {}))
    g2 = g.drop(columns=["dts", "w"]).assign(score=sc.mean(0), lo=np.quantile(sc, .05, 0), hi=np.quantile(sc, .95, 0)); g2.to_csv(OUT / f"{name}{tag}_seasons.csv", index=False)
    pc = VS.peak_career(g, sc, data); pc.to_csv(OUT / f"{name}{tag}_scores.csv", index=False)
    g3, sc3 = VS.season_scores(s, data, kind, spr, use_fit=True, **simk.get("season", {})); VS.peak_career(g3, sc3, data).to_csv(OUT / f"{name}{tag}_withfit_scores.csv", index=False)
    st = data.res[data.res.started & (data.res.session == "gp")].groupby("d").size()
    el = [i for i in range(data.n["d"]) if st.get(i, 0) >= VS.MIN_CAREER_STARTS]
    VS.adaptability(s, data, kind, spr, el, **simk.get("adapt", {})).to_csv(OUT / f"{name}{tag}_adapt.csv", index=False)
    VS.coefs(s, data).to_csv(OUT / f"{name}{tag}_coefs.csv", index=False)
    return pc

def fit_stage(name):
    cfg = SET[name]; res, wet, pace = load(); data = Data(res, wet, cfg["start"])
    m, D = inputs(cfg, data, pace)
    W, N, S = (80, 80, 1) if FAST else (cfg["warm"], cfg["samp"], cfg["seg"])
    s = fit(m, D, W, N, tree=cfg["tree"], ckpt=None if FAST else CK / f"{name}.pkl", segments=S)
    np.savez_compressed(OUT / f"{name}_post.npz", **{k: v for k, v in s.items() if k in ("tau_form", "fit_sd", "ad_t_mu", "ad_r_mu", "ad_t_sd", "ad_r_sd", "r_scale", "kappa_r", "kappa_q")})
    d = diag(s); json.dump(d, open(OUT / f"{name}_diag.json", "w")); print(name, d, flush=True)
    hy = {k: [round(float(x), 3) for x in (s[k].mean(), *np.quantile(s[k], [.05, .95]))] for k in ["tau_form", "fit_sd", "ad_t_mu", "ad_r_mu", "ad_t_sd", "ad_r_sd", "r_scale"]}
    for k in ("kappa_r", "kappa_q"):
        if k in s: hy[k] = s[k].mean((0, 1)).round(3).tolist()
    json.dump(hy, open(OUT / f"{name}_hyper.json", "w")); print(hy, flush=True)
    pc = score(name, s, data, cfg["kind"], pace)
    print(pc.dropna(subset=["peak"]).sort_values("peak").head(12)[["name", "peak", "peak_lo", "peak_hi", "peak_span"]].round(2).to_string()); print(pc.sort_values("career", ascending=False).head(12)[["name", "career", "career_lo", "career_hi", "career_avg"]].round(1).to_string())

def holdout_stage(name):
    cfg = SET[name]; res, wet, pace = load()
    full = Data(res, wet, cfg["start"])
    hold = full.res[(full.res["round"] % 5 == 0)].race_id.unique().tolist()
    data = Data(res, wet, cfg["start"], holdout=hold)
    m, D = inputs(cfg, data, pace); s = fit(m, D, 300 if not FAST else 60, 300 if not FAST else 40, tree=7, ckpt=None)
    f = {k: v.reshape((-1,) + v.shape[2:]) for k, v in s.items()}
    te = data.res[data.res.race_id.isin(hold) & (data.res.session == "gp")]; rows = []
    def perf_(x, key, wet_):
        sc = 1.0 if key == "q" else f["r_scale"]
        return f[f"skill_{key}"][:, x.d] + f[f"wet_{key}"][:, x.d] * wet_ + sc * (f["form"][:, x.dsi] + f["fit"][:, x["dt"]] + f["ad_t"][:, x.d] * x.x_t + f["ad_r"][:, x.d] * x.x_r)
    for col, key, lab in [("quali_pos", "q", "qualifying"), ("finish", "r", "race")]:
        t = te[te[col].notna()]
        if col == "finish": t = t[t.finished & ~t.no_race]
        out = []
        for (rid, team), g in t.groupby(["race_id", "team"]):
            if len(g) != 2: continue
            a, b = g.iloc[0], g.iloc[1]
            y = float(a[col] < b[col]); wt = float(a["wet_q"] if key == "q" else a["wet_r"])
            d = perf_(a, key, wt) - perf_(b, key, wt)
            if key == "r": d = d + (f["chg_r"][:, a.d] - f["chg_r"][:, b.d]) * float(a["chg_r"])
            ga, gb = (np.nan_to_num(float(a.grid), nan=20.) - 10.5) / 9.5, (np.nan_to_num(float(b.grid), nan=20.) - 10.5) / 9.5
            if cfg["kind"] == "B":
                kap = (np.c_[f["kappa_q" if key == "q" else "kappa_r"], np.ones(len(d))][:, a.decade] if "kappa_r" in f else 1.0)
                gr = (f["grid_era"][:, a.era] + f["grid_c"][:, a.c]) * (ga - gb) if key == "r" else 0.0
                p = expit(kap * d + gr).mean()
            elif key == "q":
                p = norm.cdf(d / (np.sqrt(2) * (f["sig_q"] + (f["sig_ql"] - f["sig_q"]) * wt))).mean()
            else:
                c = f["craft"][:, a.d] - f["craft"][:, b.d] + (f["craft_wet"][:, a.d] - f["craft_wet"][:, b.d]) * wt
                nz = np.random.default_rng(0).normal(size=len(d)) * np.sqrt(2) * (f["sig_r"] + (f["sig_rl"] - f["sig_r"]) * wt)
                p = expit(c + (f["grid_era"][:, a.era] + f["grid_c"][:, a.c]) * (ga - gb) + f["b_pace"] * (d + nz)).mean()
            tr = data.train[(data.train.season == a.season) & (data.train.team == a.team) & data.train.driver.isin([a.driver, b.driver]) & (data.train.session == "gp")]
            if col == "finish": tr = tr[tr.finished & ~tr.no_race]
            pv = tr.pivot_table(index="race_id", columns="driver", values=col).dropna()
            h = (pv[a.driver] < pv[b.driver]).sum() + 1 if a.driver in pv and b.driver in pv else 1
            hh = h / (len(pv) + 2)
            gridp = None
            if col == "finish": gridp = .72 if ga < gb else .28
            out.append(dict(y=y, model=p, h2h=hh, grid=gridp, season=a.season))
        o = pd.DataFrame(out); o = o[o.season >= data.report_from]
        def sc_(col2, sub):
            pr = sub[col2].clip(.001, .999)
            return dict(n=len(sub), accuracy=float(((pr > .5) == (sub.y == 1)).mean()), log_loss=float(-(sub.y * np.log(pr) + (1 - sub.y) * np.log(1 - pr)).mean()))
        eras = [("all", o)] + ([(f"{a}s", o[o.season // 10 * 10 == a]) for a in (1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020) if a >= data.report_from // 10 * 10] if cfg["start"] <= 1980 else [])
        for en, sub in eras:
            for c2, l2 in [("model", name), ("h2h", "Head-to-head so far"), ("grid", "Better grid")]:
                if c2 == "grid" and key == "q": continue
                rows.append(dict(session=lab, era=en, model=l2, **sc_(c2, sub)))
    v = pd.DataFrame(rows); v.to_csv(OUT / f"{name}_holdout.csv", index=False); print(v.round(3).to_string(), flush=True)

def synth_stage(name):
    """Invent true values including adaptability, generate data on the real calendar, then see what comes back."""
    cfg = SET[name]; res, wet, pace = load(); data = Data(res, wet, cfg["start"])
    rng = np.random.default_rng(int(os.environ.get("SYNTH_SEED", 11))); n = data.n; r = data.res.copy(); u = .3 if cfg["kind"] == "A" else 1.0
    T = dict(skill_q=rng.normal(0, .6 * u, n["d"]), skill_r=rng.normal(0, .5 * u, n["d"]), ad_t=rng.normal(-.2 * u, .3 * u, n["d"]), ad_r=rng.normal(-.2 * u, .3 * u, n["d"]), wet_r=rng.normal(0, .4 * u, n["d"]), wet_q=rng.normal(0, .4 * u, n["d"]), chg_r=rng.normal(0, .4 * u, n["d"]))
    cq, cr = rng.normal(0, 1.5 * u, n["t"]), rng.normal(0, 1.2 * u, n["t"]); F = np.zeros(n["ds"])
    for i in range(n["d"]):
        ix = data.car_idx[i][data.car_mask[i]]; w = np.cumsum(rng.normal(0, .15 * u, len(ix))); F[ix] = w - w.mean()
    fun = lambda g, key, wet_: T[f"skill_{key}"][g.d] + T[f"wet_{key}"][g.d] * wet_ + (cq if key == "q" else cr)[g.t] + F[g.dsi] + T["ad_t"][g.d] * g.x_t.values + T["ad_r"][g.d] * g.x_r.values
    if cfg["kind"] == "B":
        for rid, g in r[r.quali_pos.notna() & (r.session == "gp")].groupby("race_id"):
            r.loc[g.index, "quali_pos"] = (-(fun(g, "q", g.wet_q.values) + rng.gumbel(size=len(g)))).argsort().argsort() + 1
        for rid, g in r[r.finished].groupby("race_id"):
            gr = (np.nan_to_num(g.grid.values.astype(float), nan=20.) - 10.5) / 9.5
            r.loc[g.index, "finish"] = (-(fun(g, "r", g.wet_r.values) + T["chg_r"][g.d] * g.chg_r.values - 1.3 * gr + rng.gumbel(size=len(g)))).argsort().argsort() + 1
        data2 = Data(r.drop(columns=["wet_q", "wet_r", "chg_r", "ts", "dteam", "ds", "d", "t", "dt", "c", "dsi", "era", "decade", "prog", "x_t", "x_r", "sprint"]), wet, cfg["start"])
        m, D = model_b, inputs_b(data2)
    else:
        p = pace.copy()
        for key, sg in [("q", .15), ("r", .25)]:
            sel = p.kind == key; mm = p[sel][["race_id", "driver", "cond"]].merge(r, on=["race_id", "driver"], how="left")
            val = T[f"skill_{key}"][mm.d] + T[f"wet_{key}"][mm.d] * (mm.cond == "low").values + (cq if key == "q" else cr)[mm.t] + F[mm.dsi] + T["ad_t"][mm.d] * mm.x_t.values + T["ad_r"][mm.d] * mm.x_r.values
            p.loc[sel, "gap"] = -(val + rng.normal(0, sg, len(mm)))
        data2 = data; m, D = model_a, inputs_a(data, p)
    s = fit(m, D, 500 if not FAST else 60, 400 if not FAST else 40, tree=7)
    f = {k: v.reshape((-1,) + v.shape[2:]) for k, v in s.items()}
    st = data2.res[data2.res.started & (data2.res.session == "gp")].groupby("d").size(); el = np.array([i for i in range(n["d"]) if st.get(i, 0) >= 40])
    ev_t = data2.res[data2.res.x_team_n == 0].groupby("d").size(); ev_r = data2.res[data2.res.x_reg_n == 0].groupby("d").size()
    rows = []
    for k, sel in [("skill_q", el), ("skill_r", el), ("wet_q", el), ("wet_r", el), *([("chg_r", el)] if cfg["kind"] == "B" else []), ("ad_t", np.array([i for i in el if ev_t.get(i, 0) >= 2])), ("ad_r", np.array([i for i in el if ev_r.get(i, 0) >= 2]))]:
        x = f[k][:, sel]; t = T[k][sel]; me = x.mean(0); off = (me - t).mean() if k.startswith("skill") else 0
        lo, hi = np.quantile(x, .05, 0), np.quantile(x, .95, 0)
        rows.append(dict(model=name, coef=k, n=len(sel), spearman=spearmanr(me, t)[0] if len(sel) > 2 else None, coverage_90=float(((t + off >= lo) & (t + off <= hi)).mean())))
    first = data2.res.groupby("d").season.min(); e50 = np.array([i for i in range(n["d"]) if st.get(i, 0) >= 20 and first.get(i, 9999) <= 1960])
    if cfg["start"] < 1960:
        for k in ("skill_q", "skill_r"):
            off = (f[k][:, el].mean(0) - T[k][el]).mean(); x = f[k][:, e50]; t = T[k][e50]; me = x.mean(0); lo, hi = np.quantile(x, .05, 0), np.quantile(x, .95, 0)
            rows.append(dict(model=name, coef=k + "_1950s", n=len(e50), spearman=spearmanr(me, t)[0], coverage_90=float(((t + off >= lo) & (t + off <= hi)).mean()), era_bias=float((me - t).mean() - off)))
    v = pd.DataFrame(rows); v.to_csv(OUT / f"{name}_synth.csv", index=False); print(v.round(3).to_string(), flush=True)

def sens_stage():
    rows = []
    base = {k: pd.read_csv(OUT / f"{k}_scores.csv").set_index("driver") for k in ("A18", "B18")}
    def rec(lbl, name, res, wet, pace, tau=ADAPT_TAU):
        cfg = SET[name]; data = Data(res, wet, cfg["start"], tau=tau); m, D = inputs(cfg, data, pace)
        s = fit(m, D, 400 if not FAST else 50, 300 if not FAST else 40, tree=7)
        spr = spread_of(pace, data) if cfg["kind"] == "A" else None
        g, sc = VS.season_scores(s, data, cfg["kind"], spr, draws=80, weekends=20); pc = VS.peak_career(g, sc, data).set_index("driver")
        b = base[name]; ix = pc.index.intersection(b.index); pk = [i for i in ix if pc.peak.notna()[i] and b.peak.notna()[i]]
        rows.append(dict(change=lbl, model=name, peak_corr=spearmanr(pc.peak[pk], b.peak[pk])[0], career_corr=spearmanr(pc.career[ix], b.career[ix])[0], peak_top5=", ".join(pc.dropna(subset=["peak"]).sort_values("peak").name[:5]), career_top5=", ".join(pc.sort_values("career", ascending=False).name[:5])))
        print(rows[-1], flush=True); pd.DataFrame(rows).to_csv(OUT / "sensitivity.csv", index=False)
    res, wet, pace = load()
    r0, w0, p0 = load(orders=False)
    rec("Team orders not corrected", "B18", r0, wet, pace); rec("Team orders not corrected", "A18", r0, wet, pace)
    rec("Adaptation lasts about 12 races, not 6", "B18", res, wet, pace, 12.0); rec("Adaptation lasts about 12 races, not 6", "A18", res, wet, pace, 12.0)
    build_pace(race_low=1.20, quali_low=1.10, tag="_vlow"); rl, wl, pl_ = load(tag="_vlow")
    rec("Very low grip only (race laps 20%+ off dry)", "A18", res, wl, pl_); rec("Very low grip only (race laps 20%+ off dry)", "B18", res, wl, pace)
    x = res.copy(); x["error"] = x.error | x.collision
    rec("Collisions count as driver errors", "B18", x, wet, pace); rec("Collisions count as driver errors", "A18", x, wet, pace)
    y = res[res.session == "gp"]
    rec("Without sprints", "B18", y, wet, pace); rec("Without sprints", "A18", y, wet, pace[pace.session != "sprint"])

def sens80_stage():
    f0 = OUT / "sensitivity_B80.csv"; rows = pd.read_csv(f0).to_dict("records") if f0.exists() else []
    base = pd.read_csv(OUT / "B80_scores.csv").set_index("driver"); cfg = SET["B80"]
    def rec(lbl, res, wet, pace, tau=ADAPT_TAU, start=1966):
        if any(r["change"] == lbl for r in rows): return   # already finished before a restart
        data = Data(res, wet, start, tau=tau); m, D = inputs(cfg, data, pace)
        s = fit(m, D, 400, 300, tree=7)
        g, sc = VS.season_scores(s, data, "B", None, draws=80, weekends=20); pc = VS.peak_career(g, sc, data).set_index("driver")
        ix = pc[pc.starts >= 40].index.intersection(base[base.starts >= 40].index); pk = [i for i in ix if pc.peak.notna()[i] and base.peak.notna()[i]]
        rows.append(dict(change=lbl, model="B80", peak_corr=spearmanr(pc.peak[pk], base.peak[pk])[0], career_corr=spearmanr(pc.career[ix], base.career[ix])[0], peak_top5=", ".join(pc[pc.starts >= 40].dropna(subset=["peak"]).sort_values("peak").name[:5]), career_top5=", ".join(pc[pc.starts >= 40].sort_values("career", ascending=False).name[:5])))
        print(rows[-1], flush=True); pd.DataFrame(rows).to_csv(OUT / "sensitivity_B80.csv", index=False)
    res, wet, pace = load(); r0, _, _ = load(orders=False)
    rec("Team orders not corrected", r0, wet, pace)
    rec("Adaptation lasts about 12 races, not 6", res, wet, pace, 12.0)
    x = res.copy(); x["error"] = x.error | x.collision; rec("Collisions count as driver errors", x, wet, pace)
    y = res.copy(); y["team"] = y.team.str.split("~").str[0]; rec("Customer teams not separated", y, wet, pace)
    rec("No warm-up seasons before 1970", res, wet, pace, start=1970)
    os.environ["F1_MERGE_WET"] = "1"; rec("Wet and changeable merged", res, wet, pace); del os.environ["F1_MERGE_WET"]

def rescore(name):
    """Score an already-fitted model again from its saved draws."""
    cfg = SET[name]; res, wet, pace = load(); data = Data(res, wet, cfg["start"])
    st = pickle.load(open(CK / f"{name}.pkl", "rb"))
    s = {k: np.concatenate([c[k] for c in st["chunks"]], axis=1) for k in st["chunks"][0]}
    pc = score(name, s, data, cfg["kind"], pace)
    print(name, pc.dropna(subset=["peak"]).sort_values("peak").head(10)[["name", "peak"]].round(2).values.tolist(), pc.sort_values("career", ascending=False).head(10)[["name", "career"]].round(1).values.tolist(), flush=True)

def colscore():
    """B80 pace draws joined with the separately fitted collision tendency."""
    cfg = SET["B80"]; res, wet, pace = load(); data = Data(res, wet, cfg["start"])
    st = pickle.load(open(CK / "B80.pkl", "rb"))
    s = {k: np.concatenate([c[k] for c in st["chunks"]], axis=1) for k in st["chunks"][0]}
    c = np.load(OUT / "D80_collision_draws.npz"); s.update({k: c[k] for k in c.files})
    pc = score("D80", s, data, cfg["kind"], pace)
    print(pc.dropna(subset=["peak"]).sort_values("peak").head(10)[["name", "peak"]].round(2).values.tolist(), pc.sort_values("career", ascending=False).head(10)[["name", "career"]].round(1).values.tolist(), flush=True)

if args[0] == "colscore": colscore()
elif args[0] == "rescore": rescore(args[1])
elif args[0] == "holdout": holdout_stage(args[1])
elif args[0] == "synth": synth_stage(args[1])
elif args[0] == "sens": sens_stage()
elif args[0] == "sens80": sens80_stage()
else: fit_stage(args[0])
