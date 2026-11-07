#!/usr/bin/env python
"""2018+ pilot. Keeps pipeline A (laps) and pipeline B (results) separate throughout.
  python run_pilot.py                 full run
  python run_pilot.py --fast          smoke test
"""
import argparse, json, warnings, numpyro
numpyro.set_host_device_count(2)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from f1model.config import *
from f1model.design import Design
from f1model import pipeline as P, scoring as S, validate as V, ingest_laps

ap = argparse.ArgumentParser(); ap.add_argument("--fast", action="store_true")
ap.add_argument("--only", default="main,validate,synthetic,sensitivity")
ap.add_argument("--kinds", default="B,A")
a = ap.parse_args()
W, N = (100, 100) if a.fast else (800, 700)
WM, NM, TREE = (100, 100, 8) if a.fast else (1500, 1500, 10)   # the rankings get the longest, deepest chains
SIMK = dict(draws=60, weekends=10) if a.fast else dict(draws=200, weekends=100)
stages = a.only.split(",")
OUT.mkdir(exist_ok=True)

def diag(m, name):
    import numpyro.diagnostics as nd
    g = m.get_samples(group_by_chain=True)
    rh = {k: np.nanmax(np.asarray(nd.gelman_rubin(np.asarray(v)))) for k, v in g.items() if not k.endswith("_z")}
    ess = {k: np.nanmin(np.asarray(nd.effective_sample_size(np.asarray(v)))) for k, v in g.items() if not k.endswith("_z")}
    d = dict(pipeline=name, max_rhat=float(max(rh.values())), worst=max(rh, key=rh.get), min_ess=float(min(ess.values())),
             min_ess_param=min(ess, key=ess.get))
    print(d, flush=True); return d

res, qp, rp, wet = P.load()
ds = Design(res, wet)
started = ds.res[ds.res.started].groupby("d").race_id.nunique()
elig = [i for i in range(ds.n["d"]) if started.get(i, 0) >= MIN_RACES_TO_RANK]
spread = P.lap_spread(qp, ds)
print("drivers", ds.n["d"], "ranked", len(elig), flush=True)

def rank_pipeline(s, kind, label="", ds_=ds, simk=SIMK, elig_=None, spr=None):
    elig_ = elig if elig_ is None else elig_; spr = spread if spr is None else spr
    sc = S.main_scores(s, ds_, kind, spr, eligible=elig_, **simk)
    return S.coef_table(s, ds_, kind, elig_, sc, spr), sc

if "main" in stages:
    summary = {}
    for kind, fitter in [("B", lambda: P.fit_b(ds, warmup=WM, samples=NM, tree=TREE)), ("A", lambda: P.fit_a(ds, qp, rp, warmup=WM, samples=NM, tree=TREE))]:
        if kind not in a.kinds.split(","):
            continue
        m = fitter(); summary[kind] = diag(m, kind); json.dump(summary[kind], open(OUT / f"diag_{kind}.json", "w"))
        s = {k: np.asarray(v) for k, v in m.get_samples().items() if not k.endswith("_z")}
        np.savez_compressed(OUT / f"posterior_{kind}.npz", **s)
        t, sc = rank_pipeline(s, kind)
        p1 = S.prob_rank_top(sc["main"], elig, ds, 1); p3 = S.prob_rank_top(sc["main"], elig, ds, 3)
        t["p_best"] = t.driver.map(p1); t["p_top3"] = t.driver.map(p3)
        g, ss = S.season_scores(s, ds, kind, spread, draws=SIMK["draws"], weekends=min(SIMK["weekends"], 40))
        pc = S.peak_and_career(g, ss)
        t = t.merge(pc, on="driver", how="left").merge(S.adaptation_events(s, ds), on="driver", how="left")
        g["score"] = ss.mean(0); g.to_csv(OUT / f"season_scores_{kind}.csv", index=False)
        t.to_csv(OUT / f"ranking_{kind}.csv", index=False); print(t.head(12)[["rank", "name", "score", "score_lo", "score_hi", "p_best"]].round(2), flush=True)
    A, B = pd.read_csv(OUT / "ranking_A.csv"), pd.read_csv(OUT / "ranking_B.csv")
    j = A.merge(B, on="driver", suffixes=("_A", "_B"))
    summary["agreement"] = {c: float(spearmanr(j[c + "_A"], j[c + "_B"])[0]) for c in ["score", "skill_q", "skill_r", "adapt_sigma", "peak", "career", "wet_q", "wet_r"] if j[c + "_A"].notna().sum() > 3 and j.dropna(subset=[c + "_A", c + "_B"]).shape[0] > 3}
    summary["agreement"] = {c: float(j[[c + "_A", c + "_B"]].dropna().corr("spearman").iloc[0, 1]) for c in summary["agreement"]}
    summary["top10_overlap"] = len(set(A.driver[:10]) & set(B.driver[:10]))
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1); print(summary, flush=True)

if "validate" in stages:
    hold = V.holdout_races(ds.res)
    dh = Design(res, wet, holdout=hold)
    sam = {"B": P.fit_b(dh, warmup=W, samples=N).get_samples(), "A": P.fit_a(dh, qp, rp, warmup=W, samples=N).get_samples()}
    v = V.teammate_test(dh, sam); v.to_csv(OUT / "validation_teammates.csv", index=False); print(len(hold), "held-out races\n", v.round(3), flush=True)

if "synthetic" in stages:
    rng = np.random.default_rng(42); rows = []
    r2, truth = V.synthetic_results(ds, rng)
    dsyn = Design(r2.drop(columns=[c for c in ["wet_q", "wet_r"] if c in r2]) , wet)
    m = P.fit_b(dsyn, warmup=W, samples=N).get_samples()
    rows += [dict(pipeline="B", **x) for x in V.recovery(m, truth, dsyn, elig)]
    q2, r3, truth = V.synthetic_laps(ds, qp, rp, rng)
    m = P.fit_a(ds, q2, r3, warmup=W, samples=N).get_samples()
    rows += [dict(pipeline="A", **x) for x in V.recovery(m, truth, ds, elig)]
    pd.DataFrame(rows).to_csv(OUT / "validation_synthetic.csv", index=False); print(pd.DataFrame(rows).round(3), flush=True)

if "sensitivity" in stages:
    base = {k: pd.read_csv(OUT / f"ranking_{k}.csv").set_index("driver") for k in "AB"}
    fw, fn = (60, 60) if a.fast else (500, 400)
    simk = dict(draws=100, weekends=60) if not a.fast else dict(draws=30, weekends=5)
    rows = []
    def record(label, kind, s, d_, el, spr=None):
        t, _ = rank_pipeline(s, kind, ds_=d_, simk=simk, elig_=el, spr=spr)
        shared = t.set_index("driver").score.index.intersection(base[kind].index)
        rho = spearmanr(t.set_index("driver").score[shared], base[kind].score[shared])[0]
        rows.append(dict(scenario=label, pipeline=kind, spearman_vs_base=rho, top5=", ".join(t.name[:5]))); print(rows[-1], flush=True)
    res_c, qp_c, rp_c, wet_c = P.load(collisions_as_errors=True); dc = Design(res_c, wet_c)
    record("collisions count as errors", "B", P.fit_b(dc, warmup=fw, samples=fn).get_samples(), dc, elig)
    record("collisions count as errors", "A", P.fit_a(dc, qp_c, rp_c, warmup=fw, samples=fn).get_samples(), dc, elig)
    # without 2026: the newest regulations are only part-run
    r26 = res[res.season < 2026]; d26 = Design(r26, wet); st = d26.res[d26.res.started].groupby("d").race_id.nunique()
    e26 = [i for i in range(d26.n["d"]) if st.get(i, 0) >= MIN_RACES_TO_RANK]
    qa, ra = qp[qp.race_id.isin(r26.race_id)], rp[rp.race_id.isin(r26.race_id)]
    record("exclude 2026", "B", P.fit_b(d26, warmup=fw, samples=fn).get_samples(), d26, e26)
    record("exclude 2026", "A", P.fit_a(d26, qa, ra, warmup=fw, samples=fn).get_samples(), d26, e26, P.lap_spread(qa, d26))
    # lap-filter variants (pipeline A only unless wet labels change)
    ingest_laps.build_laps(traffic_gap=0, tag="_notraffic")
    _, qn, rn, wn = P.load(tag="_notraffic"); dn = Design(res, wn)
    record("race laps in traffic kept", "A", P.fit_a(dn, qn, rn, warmup=fw, samples=fn).get_samples(), dn, elig)
    ingest_laps.WET_LAP_SHARE = 0.10
    ingest_laps.build_laps(tag="_wet10")
    _, qw, rw, ww = P.load(tag="_wet10"); dw = Design(res, ww)
    record("wider wet definition (10% of laps)", "B", P.fit_b(dw, warmup=fw, samples=fn).get_samples(), dw, elig)
    record("wider wet definition (10% of laps)", "A", P.fit_a(dw, qw, rw, warmup=fw, samples=fn).get_samples(), dw, elig)
    pd.DataFrame(rows).to_csv(OUT / "sensitivity.csv", index=False)
