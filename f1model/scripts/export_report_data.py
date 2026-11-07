"""Collects everything the walkthrough page charts into one JSON file."""
import json, numpy as np, pandas as pd
from f1model.config import *
from f1model.pipeline import load
from f1model.design import Design
from f1model.ingest_laps import load_session, WET_LAP_SHARE
res, qp, rp, wet = load(); ds = Design(res, wet)
out = {}
keep = ["rank","driver","name","races","score","score_lo","score_hi","best_fit","poor_fit","adapt_sigma","skill_q","skill_r","wet_q","wet_r","err_pct_per_race","peak","peak_lo","peak_hi","peak_span","career","career_lo","career_hi","p_best","p_top3","new_team_n","new_team_dev","reg_n","reg_dev"]
for k in "AB":
    t = pd.read_csv(OUT/f"ranking_{k}.csv"); cols = keep + (["racecraft","lap_spread"] if k=="A" else [])
    p = np.load(OUT/f"posterior_{k}.npz")
    q5 = lambda n,i: [round(float(x),3) for x in np.quantile(p[n][:,i],[.05,.95])]
    t["wet_q_ci"] = [q5("wet_q",ds.drivers.index(d)) for d in t.driver]
    t["wet_r_ci"] = [q5("wet_r",ds.drivers.index(d)) for d in t.driver]
    t["sk_q_ci"] = [q5("skill_q",ds.drivers.index(d)) for d in t.driver]
    t["sk_r_ci"] = [q5("skill_r",ds.drivers.index(d)) for d in t.driver]
    t["sig_ci"] = [q5("sigma_dev",ds.drivers.index(d)) for d in t.driver]
    out[f"rank{k}"] = json.loads(t[cols+["wet_q_ci","wet_r_ci","sk_q_ci","sk_r_ci","sig_ci"]].round(3).to_json(orient="records"))
    s = pd.read_csv(OUT/f"season_scores_{k}.csv")
    s["name"] = s.driver.map(ds.names)
    agg = s.groupby(["driver","name","season"]).apply(lambda g: np.average(g.score,weights=g.races),include_groups=False).reset_index(name="score")
    out[f"season{k}"] = json.loads(agg.round(2).to_json(orient="records"))
    cars = []
    for i,ts in enumerate(ds.ts):
        team,season = ts.split("|")
        cars.append(dict(team=team,season=int(season),q=round(float(p["car_q"][:,i].mean()),3),r=round(float(p["car_r"][:,i].mean()),3)))
    out[f"car{k}"] = cars
    out[f"hyper{k}"] = {x:[round(float(v),3) for v in (p[x].mean(),*np.quantile(p[x],[.05,.95]))] for x in ["age1","age2"]}
    out[f"grid{k}"] = [round(float(x),2) for x in p["grid_era"].mean(0)]
    circ = p["grid_c"].mean(0); o = np.argsort(circ)
    out[f"gridc{k}"] = [dict(c=ds.circuits[i],v=round(float(circ[i]+p["grid_era"].mean()),2)) for i in o]
# one real session, to show which laps count
df = load_session(RAW/"laps2024"/"Bahrain Grand Prix","Qualifying")
ex = df[df.code.isin(["VER","LEC"])].copy()
ex["fast"] = ex.groupby(["code","qs"]).time.transform("min")
def cls(r):
    if pd.isna(r.time) or r.is_in_out: return "out/in lap"
    if not r.clean_status or r.is_deleted: return "yellow or deleted"
    if r.time > r.fast*PUSH_WINDOW_DRY: return "cool-down"
    return "push lap"
ex["cls"] = ex.apply(cls,axis=1)
out["example"] = json.loads(ex[ex.time.notna()][["code","sesT","time","qs","cls"]].round(3).to_json(orient="records"))
# lap funnel
tot_q=tot_r=0
for y in range(FIRST_SEASON,LAST_SEASON+1):
    for gp in (RAW/f"laps{y}").iterdir():
        if gp.is_dir():
            for sess,name in [("Qualifying","q"),("Race","r")]:
                n=0
                for f in (gp/sess).glob("*/laptimes.json"):
                    n+=len(json.loads(f.read_text())["time"])
                if name=="q": tot_q+=n
                else: tot_r+=n
out["funnel"] = dict(q_total=tot_q,q_kept=int(qp.n.sum()),r_total=tot_r,r_kept=int(rp.n.sum()),q_rows=len(qp),r_rows=len(rp))
out["counts"] = dict(races=int(res.race_id.nunique()),drivers=len(ds.drivers),ranked=len(out["rankA"]),wet_q=int(wet.wet_q.sum()),wet_r=int(wet.wet_r.sum()),team_seasons=ds.n["t"],dts=ds.n["dt"])
m = res.drop_duplicates("race_id").set_index("race_id")
out["wet_races"] = [f"{m.season[i]} {m.gp[i]}" for i in wet.index[wet.wet_r]]
out["validation"] = json.loads(pd.read_csv(OUT/"validation_teammates.csv").round(3).to_json(orient="records"))
for f,k in [("validation_synthetic.csv","synthetic"),("sensitivity.csv","sensitivity")]:
    out[k] = json.loads(pd.read_csv(OUT/f).round(3).to_json(orient="records"))
out["diag"] = {k:json.load(open(OUT/f"diag_{k}.json")) for k in "AB"}
tm = res[res.started].groupby(["season","team"]).driver.unique()
json.dump(out,open(OUT/"report_data.json","w"))
print({k:(len(v) if hasattr(v,"__len__") else v) for k,v in out.items()}); print(out["funnel"],out["counts"],out["hyperA"],out["hyperB"],out["gridB"],out["gridA"]); print(out["gridcB"][:3],out["gridcB"][-3:]); print(out["diag"])
print([ (r["name"],r["rank"]) for r in out["rankA"] if r["name"] in("Oscar Piastri","Isack Hadjar","Valtteri Bottas")])
print([(r["name"],r["wet_r"],r["wet_r_ci"]) for r in out["rankB"][:5]])
print(pd.DataFrame(out["example"]).groupby(["code","cls"]).size())
