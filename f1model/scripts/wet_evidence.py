"""Evidence for wet labels. 2018+: share of laps on inters/wets. All years: winner's pace and fastest lap vs other years on the same layout."""
import yaml, re, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from f1model.config import RAW, PROCESSED
from f1model.ingest_laps import load_session
B = RAW / "f1db/src/data/seasons"
def sec(t):
    if not t: return None
    p = str(t).split(":"); return sum(float(x) * 60 ** i for i, x in enumerate(reversed(p)))
rows = []
for y in range(1965, 2027):
    for rd in sorted((B / str(y) / "races").iterdir()):
        r = yaml.safe_load((rd / "race.yml").read_text()); rr = yaml.safe_load((rd / "race-results.yml").read_text()) if (rd / "race-results.yml").exists() else None
        if not rr: continue
        w = rr[0]; fl = yaml.safe_load((rd / "fastest-laps.yml").read_text()) if (rd / "fastest-laps.yml").exists() else None
        q = yaml.safe_load((rd / "qualifying-results.yml").read_text()) if (rd / "qualifying-results.yml").exists() else None
        pole = None
        if q:
            ts = [sec(x.get(k)) for x in q[:3] for k in ("time", "q3", "q2", "q1") if x.get(k)]; pole = min(ts) if ts else None
        rows.append(dict(season=y, gp=rd.name.split("-", 1)[1], layout=r.get("circuitLayoutId"), length=r.get("courseLength"), laps_sched=r.get("laps"), wlaps=w.get("laps"), wtime=sec(w.get("time")), fl=sec(fl[0].get("time")) if fl else None, pole=pole, dnf_rate=np.mean([x.get("reasonRetired") in ("Accident", "Spun off", "Spin", "Collision") for x in rr]), winner=w["driverId"]))
d = pd.DataFrame(rows); d["avg"] = d.wtime / d.wlaps
# year trend removed via nearest other races on this layout: median of up to 6 nearest seasons
def rel(col):
    out = []
    for i, r in d.iterrows():
        o = d[(d.layout == r.layout) & (d.season != r.season) & d[col].notna() & (abs(d.season - r.season) <= 6)]
        out.append(r[col] / o[col].median() - 1 if len(o) >= 2 and r[col] == r[col] else np.nan)
    return out
d["avg_rel"] = rel("avg"); d["fl_rel"] = rel("fl")
# tyre share from laps
sh = []
for y in range(2018, 2027):
    p = RAW / f"laps{y}"
    if not p.exists(): continue
    for g in sorted(p.iterdir()):
        try:
            f = load_session(g, "Race")
            if f is None or not len(f): continue
            sh.append(dict(season=y, event=g.name, wet_share=float(f.wet_tyre.mean()), laps=len(f)))
        except Exception as e: pass
s = pd.DataFrame(sh); s.to_csv("outputs_v2/wet_tyre_share.csv", index=False); d.to_csv("outputs_v2/wet_pace_evidence.csv", index=False)
print(s.sort_values("wet_share", ascending=False).head(26).round(2).to_string()); print(len(s))
