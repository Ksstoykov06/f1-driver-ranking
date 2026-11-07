"""Race margin per finisher as a share of race distance: time gap on the lead lap, or laps down."""
import yaml, re, pandas as pd
from f1model.config import RAW, PROCESSED
def secs(t):
    if not t: return None
    p = str(t).lstrip("+").split(":")
    try: return sum(float(x) * 60 ** i for i, x in enumerate(reversed(p)))
    except ValueError: return None
rows = []
for y in range(1950, 2027):
    d = RAW / f"f1db/src/data/seasons/{y}/races"
    for rd in sorted(d.iterdir()) if d.exists() else []:
        f = rd / "race-results.yml"
        if not f.exists(): continue
        rr = yaml.safe_load(f.read_text(encoding="utf8")) or []; rid = yaml.safe_load((rd / "race.yml").read_text(encoding="utf8"))["id"]
        w = next((r for r in rr if r.get("position") == 1), None)
        if not w or not w.get("laps"): continue
        wt = secs(w.get("time"))
        for r in rr:
            g = None; gp = str(r.get("gap") or ""); lp = r.get("laps")
            if r.get("position") == 1: g = 0.0; src = "w"
            elif "lap" not in gp and secs(gp) is not None and wt: g = secs(gp) / wt; src = "time"
            elif isinstance(lp, int) and lp < w["laps"]: g = (w["laps"] - lp) / w["laps"]; src = "laps"
            else: src = "none"
            rows.append(dict(race_id=rid, season=y, driver=r["driverId"], deficit=g, src=src))
o = pd.DataFrame(rows); o.to_pickle(PROCESSED / "gaps_v2.pkl")
o["dec"] = o.season // 10 * 10
print(o.groupby("dec").src.value_counts(normalize=True).unstack().round(2)); print(o.deficit.describe().round(4).to_dict())
