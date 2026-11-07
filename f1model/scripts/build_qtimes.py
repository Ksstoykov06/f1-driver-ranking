"""Qualifying deficit per driver: % behind the fastest in the last session segment they took part in, so track evolution between Q1, Q2 and Q3 doesn't count."""
import yaml, pandas as pd
from f1model.config import RAW, PROCESSED
def secs(t):
    if not t: return None
    p = str(t).split(":")
    try: return sum(float(x) * 60 ** i for i, x in enumerate(reversed(p)))
    except ValueError: return None
rows = []
for y in range(1950, 2027):
    d = RAW / f"f1db/src/data/seasons/{y}/races"
    for rd in sorted(d.iterdir()) if d.exists() else []:
        f = rd / "qualifying-results.yml"
        if not f.exists(): continue
        rr = yaml.safe_load(f.read_text(encoding="utf8")) or []; rid = yaml.safe_load((rd / "race.yml").read_text(encoding="utf8"))["id"]
        seg = {}
        for r in rr:
            for s, k in (("q3", "q3"), ("q2", "q2"), ("q1", "q1"), ("t", "time")):
                v = secs(r.get(k))
                if v: seg[r["driverId"]] = (s, v); break
        best = {}
        for s, v in seg.values(): best[s] = min(best.get(s, 1e9), v)
        for dr, (s, v) in seg.items(): rows.append(dict(race_id=rid, season=y, driver=dr, seg=s, qdef=(v / best[s] - 1)))
o = pd.DataFrame(rows); o.to_pickle(PROCESSED / "qtimes_v2.pkl"); o["dec"] = o.season // 10 * 10
print(o.groupby("dec").qdef.describe().round(4)[["count", "50%", "max"]]); print((o.qdef > .1).groupby(o.dec).mean().round(3).to_dict())
