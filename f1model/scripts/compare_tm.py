"""Compare remove-a-teammate reruns with the full base fit. Usage: compare_tm.py [x__y ...] (default all found)."""
import sys, glob, os, pandas as pd, numpy as np
G = "outputs_v2/gap/"
b = pd.read_csv(G + "base_long_fullscores.csv").set_index("driver")
c = pd.read_csv("outputs_v2/C50_scores.csv").set_index("driver") if os.path.exists("outputs_v2/C50_scores.csv") else None
runs = sys.argv[1:] or sorted(f[len(G) + 3:-15] for f in glob.glob(G + "tm_*_fullscores.csv"))
el = b[b.starts >= 25]
def rk(s): return s.loc[el.index].rank().reindex(b.index).fillna(-1)
rows = []
for r in runs:
    x, y = r.split("__"); t = pd.read_csv(G + f"tm_{r}_fullscores.csv").set_index("driver").reindex(b.index)
    for who in (x, y):
        rows.append(dict(run=r, driver=who, role="removed-from" if who == x else "teammate",
            peak_b=b.peak[who], peak_t=t.peak[who], d_peak=t.peak[who] - b.peak[who],
            rank_b=int(rk(el.peak)[who]), rank_t=int(rk(t.peak)[who]),
            car_b=b.career[who], car_t=t.career[who], crank_b=int(rk(-el.career)[who]), crank_t=int(rk(-t.career)[who]),
            peak_unc=(b.peak_hi[who] - b.peak_lo[who])))
    dm = (t.peak - b.peak).loc[el.index].abs()
    rows[-1]["others_mean_move"] = rows[-2]["others_mean_move"] = round(dm.drop([x, y], errors="ignore").mean(), 3)
o = pd.DataFrame(rows); pd.set_option("display.width", 250); print(o.round(2).to_string(index=False))
if c is not None:
    cc = c.reindex(el.index).dropna(subset=["peak"]); print("ordinary noise (C50 vs base) per-driver peak move, mean", round((cc.peak - el.peak.loc[cc.index]).abs().mean(), 3))
o.to_csv("outputs_v2/gap/tm_summary.csv", index=False)
