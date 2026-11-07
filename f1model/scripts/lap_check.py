"""Check the results model (base_long, 1950 on, no lap data) against 2018+ lap times it never saw."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr, pearsonr
P = pd.read_pickle("data/processed/pace_v2.pkl"); P = P[(P.session == "gp") & (P.cond == "dry")]
cars = pd.read_csv("outputs_v2/gap/base_long_fullcars.csv"); cars[["team", "season"]] = cars.ts.str.split("|", expand=True); cars.season = cars.season.astype(int)
S = pd.read_csv("outputs_v2/gap/base_long_fullseasons.csv")
A = pd.read_csv("outputs_v2/A18_seasons.csv"); B = pd.read_csv("outputs_v2/B18_seasons.csv")
out = {}
# 1 cars: team's average dry pace gap (both drivers) per season vs the results model's car rating
for kind, col in (("q", "car_q"), ("r", "car_r")):
    t = P[P.kind == kind].groupby(["team", "season"]).gap.mean().reset_index().merge(cars, on=["team", "season"])
    rho = spearmanr(-t.gap, t[col])[0]
    # within-season version: removes year-to-year scale differences
    w = t.groupby("season").apply(lambda g: spearmanr(-g.gap, g[col])[0]).mean()
    out[f"car_{kind}"] = dict(n=len(t), spearman=round(rho, 3), within_season=round(w, 3))
    t["res"] = (-t.gap - t.groupby("season").gap.transform(lambda x: -x.mean())) ; t["cr"] = t[col] - t.groupby("season")[col].transform("mean")
    k = np.polyfit(t.cr, t.res, 1); t["miss"] = t.res - np.polyval(k, t.cr)
    if kind == "q": worst_cars = t.reindex(t.miss.abs().sort_values(ascending=False).index).head(10)[["team", "season", "gap", col, "miss"]].round(3)
# 2 drivers: teammate gap in dry qualifying/race pace vs the results model's season-score gap
for kind in ("q", "r"):
    p = P[P.kind == kind]; m = p.merge(p, on=["race_id", "team", "season"]); m = m[m.driver_x < m.driver_y]
    g = m.assign(d=m.gap_x - m.gap_y).groupby(["season", "team", "driver_x", "driver_y"]).agg(d=("d", "median"), n=("d", "size")).reset_index()
    g = g[g.n >= 6]
    sc = S.set_index(["driver", "season"]).score
    g["s"] = [sc.get((a, s), np.nan) - sc.get((b, s), np.nan) for a, b, s in zip(g.driver_x, g.driver_y, g.season)]
    g = g.dropna(); out[f"teammates_{kind}"] = dict(n=len(g), pearson=round(pearsonr(g.d, g.s)[0], 3), spearman=round(spearmanr(g.d, g.s)[0], 3),
        same_sign=round(float((np.sign(g.d) == np.sign(g.s)).mean()), 3))
    if kind == "q": tm = g
# 3 season scores: results model vs lap model (A18) and results-only 2018 (B18)
k = ["driver", "season"]; j = S.merge(A[k + ["score"]], on=k, suffixes=("_base", "_lap")).merge(B[k + ["score"]].rename(columns={"score": "score_b18"}), on=k)
out["seasons"] = dict(n=len(j), base_vs_lap=round(pearsonr(j.score_base, j.score_lap)[0], 3), b18_vs_lap=round(pearsonr(j.score_b18, j.score_lap)[0], 3),
    base_vs_b18=round(pearsonr(j.score_base, j.score_b18)[0], 3))
j["d"] = (j.score_base - j.score_base.mean()) / j.score_base.std() - (j.score_lap - j.score_lap.mean()) / j.score_lap.std()
for k_, v in out.items(): print(k_, v)
print("cars furthest from lap pace (qualifying, gap s/lap vs field; miss = lap edge not explained by rating, + means faster than rated):"); print(worst_cars.to_string(index=False))
tm["z"] = (tm.d - tm.d.mean()) / tm.d.std() - (tm.s - tm.s.mean()) / tm.s.std()
print("teammate pairs where lap gap and model disagree most:"); print(tm.reindex(tm.z.abs().sort_values(ascending=False).index).head(8).round(3).to_string(index=False))
print("driver-seasons rated most differently (z, + = lap model rates them better):"); print(j.reindex(j.d.abs().sort_values(ascending=False).index).head(10)[["driver", "season", "team", "score_base", "score_lap", "d"]].round(2).to_string(index=False))
lp = pd.read_csv("outputs_v2/A18_scores.csv").set_index("driver"); bp = pd.read_csv("outputs_v2/gap/base_long_fullscores.csv").set_index("driver")
c = lp[["peak"]].join(bp.peak, rsuffix="_base").dropna(); print("peak, 2018+ drivers: n", len(c), "spearman", round(spearmanr(c.peak, c.peak_base)[0], 3))
c["rl"] = c.peak.rank(); c["rb"] = c.peak_base.rank(); print(c.sort_values("rl").head(12).round(2).to_string())
pd.Series(out).to_json("outputs_v2/gap/lap_check.json")
# like-for-like peak: best 3 consecutive seasons from 2018 on, races-weighted, same rule for both models
def pk(df):
    r = {}
    for d, g in df[df.season >= 2018].groupby("driver"):
        g = g.set_index("season").sort_index(); best = np.nan
        for s in g.index:
            w = [y for y in (s, s + 1, s + 2) if y in g.index]
            if len(w) == 3: v = np.average(g.loc[w, "score"], weights=g.loc[w, "races"]); best = v if np.isnan(best) else min(best, v)
        r[d] = best
    return pd.Series(r)
c2 = pd.DataFrame(dict(lap=pk(A), base=pk(S))).dropna(); c2["rl"] = c2.lap.rank(); c2["rb"] = c2.base.rank()
print("2018+ only peak: n", len(c2), "spearman", round(spearmanr(c2.lap, c2.base)[0], 3)); print(c2.sort_values("rl").head(15).round(2).to_string())
c2.to_csv("outputs_v2/gap/lap_check_peak2018.csv")
