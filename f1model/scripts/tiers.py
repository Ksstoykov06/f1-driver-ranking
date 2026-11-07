"""Tiers from posterior draws. A tier starts with the best unplaced driver (by median rank); a driver joins it
if he has at least a THR chance of really being better than that leader. Peak: all-time from 1950, 25+ starts.
Combined: 1970+ seasons only (matching the official career cut), 40+ starts, 50/50 peak and career."""
import pickle, sys, numpy as np, pandas as pd
THR = float(sys.argv[1]) if len(sys.argv) > 1 else .25
z = pickle.load(open("outputs_v2/gap/base_draws_scoredraws.pkl", "rb")); g, sc, names = z["g"].reset_index(drop=True), z["sc"], z["names"]
def summarise(g, sc, min_starts, from_year=None):
    if from_year: keep = g.index[g.season >= from_year]; g = g.loc[keep]; sc = sc[:, keep]; g = g.reset_index(drop=True)
    P, C, rows = {}, {}, []
    for d, h in g.groupby("driver"):
        h = h.sort_values("season"); a = sc[:, h.index]; ss = h.season.values
        if h.races.sum() < min_starts: continue
        w = [(a[:, i:i + 3].mean(), i) for i in range(len(ss) - 2) if ss[i + 2] - ss[i] == 2]
        if not w: continue
        _, i = min(w); P[d] = a[:, i:i + 3].mean(1); C[d] = ((10.5 - a) * (h.races.values / 22)).sum(1)
        rows.append(dict(driver=d, name=names[d], span=f"{ss[i]}–{ss[i + 2]}", starts=int(h.races.sum())))
    info = pd.DataFrame(rows).set_index("driver"); P = pd.DataFrame(P); C = pd.DataFrame(C)
    return info, P, C
def tiers(V, info, better_low, top=30):
    R = (V if better_low else -V).rank(axis=1); med = R.median().sort_values()
    order = med.index[:top].tolist(); left = order[:]; out = []; t = 0
    while left:
        t += 1; L = left[0]; members = [L]
        for d in left[1:]:
            pb = ((V[d] < V[L]) if better_low else (V[d] > V[L])).mean()
            if pb >= THR: members.append(d)
        # keep tiers contiguous in median order
        cut = max(left.index(m) for m in members); members = left[:cut + 1]; left = left[cut + 1:]
        for d in members:
            out.append(dict(tier=t, driver=d, name=info.name[d], med_rank=float(med[d]), lo=float(R[d].quantile(.05)), hi=float(R[d].quantile(.95)),
                p_top=float((R[d] <= 1).mean()), value=float(V[d].mean()), span=info.span[d], starts=info.starts[d]))
    return pd.DataFrame(out)
info, P, C = summarise(g, sc, 25)
tp = tiers(P, info, True); tp.to_csv("outputs_v2/tiers_peak.csv", index=False)
i2, P2, C2 = summarise(g, sc, 40, 1970)
pm, cm = (10.5 - P2.mean()).max(), C2.mean().max()
comb = .5 * (10.5 - P2) / pm + .5 * C2 / cm
tc = tiers(comb, i2, False); tc["peak"] = P2.mean()[tc.driver].values; tc["career"] = C2.mean()[tc.driver].values; tc.to_csv("outputs_v2/tiers_combined.csv", index=False)
tcar = tiers(C2, i2, False, 20); tcar.to_csv("outputs_v2/tiers_career.csv", index=False)
pd.set_option("display.width", 200)
for n, t in (("PEAK from 1950", tp), ("COMBINED 50/50 from 1970", tc), ("CAREER from 1970", tcar)):
    print("==", n, "threshold", THR); print(t.round(2).drop(columns=["driver"]).to_string(index=False))
