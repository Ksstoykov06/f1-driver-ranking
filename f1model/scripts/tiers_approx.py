"""Tiers without saved posterior draws: each driver's score is drawn from a normal matching the published mean and
90% interval (sd = width/3.29), independently. Ignoring correlation between drivers makes tiers a little wider
than full draws would. Peak: C50 (all-time from 1950, 25+ starts). Career and 50/50: B80 (from 1970, 40+ starts)."""
import sys, numpy as np, pandas as pd
THR = float(sys.argv[1]) if len(sys.argv) > 1 else .25
rng = np.random.default_rng(1); N = 4000
def draws(df, col):
    sd = (df[col + "_hi"] - df[col + "_lo"]) / 3.29
    return pd.DataFrame(rng.normal(df[col].values, sd.values, (N, len(df))), columns=df.index)
def tiers(V, info, better_low, top):
    R = (V if better_low else -V).rank(axis=1); med = R.median().sort_values()
    left = med.index[:top].tolist(); out = []; t = 0
    while left:
        t += 1; L = left[0]
        pb = {d: ((V[d] < V[L]) if better_low else (V[d] > V[L])).mean() for d in left[1:]}
        mem = [d for d in left[1:] if pb[d] >= THR]
        cut = max([left.index(m) for m in mem], default=0); members = left[:cut + 1]; left = left[cut + 1:]
        for d in members:
            out.append(dict(tier=t, driver=d, name=info.name[d], med_rank=float(med[d]), lo=float(R[d].quantile(.05)), hi=float(R[d].quantile(.95)),
                            p_top=float((R[d] <= 1).mean()), value=float(V[d].mean()), span=info.peak_span[d], starts=int(info.starts[d])))
    return pd.DataFrame(out)
c50 = pd.read_csv("outputs_v2/C50_scores.csv").set_index("driver")
c50 = c50[(c50.starts >= 25) & c50.peak.notna()]
tp = tiers(draws(c50, "peak"), c50, True, 30); tp.to_csv("outputs_v2/tiers_peak.csv", index=False)
b80 = pd.read_csv("outputs_v2/B80_scores.csv").set_index("driver")
b80 = b80[(b80.starts >= 40) & b80.peak.notna() & (b80["last"] >= 1970)]
P, C = draws(b80, "peak"), draws(b80, "career")
pm, cm = (10.5 - b80.peak).max(), b80.career.max()
comb = .5 * (10.5 - P) / pm + .5 * C / cm
tc = tiers(comb, b80, False, 30); tc["peak"] = b80.peak[tc.driver].values; tc["career"] = b80.career[tc.driver].values
tc.to_csv("outputs_v2/tiers_combined.csv", index=False)
tcar = tiers(C, b80, False, 20); tcar.to_csv("outputs_v2/tiers_career.csv", index=False)
pd.set_option("display.width", 220)
for n, t in (("PEAK from 1950", tp), ("COMBINED 50/50 from 1970", tc), ("CAREER from 1970", tcar)):
    print("==", n, "thr", THR); print(t.round(2).drop(columns=["driver"]).to_string(index=False))
