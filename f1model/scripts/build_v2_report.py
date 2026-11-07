"""Builds outputs/f1-model-walkthrough.html from outputs_v2."""
import json, re, sys, pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
sys.path.insert(0, "scripts")
from car_check import standings
R = Path(__file__).resolve().parents[1]; O = R / "outputs_v2"; P = R / "outputs_v2_third"
rd = lambda n: pd.read_csv(O / n)
MIN = 40
def cars(b):
    c = b.drop_duplicates(["season", "team"])[["season", "team", "car_rank"]].merge(standings(), on=["season", "team"], how="left").dropna(subset=["wcc"])
    return c, c.groupby("season")[["car_rank", "wcc"]].apply(lambda x: spearmanr(x.car_rank, x.wcc)[0])
b, bo = rd("B80_beyond_car.csv"), pd.read_csv(P / "B80_beyond_car.csv")
cn, pn = cars(b); co, po = cars(bo)
def summ(c, p):
    t = c[c.car_rank == 1]
    return [p.mean(), p.loc[1980], (t.wcc <= 2).mean(), (t.wcc > 2).sum()]
sn, so = summ(cn, pn), summ(co, po)
D = dict(check=[dict(season=int(s), new=float(pn[s]), old=(float(po[s]) if s in po.index else None)) for s in pn.index])
D["check_summary"] = [["Average correlation, all seasons", f"{so[0]:.2f}", f"{sn[0]:.2f}"], ["1980 correlation", f"{so[1]:.2f}", f"{sn[1]:.2f}"], ["1970-75 average correlation", "not covered", f"{pn.loc[1970:1975].mean():.2f}"], ["Best car finished top 2 in constructors'", f"{so[2]*100:.0f}%", f"{sn[2]*100:.0f}%"], ["Seasons where the best car finished below 2nd", int(so[3]), int(sn[3])]]
scores = {}
for m in ("B80", "B18", "A18"):
    s = rd(f"{m}_scores.csv"); s = s[s.starts >= (MIN if m == "B80" else 40)]
    n = 20 if m == "B80" else 12
    scores[m] = dict(peak=s.dropna(subset=["peak"]).sort_values("peak").head(n).to_dict("records"), career=s.sort_values("career", ascending=False).head(n).to_dict("records"))
D["scores"] = scores
c50 = rd("C50_scores.csv"); at = c50[c50.starts >= 25].dropna(subset=["peak"]).sort_values("peak").head(30)
D["alltime"] = at[["name", "peak", "peak_lo", "peak_hi", "peak_span", "starts"]].to_dict("records")
D["alltime_t"] = [[i + 1, r["name"], f"{r.peak:.2f}", f"{r.peak_lo:.1f}–{r.peak_hi:.1f}", r.peak_span, int(r.starts)] for i, (_, r) in enumerate(at.iterrows())]
D["dom"] = [[1954, "Mercedes", "Ferrari", "16%"], [1957, "Maserati", "Ferrari", "11%"], [1962, "Lotus", "Ferrari", "2%"], [1963, "Lotus", "Ferrari", "10%"]]
el = c50[(c50["first"] < 1966) & (c50.starts >= 25)].sort_values("career", ascending=False).head(16)
D["early"] = [[r["name"], f"{r.career:+.1f}", f"{r.career_avg:.1f}", int(r.starts), f"{int(r['first'])}–{int(r['last'])}"] for _, r in el.iterrows()]
s80 = rd("B80_scores.csv"); top = s80[s80.starts >= MIN].dropna(subset=["peak"]).sort_values("peak").head(24)
big = top.name.tolist()
for x in ["Lewis Hamilton", "Fernando Alonso", "Sebastian Vettel", "Felipe Massa", "Mark Webber", "Kimi Räikkönen", "Charles Leclerc", "Lando Norris", "Sergio Pérez", "Nico Rosberg", "Valtteri Bottas", "Daniel Ricciardo"]:
    if x not in big: big.append(x)
D["gain_names"] = big
D["gain"] = {n: b[(b.name == n)].sort_values("season")[["season", "team", "car_rank", "cars", "gain", "lo", "hi"]].to_dict("records") for n in big}
se = rd("B80_seasons.csv"); nm = s80.set_index("driver").name
se["name"] = se.driver.map(nm)
x = s80[s80.starts >= MIN].dropna(subset=["peak"]).copy(); x["p"] = (10.5 - x.peak) / (10.5 - x.peak).max(); x["c"] = x.career / x.career.max()
for w in (75, 50, 25): x[f"m{w}"] = x[f"m{w}"] = (w / 100 * x.p + (1 - w / 100) * x.c).rank(ascending=False).astype(int)
tp = x[(x.m75 <= 10) | (x.m50 <= 10) | (x.m25 <= 10)].sort_values("m50")
D["combined"] = [[r["name"], int(r.m50), int(r.m75), int(r.m25), f"P{r.peak:.1f}", f"{r.career:+.0f}", int(r.starts), int(r["first"])] for _, r in tp.iterrows()]
D["seasons"] = {n: se[se.name == n].sort_values("season")[["season", "team", "score", "lo", "hi"]].to_dict("records") for n in big}
y = b[(b.season == 2012)].sort_values("gain", ascending=False)
y = y[y.name.isin(["Fernando Alonso", "Felipe Massa", "Sebastian Vettel", "Mark Webber", "Lewis Hamilton", "Jenson Button", "Kimi Räikkönen", "Romain Grosjean"])]
D["t2012"] = [[r["name"], r.team.replace("-f1", "").replace("-", " ").title().replace("Mclaren", "McLaren"), f"{r.car_rank} of {r.cars}", f"{r.gain:+.2f}", f"{r.lo:+.1f} to {r.hi:+.1f}"] for _, r in y.iterrows()]
rows = []
for n in top.name.head(14).tolist() + [x for x in ["Lewis Hamilton", "Sebastian Vettel", "Kimi Räikkönen"] if x not in top.name.head(14).tolist()]:
    g = b[(b.name == n)]; bb = g[g.car_rank == 1]; l = g[g.car_rank >= 4]
    rows.append([n, len(bb), f"{bb.gain.mean():+.2f}" if len(bb) else "–", len(l), f"{l.gain.mean():+.2f}" if len(l) else "–", "P" + f"{s80[s80.name == n].peak.iloc[0]:.1f}"])
D["cars"] = rows
a = rd("B80_adapt.csv"); ad = {}
a1 = a[a.team_moves >= 3].sort_values("team_loss").head(16); a2 = a[a.rule_changes >= 3].sort_values("rule_loss").head(16)
for x in ["Michael Schumacher", "Fernando Alonso", "Lewis Hamilton"]:
    if x not in a1.name.values: a1 = pd.concat([a1, a[a.name == x]])
    if x not in a2.name.values: a2 = pd.concat([a2, a[a.name == x]])
ad = dict(team=a1.to_dict("records"), rule=a2.to_dict("records")); D["adapt"] = ad
wet = {}
for m in ("B80", "A18"):
    c = rd(f"{m}_coefs.csv"); st = rd(f"{m}_scores.csv"); keep = st[st.starts >= MIN].driver
    wet[m] = c[c.driver.isin(keep)].sort_values("wet_r", ascending=False).head(14)[["name", "wet_r", "wet_r_lo", "wet_r_hi"]].to_dict("records")
    if m == "B80": wet["CHG"] = c[c.driver.isin(keep)].sort_values("chg_r", ascending=False).head(14)[["name", "chg_r", "chg_r_lo", "chg_r_hi"]].to_dict("records"); bt = c[c.name == "Jenson Button"].iloc[0]; D["button"] = f"Button scores {bt.wet_r:+.2f} in the wet ({bt.wet_r_lo:+.2f} to {bt.wet_r_hi:+.2f}) and {bt.chg_r:+.2f} in changeable races ({bt.chg_r_lo:+.2f} to {bt.chg_r_hi:+.2f}). That's the direction you suggested, but the ranges overlap heavily and this score recovers poorly in tests, so it doesn't settle it."
D["wet"] = wet
o = pd.read_csv(R / "inputs" / "team_orders.csv"); lab = dict(swap="Swap reversed", hold="Held driver's result removed", neutral="Neither result counts")
tid = lambda x: "–" if pd.isna(x) else nm.get(x, str(x).replace("-", " ").title())
D["orders"] = [[int(r.season), r.gp.replace("-", " ").title() + (" sprint" if r.session == "sprint" else ""), lab[r.type], f"{tid(r.gave_way)} / {tid(r.benefited)}"] for _, r in o.iterrows()]
hold = []
for m in ("A18", "B18", "B80", "C50"):
    h = rd(f"{m}_holdout.csv"); h = h[h.era == "all"]
    for ses in ("qualifying", "race"):
        x = h[h.session == ses]; g = lambda k: x[x.model == k].accuracy
        gr = g("Better grid")
        hold.append([m, ses.title(), int(x.n.iloc[0]), f"{g(m).iloc[0]*100:.1f}%", f"{g('Head-to-head so far').iloc[0]*100:.1f}%", f"{gr.iloc[0]*100:.1f}%" if len(gr) else "–"])
D["hold"] = hold
names = dict(skill_q="Qualifying skill", skill_r="Race skill", wet_q="Wet qualifying", wet_r="Wet race", ad_t="Adapting to teams", ad_r="Adapting to rules", chg_r="Changeable race", skill_q_1950s="Qualifying skill, drivers starting by 1960", skill_r_1950s="Race skill, drivers starting by 1960")
D["syn"] = [[m, names[r.coef], f"{r.spearman:.2f}", f"{r.coverage_90*100:.0f}%"] for m in ("A18", "B18", "B80", "C50") for _, r in rd(f"{m}_synth.csv").iterrows()]
D["sens"] = [[r.change, r.model, f"{r.peak_corr:.2f}", f"{r.career_corr:.2f}"] for _, r in pd.concat([rd("sensitivity_B80.csv"), rd("sensitivity.csv")]).iterrows()]

# remove-a-teammate reruns
SH = {"felipe-massa": 77, "valtteri-bottas": 100, "rubens-barrichello": 104, "sergio-perez": 90, "alain-prost": 32, "gerhard-berger": 48, "nico-rosberg": 78, "trevor-taylor": 19, "francois-cevert": 46, "clay-regazzoni": 57, "mark-webber": 94, "karl-kling": 11, "riccardo-patrese": 46, "richie-ginther": 29, "lance-stroll": 89}
tm = pd.read_csv(O / "gap" / "tm_summary.csv"); nmc = c50.set_index("driver").name.to_dict()
rk_ = lambda v: "–" if v < 0 else str(int(v))
D["tm"] = []
for r_, g in tm.groupby("run", sort=False):
    a_, b_ = g[g.role == "removed-from"].iloc[0], g[g.role == "teammate"].iloc[0]
    tp_ = "no peak after" if pd.isna(b_.peak_t) else f"{rk_(b_.rank_b)} → {rk_(b_.rank_t)}"
    D["tm"].append([nmc.get(a_.driver, a_.driver), nmc.get(b_.driver, b_.driver), SH[b_.driver], f"{a_.peak_b:.2f} → {a_.peak_t:.2f}", f"{a_.d_peak:+.2f}", f"{rk_(a_.rank_b)} → {rk_(a_.rank_t)}", f"{rk_(a_.crank_b)} → {rk_(a_.crank_t)}", tp_ if b_.rank_b > 0 or not pd.isna(b_.peak_t) else "too few starts"])
D["tm"].sort(key=lambda x: -abs(float(x[4])))
def tier_rows(f, fmt):
    t = pd.read_csv(R / "outputs_v2" / f); out = []
    for _, r in t.iterrows():
        out.append([int(r.tier), r["name"], r.span.replace("-", "–"), fmt(r), f"{r.med_rank:.0f}", f"{r.lo:.0f}–{r.hi:.0f}"])
    return out
D["tiers_peak"] = tier_rows("tiers_peak.csv", lambda r: f"{r.value:.2f}")
# Early drivers whose peak may be inflated because the model rates their car below what it achieved
FLAG = {"Juan Manuel Fangio": "▲", "Graham Hill": "▲", "Jim Clark": "△", "Alberto Ascari": "△", "Nino Farina": "△", "Jack Brabham": "△"}
for row in D["tiers_peak"]: row[1] = row[1] + (" " + FLAG[row[1]] if row[1] in FLAG else "")
cvr = pd.read_csv(R / "outputs_v2" / "gap" / "car_vs_real_1950.csv"); c50s = pd.read_csv(R / "outputs_v2" / "C50_seasons.csv")
c50 = pd.read_csv(R / "outputs_v2" / "C50_scores.csv").set_index("name")
WHY = {"Juan Manuel Fangio": ("Likely inflated", "The 1954 Mercedes is rated only the 3rd-best car, yet it won 4 of the 6 races it entered. In the fake-dominance test Fangio was flattered by about 0.1."),
       "Graham Hill": ("Likely inflated", "The BRM is rated 6th and 4th-best in 1964–65, but it scored the most points both years. Removing Ginther also improved Hill's peak by 0.48, which fits a strong car being credited to him."),
       "Jim Clark": ("Possibly inflated", "The 1966–67 Lotus is rated below its real results (7th vs 5th, 3rd vs 2nd). But in the fake-dominance test Clark moved the other way, so the evidence is mixed."),
       "Alberto Ascari": ("Possibly inflated", "Ferrari won 14 of 15 races in 1952–53. The model does rate it the best car, but in tests it gives a car only about 70–75% of a real dominance, and the rest leaks to the driver."),
       "Nino Farina": ("Possibly inflated", "The Alfa Romeo won all 6 races in 1950. It's the same leak as Ascari's."),
       "Jack Brabham": ("Possibly inflated", "The 1966 Brabham is rated 2nd-best but won the championship."),
       "Jochen Rindt": ("Not flagged", "His cars are rated at or above their results (Lotus 1969: 1st vs 4th)."),
       "John Surtees": ("Not flagged", "His cars are rated at or above their results (Ferrari 1965: 2nd vs 4th)."),
       "Stirling Moss": ("Not flagged", "His car ratings are close to their results.")}
D["early_cars"] = []
for nm, (w, why) in WHY.items():
    r = c50.loc[nm]; a, b = map(int, r.peak_span.split("-")); dv = r.driver
    ss = c50s[(c50s.driver == dv) & c50s.season.between(a, b)]
    parts = []
    for _, x in ss.iterrows():
        k = cvr[(cvr.team == x.team) & (cvr.season == x.season)]
        if len(k): k = k.iloc[0]; parts.append(f"{x.season} {x.team.split('~')[0].replace('-', ' ').title().replace('Brm', 'BRM')}: {int(k.mrank)} / {int(k.prank)}")
    D["early_cars"].append([nm, f"{a}–{b}", f"{r.peak:.2f}", "; ".join(parts), w, why])
D["tiers_comb"] = tier_rows("tiers_combined.csv", lambda r: f"{r.peak:.2f} / {r.career:.1f}")
D["tiers_car"] = tier_rows("tiers_career.csv", lambda r: f"{r.value:.1f}")
D.update(json.loads((R / "report" / "text_v2.json").read_text(encoding="utf8")))
page = (R / "report" / "template.html").read_text(encoding="utf8")
css = page[:page.index("</style>") + 8].replace("F1 Driver Model Pilot", "F1 Driver Model")
css = css.replace("</style>", "select{font:400 15px var(--body);color:var(--fg);background:var(--panel);border:1px solid var(--rule);border-radius:4px;padding:5px 8px;max-width:100%}.ctl label{color:var(--fg2);font-size:14px}</style>")
js = (R / "report" / "app_v2.js").read_text(encoding="utf8")
def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)): return None if np.isnan(o) else round(float(o), 3)
    if isinstance(o, np.integer): return int(o)
    return o
html = css + (R / "report" / "body_v2.html").read_text(encoding="utf8") + "<script>const D=" + json.dumps(clean(D), ensure_ascii=False) + ";\n" + js + "</script>"
(R / "outputs" / "f1-model-walkthrough.html").write_text(html, encoding="utf8"); print("ok", len(html))
