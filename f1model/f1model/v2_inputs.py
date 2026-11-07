"""Inputs for version 2: 1980+ results with sprints, low-grip lap pace, wet labels, team orders."""
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np, pandas as pd, yaml
from .config import RAW, PROCESSED, ROOT
from .ingest_laps import load_session, race_stage1, quali_stage1
from . import lowgrip as LG

FIRST, LAST = 1961, 2026
ERR = {"Accident", "Accident damage", "Spun off", "Spin", "Ignored red light"}
COLL = {"Collision", "Collision damage"}
REG_SEASONS = (1983, 1989, 1994, 1998, 2009, 2014, 2017, 2022, 2026)

def _load(p):
    return yaml.safe_load(p.read_text(encoding="utf8")) if p.exists() else None
def _num(x):
    return int(x) if isinstance(x, int) else None

LINEAGE = {"motor-racing-developments": None, "martini-racing": None, "marlboro-team-texaco": None, "world-wide-racing": None, "stp-corporation": None, "team-rothmans-international": None,
           "tyrrell-racing-organisation": "tyrrell", "hesketh-racing": "hesketh", "williams-grand-prix-engineering": "williams", "team-surtees": "surtees", "embassy-racing": "lola"}

def customer_entrants(until=1983):
    """Before 1983 privateers often ran another maker's car, so they shouldn't count as that works team's teammates."""
    base = RAW / "f1db" / "src" / "data" / "seasons"; out = {}
    def walk(x, y, seen, en=None, c=None):
        if isinstance(x, list):
            for i in x: walk(i, y, seen, en, c)
        elif isinstance(x, dict):
            en, c = x.get("entrantId", en), x.get("constructorId", c)
            if x.get("driverId") and c and en: seen.setdefault(c, {}).setdefault(en, set()).add(x["driverId"])
            for v in x.values():
                if isinstance(v, (list, dict)): walk(v, y, seen, en, c)
    for y in range(FIRST, until):
        seen = {}; walk(_load(base / str(y) / "entrants.yml") or [], y, seen)
        for c, ents in seen.items():
            named = {e for e in ents if c.split("-")[0] in e}
            # works team: entrant carrying the maker's name, otherwise the biggest entrant (a sponsor name, e.g. Parmalat Brabham)
            works = named or {max(sorted(ents), key=lambda e: len(ents[e]))}
            for e, ds in ents.items():
                if e in LINEAGE:    # a customer car run by a team with its own history, or a works entry under another name
                    for d in ds: out[(y, c, d)] = LINEAGE[e] or c
                elif e not in works:
                    for d in ds: out[(y, c, d)] = f"{c}~{e}"
    return out

def _clean_early(df):
    """1950-60: the Indy 500 was a separate series with almost no shared drivers. In a shared car no result can be
    credited to one driver, so it isn't ranked. A driver who took over a car keeps only their own starting car's row."""
    df = df[~df.gp.str.contains("indianapolis")].copy()
    car = df.groupby(["race_id", "session", "number"]).shared.transform("any")
    df["shared_car"] = car
    df = df.sort_values("grid", na_position="last").drop_duplicates(["race_id", "session", "driver"]).copy()
    df.loc[df.shared & df.grid.isna(), "pos_text"] = "DNS"   # a relief driver who never started the race
    return df

def build_results(out="results_v2.pkl"):
    base = RAW / "f1db" / "src" / "data"
    drivers = {}
    for p in (base / "drivers").glob("*.yml"):
        d = _load(p); drivers[d["id"]] = (d["name"], d.get("abbreviation"), d.get("dateOfBirth"))
    rows = []
    for season in range(FIRST, LAST + 1):
        for rd in sorted((base / "seasons" / str(season) / "races").iterdir()):
            race, rr = _load(rd / "race.yml"), _load(rd / "race-results.yml")
            if not rr:
                continue
            gp = rd.name.split("-", 1)[1]
            q = {x["driverId"]: x for x in (_load(rd / "qualifying-results.yml") or [])}
            common = dict(season=season, round=race["round"], weekend=race["id"], gp=gp, circuit=race["circuitId"], date=str(race["date"]))
            for r in rr:
                grid = r.get("gridPosition")
                rows.append(dict(common, race_id=race["id"], session="gp", driver=r["driverId"], number=r.get("driverNumber"), team=r["constructorId"],
                                 pos_text=str(r["position"]), finish=_num(r["position"]), reason=r.get("reasonRetired"), grid=_num(grid), shared=bool(r.get("sharedCar")),
                                 quali_pos=_num(q.get(r["driverId"], {}).get("position"))))
            for r in (_load(rd / "sprint-race-results.yml") or []):
                grid = r.get("gridPosition")
                rows.append(dict(common, race_id=-race["id"], session="sprint", driver=r["driverId"], number=r.get("driverNumber"), team=r["constructorId"],
                                 pos_text=str(r["position"]), finish=_num(r["position"]), reason=r.get("reasonRetired"), grid=_num(grid), quali_pos=None))
    df = pd.DataFrame(rows)
    if FIRST < 1961: df = _clean_early(df)
    cust = customer_entrants()
    df["team"] = [cust.get((s, c, d), c) for s, c, d in zip(df.season, df.team, df.driver)]
    df["name"] = df.driver.map(lambda d: drivers[d][0]); df["code"] = df.driver.map(lambda d: drivers[d][1])
    df["dob"] = pd.to_datetime(df.driver.map(lambda d: drivers[d][2]))
    df["age"] = (pd.to_datetime(df.date) - df.dob).dt.days / 365.25
    df["started"] = ~df.pos_text.isin(["DNS", "DNQ", "DNP", "EX"])
    r = df.reason.fillna("")
    df["error"] = df.started & r.isin(ERR)
    df["collision"] = df.started & r.isin(COLL)
    df["finished"] = df.finish.notna() & (r == "")
    if "shared_car" in df: df.loc[df.shared_car, "finished"] = False
    df["no_race"] = (df.season == 2021) & (df.gp == "belgium") & (df.session == "gp")
    df = df.sort_values(["season", "round", "session"]).reset_index(drop=True)
    # how recently did this driver change team, or face a rule change?
    df["x_team_n"], df["x_reg_n"] = np.nan, np.nan
    for d, g in df[df.started & (df.session == "gp")].groupby("driver", sort=False):
        team, n_t, seas, n_r = None, np.nan, None, np.nan
        for i, row in g.iterrows():
            if team is None: pass
            elif row.team != team: n_t = 0
            elif not np.isnan(n_t): n_t += 1
            if seas is not None and row.season != seas:
                n_r = 0 if row.season in REG_SEASONS else np.nan
            elif not np.isnan(n_r): n_r += 1
            df.loc[i, ["x_team_n", "x_reg_n"]] = [n_t, n_r]
            team, seas = row.team, row.season
    df[["x_team_n", "x_reg_n"]] = df.groupby(["driver", "weekend"])[["x_team_n", "x_reg_n"]].transform("first")
    df.to_pickle(PROCESSED / out)
    return df

def build_pace(race_low=LG.RACE_LOW, quali_low=LG.QUALI_LOW, traffic=1.5, tag=""):
    res = pd.read_pickle(PROCESSED / "results_v2.pkl")
    gp_rows = res[(res.session == "gp") & (res.season >= 2018)].drop_duplicates("race_id")
    out, labels = [], []
    for season in range(2018, LAST + 1):
        root = RAW / f"laps{season}"
        for gp in sorted(p for p in root.iterdir() if p.is_dir() and (p / "Race").exists()):
            frames = {s: f for s in LG.SESS if (gp / s).exists() and (f := load_session(gp, s)) is not None}
            if "Race" not in frames or frames["Race"].lSD.dropna().empty:
                continue
            day = pd.to_datetime(frames["Race"].lSD.dropna().iloc[0]).normalize()
            sr = gp_rows[gp_rows.season == season]
            delta = (pd.to_datetime(sr.date) - day).abs()
            if delta.min() > pd.Timedelta(days=1):
                continue
            rid = sr.loc[delta.idxmin(), "race_id"]
            ref = LG.dry_reference(frames)
            fr = frames["Race"]
            add = lambda s, kind, cond, sess="gp", race=rid: out.append(s.reset_index(names="code")[["code", "gap", "n", "lap_sd", "number"]].assign(race_id=race, kind=kind, cond=cond, session=sess)) if s is not None else None
            dryr = fr[~fr.wet_tyre]
            if len(dryr) > 50:
                try: add(race_stage1(dryr, traffic), "r", "dry")
                except Exception: pass
            n_low = 0; laps_all = fr.lap.nunique()
            if ref:
                lg, n_low = LG.low_race_gaps(LG.race_laps(fr), ref, race_low)
                add(lg, "r", "low")
            sp = next((frames[s] for s in ["Sprint", "Sprint Race"] if s in frames), None)
            if sp is not None and -rid in set(res.race_id):
                try: add(race_stage1(sp[~sp.wet_tyre], traffic), "r", "dry", "sprint", -rid)
                except Exception: pass
            qf = frames.get("Qualifying"); ql = 0
            if qf is not None:
                dq = qf[~qf.wet_tyre]
                try:
                    if dq.time.notna().sum() > 20: add(quali_stage1(dq), "q", "dry")
                except Exception: pass
                if ref:
                    lq = LG.low_quali_gaps(qf, ref, quali_low)
                    add(lq, "q", "low"); ql = lq is not None
            wet_r = 1.0 if n_low / max(laps_all, 1) >= .5 else .5 if n_low / max(laps_all, 1) >= .15 else 0.0
            labels.append(dict(race_id=rid, season=season, wet_r=wet_r, low_laps=n_low, laps=laps_all, wet_q=1.0 if ql else 0.0))
    f = pd.concat(out, ignore_index=True)
    f = f.merge(res[["race_id", "number", "driver", "team", "season"]], on=["race_id", "number"], how="left")
    print(tag, "pace rows", f.groupby(["kind", "cond"]).size().to_dict(), "unmatched", f.driver.isna().sum())
    f.dropna(subset=["driver"]).to_pickle(PROCESSED / f"pace_v2{tag}.pkl")
    L = pd.DataFrame(labels); L.to_pickle(PROCESSED / f"wet_laps_v2{tag}.pkl")
    return f, L

def build_wet(tag=""):
    """Lap-based from 2018; a checked list and pole-speed detector before that."""
    res = pd.read_pickle(PROCESSED / os.environ.get("F1_RES", "results_v2.pkl"))
    L = pd.read_pickle(PROCESSED / f"wet_laps_v2{tag}.pkl").set_index("race_id")
    gp = res[res.session == "gp"].drop_duplicates("race_id")[["race_id", "season", "gp"]]
    # share of the race on inters or wets: from tyre data since 2018, from checked race reports before that
    ev = pd.read_csv(ROOT / "inputs" / "wet_evidence.csv"); ty = pd.read_csv(ROOT / "inputs" / "wet_tyre_share_2018plus.csv")
    h = {(a, b): c for a, b, c in pd.concat([ev, ty])[["season", "gp", "share_pct"]].values}
    for a, b, c, _ in pd.read_csv(ROOT / "inputs" / "wet_leadin.csv").values: h[(a, b)] = float(c)
    missing = set(h) - set(zip(gp.season, gp.gp))
    if missing: print("WET LIST UNMATCHED", sorted(missing))
    det = pd.read_pickle(PROCESSED / "speed_detector.pkl")
    det = det[(det.q_rel < -.08) & ~((det.season == 2005) & (det.q_rel < -.4))]
    qd = set(zip(det.season, det.gp))
    rows = []
    for _, r in gp.iterrows():
        s = h.get((r.season, r.gp), 0.0)
        if r.season >= 2018:
            w = L.loc[r.race_id] if r.race_id in L.index else None
            wq = 0.0 if w is None else w.wet_q
        else:
            wq = 1.0 if (r.season, r.gp) in qd else 0.0
        rows.append(dict(race_id=r.race_id, wet_r=float(s >= 50), chg_r=float(20 <= s < 50), wet_share=s, wet_q=wq))
    return pd.DataFrame(rows).set_index("race_id")

def apply_team_orders(res, on=True):
    """Only orders that changed the order between teammates are reversed; 'hold' removes the driver
    told to stay behind, because we can't know whether they'd have passed."""
    res = res.copy(); res["order_flag"] = ""
    if not on:
        return res
    t = pd.read_csv(ROOT / "inputs" / "team_orders.csv")
    for o in t.itertuples():
        m = (res.season == o.season) & (res.gp == o.gp) & (res.session == o.session)
        a, b = res.index[m & (res.driver == o.gave_way)], res.index[m & (res.driver == o.benefited)]
        if not len(a) or not len(b):
            print("ORDER UNMATCHED", o); continue
        if o.type == "swap" and res.loc[b[0], "finish"] == res.loc[b[0], "finish"] and res.loc[a[0], "finish"] == res.loc[a[0], "finish"]:
            fa, fb = res.loc[a[0], "finish"], res.loc[b[0], "finish"]
            if fb < fa:          # the beneficiary is ahead, so the order did change the result
                res.loc[a[0], "finish"], res.loc[b[0], "finish"] = fb, fa
                res.loc[[a[0], b[0]], "order_flag"] = "reversed"
        elif o.type == "neutral":     # unclear who'd have finished ahead, so neither result counts
            res.loc[[a[0], b[0]], "finished"] = False; res.loc[[a[0], b[0]], "order_flag"] = "neutral"
        elif o.type == "hold":
            res.loc[a[0], "finished"] = False; res.loc[a[0], "order_flag"] = "held"
    return res

def detect_swaps():
    """2018+: teammates swapping position on track with no pit stop nearby. A review list only."""
    res = pd.read_pickle(PROCESSED / "results_v2.pkl")
    gp_rows = res[(res.session == "gp") & (res.season >= 2018)]
    meta = gp_rows.drop_duplicates("race_id")
    out = []
    for season in range(2018, LAST + 1):
        for gp in sorted(p for p in (RAW / f"laps{season}").iterdir() if p.is_dir() and (p / "Race").exists()):
            df = load_session(gp, "Race")
            if df is None or df.lSD.dropna().empty: continue
            day = pd.to_datetime(df.lSD.dropna().iloc[0]).normalize()
            sr = meta[meta.season == season]; delta = (pd.to_datetime(sr.date) - day).abs()
            if delta.min() > pd.Timedelta(days=1): continue
            rid = sr.loc[delta.idxmin(), "race_id"]; rr = gp_rows[gp_rows.race_id == rid]
            for team, g in rr.groupby("team"):
                if len(g) != 2: continue
                n = g.number.tolist(); a = df[df.number == n[0]].set_index("lap"); b = df[df.number == n[1]].set_index("lap")
                j = a[["pos", "sesT", "is_in_out", "clean_status"]].join(b[["pos", "sesT", "is_in_out"]], lsuffix="_a", rsuffix="_b", how="inner").dropna(subset=["pos_a", "pos_b"])
                j["ahead"] = j.pos_a < j.pos_b
                j["adj"] = (j.pos_a - j.pos_b).abs() == 1
                for lap in j.index[1:]:
                    if lap < 5 or lap not in j.index or lap - 1 not in j.index: continue
                    w = j.loc[max(lap - 2, 0):lap + 1]
                    if j.loc[lap, "ahead"] != j.loc[lap - 1, "ahead"] and j.loc[lap, "adj"] and j.loc[lap - 1, "adj"] and not (w.is_in_out_a.any() or w.is_in_out_b.any()) and j.loc[lap, "clean_status"]:
                        gv, bn = (g.iloc[0], g.iloc[1]) if j.loc[lap, "pos_a"] > j.loc[lap, "pos_b"] else (g.iloc[1], g.iloc[0])
                        out.append(dict(season=season, gp=g.gp.iloc[0], lap=int(lap), team=team, passed=bn["name"], was_passed=gv["name"], final_passed=bn.finish, final_was_passed=gv.finish))
    o = pd.DataFrame(out); o.to_csv(ROOT / "inputs" / "swap_candidates_2018plus.csv", index=False)
    return o

if __name__ == "__main__":
    r = build_results(); print(r.groupby(r.season // 10 * 10).agg(rows=("driver", "size"), drivers=("driver", "nunique"), races=("race_id", "nunique")))
    print("reg exposed", (r.x_reg_n == 0).sum(), "team exposed", (r.x_team_n == 0).sum(), "quali missing among started", r[(r.session == "gp") & r.started].quali_pos.isna().mean())
    print(r[r.driver == "fernando-alonso"].groupby("season")[["x_team_n", "x_reg_n"]].min().T.to_string())
    f, L = build_pace()
    w = build_wet(); print(w.groupby(pd.Series(w.index).map(r.drop_duplicates("race_id").set_index("race_id").season // 10 * 10).values).agg(lambda x: (x > 0).sum()))
    m = r.drop_duplicates("race_id").set_index("race_id"); print([f"{m.season[i]} {m.gp[i]} {w.wet_r[i]}" for i in w.index[(w.wet_r > 0) & (m.season[w.index] >= 2018)]]); print([f"{m.season[i]} {m.gp[i]}" for i in w.index[(w.wet_q > 0) & (m.season[w.index] >= 2018)]])
    rr = apply_team_orders(r); print(rr[rr.order_flag != ""][["season", "gp", "name", "finish", "order_flag"]].to_string())
    s = detect_swaps(); print(len(s)); print(s.to_string())
