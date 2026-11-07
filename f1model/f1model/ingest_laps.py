"""Pipeline A input: every lap from every qualifying session and race, reduced to one
adjusted pace number per driver per session (stage 1). Stage 2 lives in models.py."""
import json, sys
import numpy as np, pandas as pd
from .config import *

WET = {"INTERMEDIATE", "WET"}
FIELDS = ["time", "lap", "compound", "life", "pos", "status", "sesT", "pin", "pout", "del", "qs", "dNum", "lSD", "fresh"]

def _read(path):
    d = json.loads(path.read_text())
    df = pd.DataFrame({k: d.get(k, [None] * len(d["time"])) for k in FIELDS}).astype(object).where(lambda x: x != "None", np.nan)
    df["code"] = path.parent.name
    return df

def load_session(gp_dir, session):
    files = sorted((gp_dir / session).glob("*/laptimes.json"))
    if not files:
        return None
    df = pd.concat([_read(f) for f in files], ignore_index=True)
    for c in ["time", "lap", "life", "pos", "sesT"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["number"] = pd.to_numeric(df["dNum"], errors="coerce")
    df["clean_status"] = df["status"].astype(str).eq("1")
    df["is_in_out"] = df["pin"].notna() | df["pout"].notna()
    df["is_deleted"] = df["del"].astype(str).eq("True")
    df["wet_tyre"] = df["compound"].isin(WET)
    return df

def _fe(s):
    return pd.get_dummies(s, dtype=float)

def _fit_driver_effects(y, drivers, X_other):
    """OLS of y on driver + nuisance effects; returns session-centred driver effects and residuals."""
    D = _fe(drivers)
    X = np.hstack([D.values, X_other])
    y = y - y.mean()   # keeps the minimum-norm solution stable if the design is rank-deficient
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    eff = pd.Series(beta[: D.shape[1]], index=D.columns)
    return eff - eff.mean(), resid

def quali_stage1(df):
    """Adjusted average of ALL push laps, so one hero lap cannot define a session."""
    t = df[df.time.notna() & ~df.is_in_out & df.clean_status & ~df.is_deleted].copy()
    if t.empty:
        return None
    wet_share = df[df.time.notna()].wet_tyre.mean()
    wet = wet_share >= WET_LAP_SHARE
    t["seg"] = t.qs.fillna("Q")
    best = t.groupby(["code", "seg"]).time.transform("min")
    fastest = t.groupby("seg").time.transform("min")
    t = t[(t.time <= best * (PUSH_WINDOW_WET if wet else PUSH_WINDOW_DRY))
          & (t.time <= fastest * (SEGMENT_WINDOW_WET if wet else SEGMENT_WINDOW_DRY))].copy()
    if t.empty:
        return None
    t["y"] = 100 * np.log(t.time)
    # Only comparable runs: fresh-ish tyres and, in the dry, each segment's main compound.
    t = t[t.life.fillna(2) <= 4].copy()
    if wet:
        # On a changing track, later laps are faster for everyone, so an average would reward
        # timing, not driving: use each driver's best lap per segment instead.
        t = t.loc[t.groupby(["code", "seg"]).time.idxmin()].copy()
    else:
        main = t.groupby("seg").compound.agg(lambda c: c.mode().iat[0] if c.notna().any() else None)
        t = t[t.compound.isna() | (t.compound == t.seg.map(main))].copy()
    t["y"] = 100 * np.log(t.time)
    # Alternate robust driver and segment effects (segment effects absorb track evolution).
    seg = pd.Series(0.0, index=t.seg.unique()); eff = pd.Series(0.0, index=t.code.unique())
    for _ in range(15):
        seg = (t.y - t.code.map(eff)).groupby(t.seg).median()
        eff = (t.y - t.seg.map(seg)).groupby(t.code).mean()
    eff = eff - eff.mean()
    resid = t.y - t.seg.map(seg) - t.code.map(eff)
    resid = resid - resid.mean()
    t["r"] = resid
    g = t.groupby("code")
    return pd.DataFrame(dict(gap=eff, n=g.size(), lap_sd=g.r.std(), wet=wet, wet_share=wet_share,
                             hero=g.r.min(), number=g.number.first()))

def race_stage1(df, traffic_gap=TRAFFIC_GAP_S):
    """Fuel, track evolution and conditions absorbed by lap effects; tyres by compound × age."""
    d = df[df.time.notna()]
    wet_share = d.wet_tyre.mean()
    d = d.sort_values(["lap", "sesT"]).copy()
    d["gap_ahead"] = d.groupby("lap").sesT.diff()
    t = d[(d.lap > 1) & ~d.is_in_out & d.clean_status & ~d.is_deleted & d.compound.notna()].copy()
    if traffic_gap:
        t = t[~(t.gap_ahead < traffic_gap)]
    med = t.groupby("code").time.transform("median")
    t = t[t.time <= med * RACE_LAP_WINDOW]
    keep = t.groupby("code").time.transform("size") >= MIN_RACE_LAPS
    t = t[keep].copy()
    if t.code.nunique() < 4:
        return None
    t["y"] = 100 * np.log(t.time)
    Lp = _fe(t.lap).iloc[:, 1:].values
    C = _fe(t.compound); A = C.values * (t.life.fillna(t.life.median()).values[:, None] / 20)
    eff, resid = _fit_driver_effects(t.y.values, t.code, np.hstack([Lp, C.iloc[:, 1:].values, A]))
    t["r"] = resid
    g = t.groupby("code")
    return pd.DataFrame(dict(gap=eff, n=g.size(), lap_sd=g.r.std(), wet=wet_share >= WET_LAP_SHARE,
                             wet_share=wet_share, number=g.number.first()))

def build_laps(traffic_gap=TRAFFIC_GAP_S, tag=""):
    res = pd.read_pickle(PROCESSED / "results.pkl")
    races = res.drop_duplicates("race_id")[["season", "race_id", "date", "quali_date", "round", "gp"]]
    q_out, r_out, unmatched = [], [], []
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        root = RAW / f"laps{season}"
        for gp in sorted(p for p in root.iterdir() if p.is_dir() and (p / "Race").exists()):
            rf = load_session(gp, "Race")
            if rf is None or rf.lSD.dropna().empty:
                unmatched.append((season, gp.name, "no race laps")); continue
            day = pd.to_datetime(rf.lSD.dropna().iloc[0]).normalize()
            sr = races[races.season == season]
            delta = (pd.to_datetime(sr.date) - day).abs()
            if delta.min() > pd.Timedelta(days=1):
                unmatched.append((season, gp.name, "no date match")); continue
            rid = sr.loc[delta.idxmin(), "race_id"]
            s = race_stage1(rf, traffic_gap)
            if s is not None:
                r_out.append(s.reset_index(names="code").assign(race_id=rid))
            qf = load_session(gp, "Qualifying")
            if qf is not None:
                s = quali_stage1(qf)
                if s is not None:
                    q_out.append(s.reset_index(names="code").assign(race_id=rid))
    lookup = res[["race_id", "number", "driver", "team", "season"]]
    out = {}
    for name, frames in [("quali", q_out), ("race", r_out)]:
        f = pd.concat(frames, ignore_index=True).merge(lookup, on=["race_id", "number"], how="left")
        out[name] = f
        print(name, "rows", len(f), "unmatched driver rows", f.driver.isna().sum(), f[f.driver.isna()].code.unique()[:10])
        f.dropna(subset=["driver"]).to_pickle(PROCESSED / f"{name}_pace{tag}.pkl")
    print("sessions not matched:", unmatched)
    return out

if __name__ == "__main__":
    build_laps()
