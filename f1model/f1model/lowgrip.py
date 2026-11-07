"""Your wet method: find laps that are slow because grip is low, not just because rain fell.

Each weekend's reference is the fastest clean lap on dry tyres in any session. A race lap counts as
low-grip when the field's median time for that lap is far off that reference AND the field is on
wet or intermediate tyres. Drivers are then compared only on those laps.
"""
import numpy as np, pandas as pd
from .config import *
from .ingest_laps import load_session, WET

SESS = ["Practice 1", "Practice 2", "Practice 3", "Sprint Qualifying", "Sprint Shootout", "Qualifying", "Sprint", "Sprint Race", "Race"]
RACE_LOW = 1.14    # field median on this lap vs weekend dry reference (calibrated below)
QUALI_LOW = 1.05

def dry_reference(frames):
    best = np.inf
    for df in frames.values():
        t = df[df.time.notna() & df.clean_status & ~df.is_in_out & ~df.is_deleted & df.compound.notna() & ~df.compound.isin(WET)]
        if len(t):
            best = min(best, t.time.min())
    return best if np.isfinite(best) else None

def race_laps(df):
    d = df[df.time.notna() & (df.lap > 1) & df.clean_status & ~df.is_in_out & ~df.is_deleted].copy()
    d["field"] = d.groupby("lap").time.transform("median")
    d["wet_share"] = d.groupby("lap").wet_tyre.transform("mean")
    return d

def low_race_gaps(d, ref, race_low=RACE_LOW):
    """Driver gaps on low-grip laps only, with lap effects absorbing the drying track."""
    low = d[(d.field >= ref * race_low) & (d.wet_share >= .5) & d.wet_tyre].copy()
    low = low[low.time <= low.groupby("lap").time.transform("median") * 1.12]
    low = low[low.groupby("code").time.transform("size") >= 4]
    if low.code.nunique() < 4:
        return None, len(low)
    low["y"] = 100 * np.log(low.time)
    low["y"] -= low.groupby("lap").y.transform("mean")
    eff = pd.Series(0.0, index=low.code.unique())
    for _ in range(10):
        lap = (low.y - low.code.map(eff)).groupby(low.lap).mean()
        eff = (low.y - low.lap.map(lap)).groupby(low.code).mean()
    eff -= eff.mean()
    res = low.y - low.lap.map(lap) - low.code.map(eff)
    g = res.groupby(low.code)
    return pd.DataFrame(dict(gap=eff, n=g.size(), lap_sd=g.std(), number=low.groupby("code").number.first())), low.lap.nunique()

def low_quali_gaps(df, ref, quali_low=QUALI_LOW):
    t = df[df.time.notna() & df.clean_status & ~df.is_in_out & ~df.is_deleted & df.wet_tyre].copy()
    if t.empty:
        return None
    t["seg"] = t.qs.fillna("Q")
    fast = t.groupby("seg").time.transform("min")
    t = t[fast >= ref * quali_low]
    if t.code.nunique() < 4:
        return None
    t = t[t.time <= t.groupby("seg").time.transform("min") * 1.15]
    t = t.loc[t.groupby(["code", "seg"]).time.idxmin()].copy()
    t["y"] = 100 * np.log(t.time)
    eff = pd.Series(0.0, index=t.code.unique())
    for _ in range(10):
        s = (t.y - t.code.map(eff)).groupby(t.seg).median()
        eff = (t.y - t.seg.map(s)).groupby(t.code).mean()
    eff -= eff.mean()
    r = t.y - t.seg.map(s) - t.code.map(eff)
    g = r.groupby(t.code)
    return pd.DataFrame(dict(gap=eff, n=g.size(), lap_sd=g.std(), number=t.groupby("code").number.first()))
