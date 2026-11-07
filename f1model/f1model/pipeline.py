"""Data preparation + fitting for each pipeline. Nothing here mixes A and B."""
import numpy as np, pandas as pd, jax.numpy as jnp
from .config import *
from .design import *
from . import models

def load(collisions_as_errors=False, tag=""):
    res = add_reasons(pd.read_pickle(PROCESSED / "results.pkl"), collisions_as_errors)
    qp, rp = pd.read_pickle(PROCESSED / f"quali_pace{tag}.pkl"), pd.read_pickle(PROCESSED / f"race_pace{tag}.pkl")
    # F1DB 'NC' and retirements are out of the ranking; DNS/DNQ never entered it
    return res, qp, rp, wet_labels(qp, rp)

def _basic(df, wet_col):
    v = lambda c: jnp.asarray(df[c].values)
    return dict(d=v("d"), t=v("t"), dt=v("dt"), c=v("c"), era=v("era"), prog=jnp.asarray(df.prog.values),
                age=jnp.asarray(df.age_c.values), wet=jnp.asarray(df[wet_col].values.astype(float)))

def _ranked(df, order, wet_col):
    df = df.reset_index(drop=True)
    idx, mask = rank_arrays(df, order)
    a = {k: x[idx] for k, x in _basic(df, wet_col).items()}
    a["grid"] = jnp.asarray(grid_norm(df.grid.values))[idx] if "grid" in df else None
    a["mask"] = jnp.asarray(mask)
    return a, df, idx

def _errors(tr):
    e = tr[tr.started & ~tr.no_race]
    return dict(d=jnp.asarray(e.d.values), wet=jnp.asarray(e.wet_r.values.astype(float)), y=jnp.asarray(e.error.values.astype(float)))

def data_b(ds):
    tr = ds.train
    q = tr[tr.quali_pos.notna()]
    r = tr[tr.finished & ~tr.no_race]
    return (_ranked(q, "quali_pos", "wet_q")[0], _ranked(r, "finish", "wet_r")[0], _errors(tr), ds.n, jnp.asarray(ds.dts_driver))

def _pace(tr, pace, wet_col):
    p = pace.drop(columns=["team", "season", "wet", "wet_share", "number"]).merge(tr, on=["race_id", "driver"])
    med = p.lap_sd[p.lap_sd > 0].median()
    p["se"] = np.clip(p.lap_sd.fillna(med).clip(lower=med / 2, upper=3 * med), 0, None) / np.sqrt(p.n)
    p["adv"] = -p.gap
    return p

def data_a(ds, qp, rp):
    tr = ds.train
    Q = _pace(tr, qp, "wet_q"); R = _pace(tr, rp, "wet_r")
    def flat(p, w):
        b = _basic(p, w); b["adv"] = jnp.asarray(p.adv.values); b["se"] = jnp.asarray(p.se.values); return b
    fin = R[R.finished & ~R.no_race].reset_index(drop=True)
    craft, df, idx = _ranked(fin, "finish", "wet_r")
    craft["adv"] = jnp.asarray(fin.adv.values)[idx]
    return (flat(Q, "wet_q"), flat(R, "wet_r"), craft, _errors(tr), ds.n, jnp.asarray(ds.dts_driver))

def fit_b(ds, **kw):
    return models.fit(models.model_results, *data_b(ds), **kw)

def fit_a(ds, qp, rp, **kw):
    return models.fit(models.model_laps, *data_a(ds, qp, rp), **kw)

def lap_spread(qp, ds):
    """Each driver's lap-to-lap spread on dry push laps, shrunk toward the field for small samples."""
    d = qp[(~qp.wet) & qp.lap_sd.notna() & (qp.n >= 3)]
    g = d.groupby("driver").lap_sd.agg(["median", "size"])
    overall = d.lap_sd.median()
    s = (g["median"] * g["size"] + overall * 10) / (g["size"] + 10)
    return np.array([s.get(x, overall) for x in ds.drivers])
