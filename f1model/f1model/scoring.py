"""From posterior draws to rankings. Run separately for each pipeline."""
import numpy as np, pandas as pd
from scipy.special import expit
from .config import *
from .simulate import sim_results, sim_laps

def _thin(s, draws, rng):
    n = len(s["skill_q"]); ix = rng.choice(n, size=min(draws, n), replace=False)
    return {k: np.asarray(v)[ix] for k, v in s.items()}

def _age(s, age_c, laps):
    f = 0.3 if laps else 1.0
    return f * (s["age1"][:, None] * age_c + s["age2"][:, None] * age_c ** 2)

def _entrants(s, kind, d_idx, dev_q, dev_r, age_term, lap_sd=None):
    e = dict(q=s["skill_q"][:, d_idx] + dev_q + age_term, r=s["skill_r"][:, d_idx] + dev_r + age_term,
             wq=s["wet_q"][:, d_idx], wr=s["wet_r"][:, d_idx],
             g=np.repeat(s["grid_era"].mean(1)[:, None], len(d_idx), 1),
             perr=expit(s["err_a"][:, None] + s["err_d"][:, d_idx]),
             perr_w=expit(s["err_a"][:, None] + s["err_wet"][:, None] + s["err_d"][:, d_idx]))
    e["fit_sd"] = np.zeros_like(e["q"]); e["fit_r"] = np.ones_like(e["q"])
    if kind == "A":
        e.update(craft=s["craft"][:, d_idx], cw=s["craft_wet"][:, d_idx], b=s["b_pace"],
                 sig_q=s["sig_q"], sig_r=s["sig_r"], lap_sd=lap_sd[d_idx])
    return e

def _dev_scales(s, kind):
    base = s["dev_scale_laps"][:, None] if kind == "A" else 1.0
    return base, s["dev_scale_r"][:, None]

def main_scores(s, ds, kind, lap_sd=None, draws=200, weekends=100, seed=1, eligible=None):
    """Age-neutral, identical cars. `main` samples each driver's car fit from their own spread;
    `best_fit` and `poor_fit` hold it at the upper/lower quartile (z = ±0.674)."""
    rng = np.random.default_rng(seed)
    s = _thin(s, draws, rng)
    idx = np.array(eligible); D = len(s["skill_q"]); sim = sim_laps if kind == "A" else sim_results
    base, sr = _dev_scales(s, kind)
    sig = s["sigma_dev"][:, idx] * base
    out = {}
    for name, z in [("main", None), ("best_fit", 0.674), ("poor_fit", -0.674)]:
        dq = sig * (z or 0.0)
        e = _entrants(s, kind, idx, dq, dq * sr, 0.0 * dq, lap_sd)
        if z is None:
            e["fit_sd"], e["fit_r"] = sig, np.broadcast_to(sr, sig.shape)
        out[name] = sim(e, ds.wet_pairs, rng, weekends)
    return out

def season_scores(s, ds, kind, lap_sd=None, draws=150, weekends=30, seed=2, min_races=8):
    """Each driver-season, with its real age and deviation, in an all-time identical-car field."""
    rng = np.random.default_rng(seed); s = _thin(s, draws, rng)
    r = ds.res[ds.res.started]
    g = r.groupby("dts").agg(driver=("driver", "first"), season=("season", "first"), races=("race_id", "nunique"),
                             age_c=("age_c", "mean"), dt=("dt", "first"), d=("d", "first")).reset_index()
    g = g[g.races >= min_races].reset_index(drop=True)
    d_idx, dt = g.d.values, g.dt.values
    e = _entrants(s, kind, d_idx, s["dev_q"][:, dt], s["dev_r"][:, dt], _age(s, g.age_c.values, kind == "A"), lap_sd)
    sc = (sim_laps if kind == "A" else sim_results)(e, ds.wet_pairs, rng, weekends)
    return g, sc

def peak_and_career(g, sc):
    """Peak = best run of 3 consecutive seasons (chosen on the mean, so no winner's-curse from noise);
    career value = races-weighted positions gained against an average (10.5) driver, in season units."""
    rows = []
    mean = sc.mean(0)
    for drv, h in g.groupby("driver"):
        by = h.groupby("season").apply(lambda x: np.average(sc[:, x.index], axis=1, weights=x.races), include_groups=False)
        seasons = sorted(by.index); arr = np.vstack([by[y] for y in seasons])
        races = h.groupby("season").races.sum().loc[seasons].values
        peak = None
        if len(seasons) >= 3:
            wins = [(arr[i:i + 3].mean(), i) for i in range(len(seasons) - 2) if seasons[i + 2] - seasons[i] == 2]
            if wins:
                _, i = min(wins); peak = arr[i:i + 3].mean(0); span = f"{seasons[i]}-{seasons[i + 2]}"
        career = ((10.5 - arr) * (races[:, None] / 22)).sum(0)
        rows.append(dict(driver=drv, peak=None if peak is None else peak.mean(),
                         peak_lo=None if peak is None else np.quantile(peak, .05),
                         peak_hi=None if peak is None else np.quantile(peak, .95),
                         peak_span=None if peak is None else span,
                         career=career.mean(), career_lo=np.quantile(career, .05), career_hi=np.quantile(career, .95),
                         seasons=len(seasons)))
    return pd.DataFrame(rows)

def adaptation_events(s, ds):
    """Average deviation (in standard-deviation units of the whole field) in seasons with a test of
    adaptation, against the driver's other seasons. Small samples: read with `n`."""
    r = ds.res[ds.res.started].groupby("dts").agg(driver=("driver", "first"), team=("team", "first"),
                                                 season=("season", "first"), dt=("dt", "first")).reset_index()
    z = np.asarray(s["dev_q"]).mean(0); r["dev"] = z[r.dt.values] / z.std()
    r = r.sort_values(["driver", "season"])
    r["new_team"] = r.groupby("driver").team.shift().ne(r.team) & r.groupby("driver").team.shift().notna()
    r["reg"] = r.season.isin(REG_CHANGE_SEASONS) & r.groupby("driver").season.shift().notna()
    rows = []
    for drv, h in r.groupby("driver"):
        o = h.dev[~h.new_team & ~h.reg]
        rows.append(dict(driver=drv, new_team_n=int(h.new_team.sum()), new_team_dev=h.dev[h.new_team].mean(),
                         reg_n=int(h.reg.sum()), reg_dev=h.dev[h.reg].mean(), other_dev=o.mean()))
    return pd.DataFrame(rows)

def coef_table(s, ds, kind, eligible, sc, lap_sd=None):
    rows = []
    q = lambda a, p: np.quantile(a, p, axis=0)
    for pos, i in enumerate(eligible):
        drv = ds.drivers[i]
        row = dict(driver=drv, name=ds.names[drv], races=int(ds.res[ds.res.driver == drv].started.sum()),
                   score=sc["main"][:, pos].mean(), score_lo=q(sc["main"][:, pos], .05), score_hi=q(sc["main"][:, pos], .95),
                   best_fit=sc["best_fit"][:, pos].mean(), poor_fit=sc["poor_fit"][:, pos].mean(),
                   adapt_sigma=np.asarray(s["sigma_dev"])[:, i].mean())
        for k in ["skill_q", "skill_r", "wet_q", "wet_r"]:
            row[k] = np.asarray(s[k])[:, i].mean()
        row["err_pct_per_race"] = 100 * expit(np.asarray(s["err_a"]) + np.asarray(s["err_d"])[:, i]).mean()
        if kind == "A":
            row["racecraft"] = np.asarray(s["craft"])[:, i].mean()
            row["lap_spread"] = lap_sd[i]
        rows.append(row)
    t = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    t.insert(0, "rank", t.index + 1)
    return t

def prob_rank_top(sc_main, eligible, ds, top=1):
    r = sc_main.argsort(1).argsort(1) < top
    return {ds.drivers[e]: r[:, j].mean() for j, e in enumerate(eligible)}
