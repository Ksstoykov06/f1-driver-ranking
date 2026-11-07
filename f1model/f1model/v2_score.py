"""Peak, career and adaptability scores, kept separate. All use identical cars in a shared field.
Long-term team fit is left out by default, because it can absorb car strength; use_fit=True adds it back."""
import numpy as np, pandas as pd
from scipy.special import expit
from .simulate import sim_results, sim_laps

MIN_SEASON_STARTS = 8
MIN_CAREER_STARTS = 40

def _flat(s):
    return {k: v.reshape((-1,) + v.shape[2:]) for k, v in s.items()}

def _thin(s, n, rng):
    N = len(s["tau_form"]); i = rng.choice(N, min(n, N), replace=False)
    return {k: v[i] for k, v in s.items()}

def _ent(s, kind, d, q, r, spread):
    a = s["err_dec"].mean(1)
    e = dict(q=q, r=r, wq=s["wet_q"][:, d], wr=s["wet_r"][:, d], cr=s["chg_r"][:, d], g=np.repeat(s["grid_era"].mean(1)[:, None], len(d), 1),
             perr=expit(a[:, None] + s["err_d"][:, d]), perr_w=expit(a[:, None] + s["err_wet"][:, None] + s["err_d"][:, d]))
    if "col_d" in s:  # collision tendency: a further way to lose the race, on top of solo errors
        pc = expit((s["col_dec"].mean(1) + s["col_g2"] * 1 / 12)[:, None] + s["col_d"][:, d])
        e["perr"] = 1 - (1 - e["perr"]) * (1 - pc); e["perr_w"] = 1 - (1 - e["perr_w"]) * (1 - expit((s["col_dec"].mean(1) + s["col_g2"] / 12 + s["col_wet"])[:, None] + s["col_d"][:, d]))
    e["fit_sd"] = np.zeros_like(q); e["fit_r"] = np.ones_like(q)
    if kind == "A":
        e.update(craft=s["craft"][:, d], cw=s["craft_wet"][:, d], b=s["b_pace"], sig_q=s["sig_q"], sig_r=s["sig_r"], lap_sd=spread[d])
    return e

def driver_seasons(data, min_starts=MIN_SEASON_STARTS):
    r = data.res[data.res.started & (data.res.session == "gp") & (data.res.season >= getattr(data, "report_from", 0))]
    rows = []
    n_races = r.groupby("season").race_id.nunique()
    for (d, s), g in r.groupby(["driver", "season"]):
        # short early calendars: 60% of the season if that's below the usual minimum
        if len(g) < min(min_starts, int(np.ceil(.6 * n_races[s]))):
            continue
        fit = g.groupby("dt").size()
        rows.append(dict(driver=d, season=s, races=len(g), d=g.d.iloc[0], dsi=g.dsi.iloc[0], dts=fit.index.tolist(), w=(fit / fit.sum()).tolist(), x_t=g.x_t.mean(), x_r=g.x_r.mean(), team=g.team.mode().iat[0]))
    return pd.DataFrame(rows)

def season_scores(s, data, kind, spread=None, draws=120, weekends=24, seed=3, use_fit=False):
    rng = np.random.default_rng(seed); s = _thin(_flat(s), draws, rng)
    g = driver_seasons(data); d = g.d.values
    base = s["form"][:, g.dsi.values]
    fit = np.stack([sum(w * s["fit"][:, i] for i, w in zip(a, b)) for a, b in zip(g.dts, g.w)], 1)
    # Only differences from the average driver count. The field-wide average after a change can't be
    # separated from selection effects, for example promoted drivers joining a team on good form.
    ad = (s["ad_t"][:, d] - s["ad_t_mu"][:, None]) * g.x_t.values + (s["ad_r"][:, d] - s["ad_r_mu"][:, None]) * g.x_r.values
    common = base + (fit if use_fit else 0.0) + ad
    e = _ent(s, kind, d, s["skill_q"][:, d] + common, s["skill_r"][:, d] + s["r_scale"][:, None] * common, spread)
    sim = sim_laps if kind == "A" else sim_results
    return g, sim(e, data.wet_pairs, rng, weekends)

def peak_career(g, sc, data):
    """Peak: best three consecutive seasons. Career: total places gained on an average driver,
    with 22 starts counting as one season, plus the career average for comparison."""
    rows = []
    for drv, h in g.groupby("driver"):
        h = h.sort_values("season"); a = sc[:, h.index]; ss = h.season.values
        row = dict(driver=drv, name=data.names[drv], seasons=len(h), starts=int(h.races.sum()), first=int(ss[0]), last=int(ss[-1]))
        w = [(a[:, i:i + 3].mean(), i) for i in range(len(ss) - 2) if ss[i + 2] - ss[i] == 2]
        if w:
            _, i = min(w); p = a[:, i:i + 3].mean(1)
            row.update(peak=p.mean(), peak_lo=np.quantile(p, .05), peak_hi=np.quantile(p, .95), peak_span=f"{ss[i]}-{ss[i + 2]}")
        c = ((10.5 - a) * (h.races.values / 22)).sum(1)
        wavg = np.average(a, axis=1, weights=h.races.values)
        row.update(career=c.mean(), career_lo=np.quantile(c, .05), career_hi=np.quantile(c, .95), career_avg=wavg.mean())
        rows.append(row)
    return pd.DataFrame(rows)

def adaptability(s, data, kind, spread=None, eligible=None, draws=80, weekends=30, seed=5):
    """Places lost in the first race after a change, compared with how the average driver copes."""
    rng = np.random.default_rng(seed); s = _thin(_flat(s), draws, rng)
    sim = sim_laps if kind == "A" else sim_results
    idx = np.array(eligible); n = len(idx); dd = np.concatenate([idx, idx, idx])
    r = s["r_scale"][:, None]
    shift = np.concatenate([np.zeros((len(s["tau_form"]), n)), s["ad_t"][:, idx] - s["ad_t_mu"][:, None], s["ad_r"][:, idx] - s["ad_r_mu"][:, None]], 1)
    e = _ent(s, kind, dd, s["skill_q"][:, dd] + shift, s["skill_r"][:, dd] + r * shift, spread)
    pos = sim(e, data.wet_pairs, rng, weekends)
    t, g = pos[:, n:2 * n] - pos[:, :n], pos[:, 2 * n:] - pos[:, :n]
    x = data.res[data.res.session == "gp"]
    ev_t = x[x.x_team_n == 0].groupby("driver").size(); ev_r = x[x.x_reg_n == 0].groupby("driver").size()
    return pd.DataFrame([dict(driver=data.drivers[i], name=data.names[data.drivers[i]], team_moves=int(ev_t.get(data.drivers[i], 0)), rule_changes=int(ev_r.get(data.drivers[i], 0)),
                              team_loss=t[:, j].mean(), team_lo=np.quantile(t[:, j], .05), team_hi=np.quantile(t[:, j], .95),
                              rule_loss=g[:, j].mean(), rule_lo=np.quantile(g[:, j], .05), rule_hi=np.quantile(g[:, j], .95)) for j, i in enumerate(idx)])

def coefs(s, data):
    f = _flat(s); out = []
    for i, d in enumerate(data.drivers):
        row = dict(driver=d, name=data.names[d])
        for k in ["skill_q", "skill_r", "wet_q", "wet_r", "chg_r"]:
            row[k], row[k + "_lo"], row[k + "_hi"] = f[k][:, i].mean(), *np.quantile(f[k][:, i], [.05, .95])
        row["err_pct"] = 100 * expit(f["err_dec"].mean(1) + f["err_d"][:, i]).mean()
        if "craft" in f:
            row["racecraft"] = f["craft"][:, i].mean()
        out.append(row)
    return pd.DataFrame(out)
