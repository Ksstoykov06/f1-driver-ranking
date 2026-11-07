import os
"""Version 2 models.

performance = driver skill + career form (smooth drift) + driver-team fit
            + adaptation (a temporary change after a team move or rule change, per driver)
            + wet adjustment (driver and car) + team-season car (with in-season drift)

Model B ranks results; Model A explains lap pace, plus racecraft. Neither uses an age curve any more:
career form lets each driver improve and decline, and teammates keep it anchored.
"""
import numpy as np, pandas as pd, jax, jax.numpy as jnp, pickle, numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median

ADAPT_TAU = 6.0
COMEBACK_MULT = 2.0
FORM_PRIOR = .12    # teammates moving together should read as the car, not as both improving

class Data:
    def __init__(self, res, wet, start=1980, end=2026, holdout=(), tau=ADAPT_TAU):
        r = res[(res.season >= start) & (res.season <= end)].copy().reset_index(drop=True)
        r["wet_q"] = r.race_id.map(wet.wet_q).fillna(0.0).astype(float)
        r["wet_r"] = r.race_id.map(wet.wet_r).fillna(0.0).astype(float)
        r["chg_r"] = r.race_id.map(wet.chg_r).fillna(0.0).astype(float)
        if os.environ.get("F1_MERGE_WET"):   # test: changeable races count as wet, no separate score
            r["wet_r"] = r[["wet_r", "chg_r"]].max(axis=1); r["chg_r"] = 0.0
        r["ts"] = r.team + "|" + r.season.astype(str)
        r["dteam"] = r.driver + "|" + r.team
        r["ds"] = r.driver + "|" + r.season.astype(str)
        self.drivers = sorted(r.driver.unique()); self.ts = sorted(r.ts.unique()); self.dteams = sorted(r.dteam.unique())
        self.circuits = sorted(r.circuit.unique()); self.dss = sorted(r.ds.unique())
        ix = lambda col, keys: r[col].map({k: i for i, k in enumerate(keys)}).values
        r["d"], r["t"], r["dt"], r["c"], r["dsi"] = ix("driver", self.drivers), ix("ts", self.ts), ix("dteam", self.dteams), ix("circuit", self.circuits), ix("ds", self.dss)
        early = start < 1960   # the 1950s get their own grid effect and noise level
        r["era"] = np.digitize(r.season, ([1966] if early else []) + [1990, 2000, 2010, 2018, 2022, 2026])
        r["decade"] = np.digitize(r.season, ([1966] if early else []) + [1980, 1990, 2000, 2010])
        r["prog"] = r["round"] / r.groupby("season")["round"].transform("max") - .5
        r["x_t"] = np.exp(-r.x_team_n.fillna(1e6).clip(upper=1000) / tau) * r.x_team_n.notna()
        r["x_r"] = np.exp(-r.x_reg_n.fillna(1e6).clip(upper=1000) / tau) * r.x_reg_n.notna()
        r["sprint"] = (r.session == "sprint").astype(float)
        self.res = r; self.holdout = set(holdout); self.train = r[~r.race_id.isin(self.holdout)]
        self.names = r.drop_duplicates("driver").set_index("driver").name.to_dict()
        self.n = dict(d=len(self.drivers), t=len(self.ts), dt=len(self.dteams), c=len(self.circuits), ds=len(self.dss), era=8 if early else 7, dec=6 if early else 5)
        teams = sorted({k.split("|")[0] for k in self.ts}); self.n["team"] = len(teams); self.n["ts_team"] = np.array([teams.index(k.split("|")[0]) for k in self.ts])
        yrs = sorted({int(k.split("|")[1]) for k in self.ts}); self.n["ts_year"] = np.array([yrs.index(int(k.split("|")[1])) for k in self.ts])
        # each team's cars in season order: a team's car is compared with its own recent cars, not its all-time average
        self.n["chain"] = -np.ones((len(teams), len(yrs)), int); self.n["chain"][self.n["ts_team"], self.n["ts_year"]] = np.arange(len(self.ts))
        if os.environ.get("F1_LINK"):   # test: a customer's car shares part of the works car's level that year
            from f1model.v2_inputs import customer_entrants
            ix_ = {k: i for i, k in enumerate(self.ts)}; link = {}
            for (y, c, d_), t_ in customer_entrants().items():
                if t_ != c and f"{t_}|{y}" in ix_ and f"{c}|{y}" in ix_: link.setdefault(ix_[f"{t_}|{y}"], ix_[f"{c}|{y}"])
            self.n["link_child"] = np.array(sorted(link), int); self.n["link_parent"] = np.array([link[k] for k in sorted(link)], int)
        self.report_from = start if early or start >= 1970 else 1970
        self.multi_decade = start < 2010
        # career matrix for the form random walk
        idx = {k: i for i, k in enumerate(self.dss)}
        by = {}
        for k in self.dss:
            d, s = k.split("|"); by.setdefault(d, []).append(int(s))
        K = max(map(len, by.values()))
        self.car_idx = np.zeros((len(self.drivers), K), int); self.car_mask = np.zeros((len(self.drivers), K), bool); self.step = np.ones(len(self.dss))
        for i, d in enumerate(self.drivers):
            ss = sorted(by[d])
            for j, s in enumerate(ss):
                self.car_idx[i, j] = idx[f"{d}|{s}"]; self.car_mask[i, j] = True
                gap = 0 if j == 0 else s - ss[j - 1]
                # time away from F1 lets a driver's level move much further than a normal season does
                self.step[idx[f"{d}|{s}"]] = 0.0 if j == 0 else (COMEBACK_MULT * np.sqrt(gap) if gap >= 2 else 1.0)
        self.dt_driver = np.array([self.drivers.index(k.split("|")[0]) for k in self.dteams])
        counts = r[r.started & (r.session == "gp")].groupby("dt").size()
        self.dt_weight = np.array([max(counts.get(i, 0), 1) for i in range(len(self.dteams))], float)
        self.wet_pairs = r[r.session == "gp"].drop_duplicates("race_id")[["wet_q", "wet_r", "chg_r"]].values

    def base(self, df, wet):
        return dict(d=df.d.values, t=df.t.values, dt=df.dt.values, ds=df.dsi.values, c=df.c.values, era=df.era.values, dec=df.decade.values, prog=df.prog.values,
                    x_t=df.x_t.values, x_r=df.x_r.values, wet=df[wet].values, sprint=df.sprint.values)

def ranked(df, order, wet):
    df = df.reset_index(drop=True)
    gs = [g.sort_values(order).index.values for _, g in df.groupby("race_id")]
    gs = [g for g in gs if len(g) >= 2]
    w = max(map(len, gs)); idx = np.zeros((len(gs), w), int); mask = np.zeros((len(gs), w), bool)
    for i, g in enumerate(gs):
        idx[i, :len(g)] = g; mask[i, :len(g)] = True
    b = {k: v[idx] for k, v in {**_b(df, wet), "grid": _grid(df)}.items()}
    b["mask"] = mask
    return b, df, idx

def _b(df, wet):
    return dict(d=df.d.values, t=df.t.values, dt=df.dt.values, ds=df.dsi.values, c=df.c.values, era=df.era.values, dec=df.decade.values, prog=df.prog.values,
                x_t=df.x_t.values, x_r=df.x_r.values, wet=df[wet].values.astype(float), sprint=df.sprint.values, chg=(df.chg_r.values.astype(float) if wet != "wet_q" else np.zeros(len(df))))

def _grid(df):
    return (np.nan_to_num(df.grid.values.astype(float), nan=20.0).clip(max=26) - 10.5) / 9.5

def pl(u, mask):
    u = jnp.where(mask, u, -jnp.inf)
    tail = jax.lax.cumlogsumexp(u, axis=1, reverse=True)
    return jnp.where(mask, u - tail, 0.0).sum()

def nc(name, scale, size):
    return numpyro.deterministic(name, scale * numpyro.sample(name + "_z", dist.Normal(0, 1).expand([size])))

def latent(D, unit):
    n = D["n"]; S = {}
    tau = numpyro.sample("tau_form", dist.HalfNormal(FORM_PRIOR * unit))
    if os.environ.get("F1_FORM"): tau = float(os.environ["F1_FORM"]) * unit   # test: fix how far a driver's form can move, so more of a dominant season goes to the car
    z = numpyro.sample("form_z", dist.Normal(0, 1).expand([n["ds"]]))
    steps = z * tau * D["step"]
    m = D["car_mask"]; cum = jnp.cumsum(steps[D["car_idx"]] * m, axis=1)
    cum = cum - (cum * m).sum(1, keepdims=True) / m.sum(1, keepdims=True)
    if "ds_age" in D:   # test: a shared age curve under the career random walk, centred per driver
        a1 = numpyro.sample("age_1", dist.Normal(0, 1)); a2 = numpyro.sample("age_2", dist.Normal(0, 1))
        A = D["ds_age"][D["car_idx"]]; ag = a1 * A + a2 * A ** 2
        cum = cum + (ag - (ag * m).sum(1, keepdims=True) / m.sum(1, keepdims=True)) * m
    S["form"] = numpyro.deterministic("form", jnp.zeros(n["ds"]).at[D["car_idx"][m]].set(cum[m]))
    S["fit_sd"] = numpyro.sample("fit_sd", dist.HalfNormal(.4 * unit))
    fz = S["fit_sd"] * numpyro.sample("fit_z", dist.Normal(0, 1).expand([n["dt"]]))
    # weighted by races, so a driver's skill reflects where they spent their career
    tot = jax.ops.segment_sum(fz * D["dt_weight"], D["dt_driver"], n["d"]); cnt = jax.ops.segment_sum(D["dt_weight"], D["dt_driver"], n["d"])
    S["fit"] = numpyro.deterministic("fit", fz - (tot / cnt)[D["dt_driver"]])
    S["ad_t_mu"] = numpyro.sample("ad_t_mu", dist.Normal(0, .5 * unit)); S["ad_r_mu"] = numpyro.sample("ad_r_mu", dist.Normal(0, .5 * unit))
    S["ad_t"] = numpyro.deterministic("ad_t", S["ad_t_mu"] + nc("ad_t_d", numpyro.sample("ad_t_sd", dist.HalfNormal(.4 * unit)), n["d"]))
    S["ad_r"] = numpyro.deterministic("ad_r", S["ad_r_mu"] + nc("ad_r_d", numpyro.sample("ad_r_sd", dist.HalfNormal(.4 * unit)), n["d"]))
    S["r_scale"] = numpyro.sample("r_scale", dist.LogNormal(0, .5))
    return S

def perf(S, a, sk, wetd, car, drift, wetc, scale=1.0):
    return sk[a["d"]] + car[a["t"]] + drift[a["t"]] * a["prog"] + (wetd[a["d"]] + wetc[a["t"]]) * a["wet"] \
        + scale * (S["form"][a["ds"]] + S["fit"][a["dt"]] + S["ad_t"][a["d"]] * a["x_t"] + S["ad_r"][a["d"]] * a["x_r"])

def common(n, u):
    v = {}
    for k, skd, cd, wd, wc, dd in [("q", 1.5, 3, .4, .4, 1), ("r", 1.5, 3, .4, .4, 1)]:
        v["sk" + k] = nc("skill_" + k, numpyro.sample(f"sk_{k}_sd", dist.HalfNormal(skd * u)), n["d"])
        # a car follows on from the team's previous car (AR1), so dominance carries over but old eras do not
        rho = numpyro.sample(f"car_{k}_rho", dist.Beta(2, 2)); s_car = numpyro.sample(f"car_{k}_sd", dist.HalfNormal(cd * u))
        import os as _o
        if _o.environ.get("F1_CAR_RHO"): rho = float(_o.environ["F1_CAR_RHO"])   # test: how much a car carries over from the team's last car
        _t = _o.environ.get("F1_CAR_T")   # test: heavy-tailed car steps, so a very dominant car is not treated as near-impossible
        if _t: zc = numpyro.sample(f"car_{k}_z", dist.StudentT(float(_t) if _t != "est" else 2 + numpyro.sample(f"car_{k}_nu", dist.Gamma(2, .1)), 0, 1).expand([n["t"]]))
        else: zc = numpyro.sample(f"car_{k}_z", dist.Normal(0, 1).expand([n["t"]]))
        M = n["chain"]; x = zc[M[:, 0]]; cols = [x]
        for y in range(1, M.shape[1]):
            x = jnp.where(M[:, y - 1] >= 0, rho * x + jnp.sqrt(1 - rho ** 2) * zc[M[:, y]], zc[M[:, y]]); cols.append(x)
        car_ = s_car * jnp.stack(cols, 1)[n["ts_team"], n["ts_year"]]
        if "link_child" in n:
            car_ = car_.at[n["link_child"]].add(numpyro.sample(f"link_{k}", dist.Beta(2, 2)) * car_[n["link_parent"]])
        v["car" + k] = numpyro.deterministic("car_" + k, car_)
        v["dr" + k] = nc("drift_" + k, numpyro.sample(f"dr_{k}_sd", dist.HalfNormal(dd * u)), n["t"])
        v["wet" + k] = nc("wet_" + k, numpyro.sample(f"wet_{k}_sd", dist.HalfNormal(wd * u)), n["d"])
        v["wc" + k] = nc("wetcar_" + k, numpyro.sample(f"wc_{k}_sd", dist.HalfNormal(wc * u)), n["t"])
    v["chg"] = nc("chg_r", numpyro.sample("chg_r_sd", dist.HalfNormal(.4 * u)), n["d"])   # changeable races: 20-50% on wet tyres
    return v

def errors(E, n):
    a = numpyro.sample("err_dec", dist.Normal(-3.5, 1).expand([n["dec"]]))
    w = numpyro.sample("err_wet", dist.Normal(.5, 1))
    e = nc("err_d", numpyro.sample("err_sd", dist.HalfNormal(.5)), n["d"])
    numpyro.sample("err_obs", dist.Bernoulli(logits=a[E["dec"]] + w * E["wet"] + e[E["d"]]), obs=E["y"])

def grid_terms(a, n):
    g = numpyro.sample("grid_era", dist.Normal(-2, 2).expand([n["era"]]))
    gc = nc("grid_c", numpyro.sample("grid_c_sd", dist.HalfNormal(1)), n["c"])
    gs = numpyro.sample("grid_sprint", dist.Normal(0, 1))
    return (g[a["era"]] + gc[a["c"]] + gs * a["sprint"]) * a["grid"]

def model_b(D):
    n = D["n"]; S = latent(D, 1.0); v = common(n, 1.0)
    kq = kr = jnp.ones(5)
    if D["multi"]:
        kq = jnp.append(numpyro.sample("kappa_q", dist.LogNormal(0, .5).expand([n["dec"] - 1])), 1.0)
        kr = jnp.append(numpyro.sample("kappa_r", dist.LogNormal(0, .5).expand([n["dec"] - 1])), 1.0)
    q, r = D["q"], D["r"]
    qm = kq[q["dec"]] * perf(S, q, v["skq"], v["wetq"], v["carq"], v["drq"], v["wcq"])
    numpyro.factor("q_pl", pl(qm, q["mask"]))
    if "qt" in q:   # gaps within a session, centred, heavy tails; wet sessions noisier; down-weighted because order and gap overlap
        gm = q["qt_m"]; cnt = jnp.maximum(gm.sum(1, keepdims=True), 1)
        cen = lambda x: (x - (x * gm).sum(1, keepdims=True) / cnt) * gm
        b_ = numpyro.sample("qt_b", dist.HalfNormal(2).expand([n["dec"]])); s_ = numpyro.sample("qt_s", dist.HalfNormal(2).expand([n["dec"]])); sw = numpyro.sample("qt_wet", dist.LogNormal(0, 1))
        numpyro.factor("qt", D["qt_w"] * (dist.StudentT(4, b_[q["dec"]] * cen(qm), (s_[q["dec"]] + 1e-3) * (1 + sw * q["wet"])).log_prob(cen(q["qt"])) * gm).sum())
    rm = kr[r["dec"]] * perf(S, r, v["skr"], v["wetr"], v["carr"], v["drr"], v["wcr"], S["r_scale"]) + grid_terms(r, n) + (v["chg"][r["d"]] + .5 * v["wcr"][r["t"]]) * r["chg"]
    numpyro.factor("r_pl", pl(rm, r["mask"]))
    if "gap" in r:   # margins within a race, centred, heavy tails for pit problems; down-weighted because order and margin overlap
        gm = r["gap_m"]; cnt = jnp.maximum(gm.sum(1, keepdims=True), 1)
        cen = lambda x: (x - (x * gm).sum(1, keepdims=True) / cnt) * gm
        b_ = numpyro.sample("gap_b", dist.HalfNormal(2).expand([n["dec"]])); s_ = numpyro.sample("gap_s", dist.HalfNormal(2).expand([n["dec"]]))
        numpyro.factor("gap", D["gap_w"] * (dist.StudentT(4, b_[r["dec"]] * cen(rm), s_[r["dec"]] + 1e-3).log_prob(cen(r["gap"])) * gm).sum())
    errors(D["err"], n)

def model_a(D):
    n = D["n"]; S = latent(D, .3); v = common(n, .3)
    sg = {k: numpyro.sample("sig_" + k, dist.HalfNormal(.7)) for k in ["q", "r", "ql", "rl"]}
    for key, sk, w, c, dr, wc, ss, sl, scale in [("q", v["skq"], v["wetq"], v["carq"], v["drq"], v["wcq"], sg["q"], sg["ql"], 1.0), ("r", v["skr"], v["wetr"], v["carr"], v["drr"], v["wcr"], sg["r"], sg["rl"], S["r_scale"])]:
        a = D[key]; m = perf(S, a, sk, w, c, dr, wc, scale)
        numpyro.sample(key + "_obs", dist.StudentT(4, m, jnp.sqrt(a["se"] ** 2 + (ss + (sl - ss) * a["wet"]) ** 2)), obs=a["adv"])
    cr = D["craft"]
    craft = nc("craft", numpyro.sample("craft_sd", dist.HalfNormal(.5)), n["d"])
    cw = nc("craft_wet", numpyro.sample("craft_wet_sd", dist.HalfNormal(.3)), n["d"])
    strat = nc("strategy", numpyro.sample("strat_sd", dist.HalfNormal(.5)), n["t"])
    b = numpyro.sample("b_pace", dist.LogNormal(0, 1))
    numpyro.factor("craft_pl", pl(craft[cr["d"]] + cw[cr["d"]] * cr["wet"] + v["chg"][cr["d"]] * cr["chg"] + strat[cr["t"]] + b * cr["adv"] + grid_terms(cr, n), cr["mask"]))
    errors(D["err"], n)

def _ctx(data):
    return dict(n=data.n, car_idx=data.car_idx, car_mask=data.car_mask, step=data.step, dt_driver=data.dt_driver, dt_weight=data.dt_weight, multi=data.multi_decade)

def _age(data):
    a = data.res.groupby("dsi").age.mean(); return ((a.reindex(range(data.n["ds"])).fillna(28.) - 28) / 10).values.astype(np.float32)

def _err(tr):
    e = tr[tr.started & ~tr.no_race & (tr.session == "gp")]
    return dict(d=e.d.values, dec=e.decade.values, wet=e.wet_r.values + .5 * e.chg_r.values, y=e.error.values.astype(float))

def inputs_b(data):
    tr = data.train; D = _ctx(data)
    if os.environ.get("F1_AGE"): D["ds_age"] = _age(data)
    D["q"] = ranked(tr[tr.quali_pos.notna() & (tr.session == "gp")], "quali_pos", "wet_q")[0]
    D["r"] = ranked(tr[tr.finished & ~tr.no_race], "finish", "wet_r")[0]
    if os.environ.get("F1_QT"):   # test: use the qualifying gap as a % of lap time, not only the order
        from f1model.config import PROCESSED
        g = pd.read_pickle(PROCESSED / "qtimes_v2.pkl").drop_duplicates(["race_id", "driver"]).set_index(["race_id", "driver"]).qdef
        b, df, idx = ranked(tr[tr.quali_pos.notna() & (tr.session == "gp")], "quali_pos", "wet_q")
        x = pd.MultiIndex.from_arrays([df.race_id, df.driver]).map(g.to_dict()).values.astype(float)
        b["qt"] = np.nan_to_num(-100 * np.minimum(x, .10))[idx]; b["qt_m"] = np.isfinite(x)[idx] & b["mask"]; D["q"] = b; D["qt_w"] = float(os.environ["F1_QT"])
    if os.environ.get("F1_GAP"):   # test: also use how far behind each finisher was, not only the order
        from f1model.config import PROCESSED
        g = pd.read_pickle(PROCESSED / "gaps_v2.pkl").drop_duplicates(["race_id", "driver"]).set_index(["race_id", "driver"]).deficit
        b, df, idx = ranked(tr[tr.finished & ~tr.no_race], "finish", "wet_r")
        x = pd.MultiIndex.from_arrays([df.race_id, df.driver]).map(g.to_dict()).values.astype(float)
        b["gap"] = np.nan_to_num(-100 * np.minimum(x, .15))[idx]; b["gap_m"] = np.isfinite(x)[idx] & b["mask"]; D["r"] = b; D["gap_w"] = float(os.environ["F1_GAP"])
    D["err"] = _err(tr)
    return D

def inputs_a(data, pace):
    tr = data.train; D = _ctx(data)
    p = pace.drop(columns=["team", "season", "number", "code", "session"], errors="ignore").merge(tr, on=["race_id", "driver"])
    med = p.lap_sd[p.lap_sd > 0].median()
    p["se"] = p.lap_sd.fillna(med).clip(lower=med / 2, upper=3 * med) / np.sqrt(p.n); p["adv"] = -p.gap
    p["low"] = (p.cond == "low").astype(float)
    for key in "qr":
        x = p[p.kind == key]; b = _b(x, "low"); b["wet"] = x.low.values; b["adv"], b["se"] = x.adv.values, x.se.values
        D[key] = b
    rr = p[p.kind == "r"]
    # wet races use low-grip pace, otherwise dry pace
    w = rr.sort_values("low").drop_duplicates(["race_id", "driver"], keep="last")
    f = w[w.finished & ~w.no_race].reset_index(drop=True)
    c, df, idx = ranked(f, "finish", "low"); c["adv"] = f.adv.values[idx]
    D["craft"] = c; D["err"] = _err(tr)
    return D

def to_jax(D):
    f = lambda x: {k: f(v) for k, v in x.items()} if isinstance(x, dict) else jnp.asarray(x) if isinstance(x, np.ndarray) else x
    return f(D)

def fit(model, D, warmup, samples, tree=8, chains=2, seed=0, ckpt=None, segments=1):
    """Warm up, then sample in segments, saving after each so a restart loses little."""
    D = to_jax(D); kernel = NUTS(model, target_accept_prob=.85, max_tree_depth=tree, init_strategy=init_to_median(num_samples=5))
    m = MCMC(kernel, num_warmup=warmup, num_samples=samples // segments, num_chains=chains, chain_method="parallel", progress_bar=False)
    start = 0; chunks = []
    if ckpt and ckpt.exists():
        st = pickle.load(open(ckpt, "rb")); chunks, start = st["chunks"], len(st["chunks"]); m.post_warmup_state = st["state"]
    for i in range(start, segments):
        m.run(m.post_warmup_state.rng_key if m.post_warmup_state is not None else jax.random.PRNGKey(seed), D)
        keep = {k: np.asarray(v).astype(np.float32) for k, v in m.get_samples(group_by_chain=True).items() if k.endswith("_z") is False}
        chunks.append(keep); m.post_warmup_state = m.last_state
        if ckpt:
            pickle.dump(dict(chunks=chunks, state=jax.device_get(m.last_state)), open(ckpt, "wb"))
    return {k: np.concatenate([c[k] for c in chunks], axis=1) for k in chunks[0]}
