"""Two deliberately separate models.

Shared idea (both pipelines):
  performance = driver skill + wet adjustment + team-season car (with in-season drift)
              + driver-team-season deviation ("fit or form") + age curve

Pipeline B ("results"): rank-ordered logit on the qualifying and finishing ORDER.
Pipeline A ("laps"):    pace gaps from push laps/clean race laps, plus a "racecraft" rank model
                        for what pace and grid together do not explain.

Each driver has their own deviation scale `sigma_dev`: a low value means they perform
close to their level in every car, i.e. adaptability.
Linear age cannot be separated from driver and season effects (teammates age together), so
it is held by a tight prior; only the curvature is really learned from data.
"""
import jax, jax.numpy as jnp, numpy as np
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median

def _centre_by_driver(dev, dts_driver, n_drivers):
    """Season deviations sum to zero within each driver. Anything persistent is then skill by
    definition, and the deviation scale measures real variation between a driver's seasons."""
    total = jax.ops.segment_sum(dev, dts_driver, n_drivers)
    count = jax.ops.segment_sum(jnp.ones_like(dev), dts_driver, n_drivers)
    return dev - (total / count)[dts_driver]

def _pl_loglik(u, mask):
    """Plackett-Luce log-likelihood; u is ordered best→worst, mask marks real entries."""
    u = jnp.where(mask, u, -jnp.inf)
    tail = jax.lax.cumlogsumexp(u, axis=1, reverse=True)
    return jnp.where(mask, u - tail, 0.0).sum()

def _shared(n):
    """Latents common to both pipelines, non-centred so NUTS behaves."""
    def nc(name, scale, size):
        return numpyro.deterministic(name, scale * numpyro.sample(name + "_z", dist.Normal(0, 1).expand([size])))
    s = {}
    log_mu = numpyro.sample("log_dev_mu", dist.Normal(-1.5, 0.7))
    log_tau = numpyro.sample("log_dev_tau", dist.HalfNormal(0.4))
    eta = numpyro.sample("sigma_eta", dist.Normal(0, 1).expand([n["d"]]))
    s["sigma_dev"] = numpyro.deterministic("sigma_dev", jnp.exp(log_mu + log_tau * eta))
    s["z_dev"] = numpyro.sample("z_dev", dist.Normal(0, 1).expand([n["dt"]]))
    s["nc"] = nc
    s["age1"] = numpyro.sample("age1", dist.Normal(0, 0.05))
    s["age2"] = numpyro.sample("age2", dist.Normal(-0.5, 0.5))
    return s

def _error_model(df_err, n, nc):
    a = numpyro.sample("err_a", dist.Normal(-3.5, 1))
    a_wet = numpyro.sample("err_wet", dist.Normal(0.5, 1))
    e = nc("err_d", numpyro.sample("err_sd", dist.HalfNormal(0.5)), n["d"])
    numpyro.sample("err_obs", dist.Bernoulli(logits=a + a_wet * df_err["wet"] + e[df_err["d"]]), obs=df_err["y"])

def model_results(q, r, err, n, dts_driver):
    s = _shared(n); nc = s["nc"]
    sk_q = nc("skill_q", numpyro.sample("sk_q_sd", dist.HalfNormal(1.5)), n["d"])
    sk_r = nc("skill_r", numpyro.sample("sk_r_sd", dist.HalfNormal(1.5)), n["d"])
    wet_q = nc("wet_q", numpyro.sample("wet_q_sd", dist.HalfNormal(0.4)), n["d"])
    wet_r = nc("wet_r", numpyro.sample("wet_r_sd", dist.HalfNormal(0.4)), n["d"])
    car_q = nc("car_q", numpyro.sample("car_q_sd", dist.HalfNormal(3)), n["t"])
    car_r = nc("car_r", numpyro.sample("car_r_sd", dist.HalfNormal(3)), n["t"])
    dr_q = nc("drift_q", numpyro.sample("dr_q_sd", dist.HalfNormal(1)), n["t"])
    dr_r = nc("drift_r", numpyro.sample("dr_r_sd", dist.HalfNormal(1)), n["t"])
    dev_scale_r = numpyro.sample("dev_scale_r", dist.LogNormal(0, 0.5))
    dev_q = numpyro.deterministic("dev_q", _centre_by_driver(s["sigma_dev"][dts_driver] * s["z_dev"], dts_driver, n["d"]))
    dev_r = numpyro.deterministic("dev_r", dev_q * dev_scale_r)
    age = lambda a: s["age1"] * a + s["age2"] * a ** 2
    uq = sk_q[q["d"]] + wet_q[q["d"]] * q["wet"] + car_q[q["t"]] + dr_q[q["t"]] * q["prog"] + dev_q[q["dt"]] + age(q["age"])
    numpyro.factor("quali_pl", _pl_loglik(uq, q["mask"]))
    g0 = numpyro.sample("grid_era", dist.Normal(-2, 2).expand([3]))
    g_c = nc("grid_c", numpyro.sample("grid_c_sd", dist.HalfNormal(1)), n["c"])
    ur = (sk_r[r["d"]] + wet_r[r["d"]] * r["wet"] + car_r[r["t"]] + dr_r[r["t"]] * r["prog"] + dev_r[r["dt"]]
          + age(r["age"]) + (g0[r["era"]] + g_c[r["c"]]) * r["grid"])
    numpyro.factor("race_pl", _pl_loglik(ur, r["mask"]))
    _error_model(err, n, nc)

def model_laps(q, r, craft, err, n, dts_driver):
    s = _shared(n); nc = s["nc"]
    # pace advantage in % of lap time; positive = faster
    sk_q = nc("skill_q", numpyro.sample("sk_q_sd", dist.HalfNormal(0.5)), n["d"])
    sk_r = nc("skill_r", numpyro.sample("sk_r_sd", dist.HalfNormal(0.5)), n["d"])
    wet_q = nc("wet_q", numpyro.sample("wet_q_sd", dist.HalfNormal(0.2)), n["d"])
    wet_r = nc("wet_r", numpyro.sample("wet_r_sd", dist.HalfNormal(0.2)), n["d"])
    car_q = nc("car_q", numpyro.sample("car_q_sd", dist.HalfNormal(1.5)), n["t"])
    car_r = nc("car_r", numpyro.sample("car_r_sd", dist.HalfNormal(1.5)), n["t"])
    dr_q = nc("drift_q", numpyro.sample("dr_q_sd", dist.HalfNormal(0.5)), n["t"])
    dr_r = nc("drift_r", numpyro.sample("dr_r_sd", dist.HalfNormal(0.5)), n["t"])
    dev_scale = numpyro.sample("dev_scale_laps", dist.LogNormal(-1, 0.7))   # % per unit of deviation
    dev_scale_r = numpyro.sample("dev_scale_r", dist.LogNormal(0, 0.5))
    dev_q = numpyro.deterministic("dev_q", _centre_by_driver(dev_scale * s["sigma_dev"][dts_driver] * s["z_dev"], dts_driver, n["d"]))
    dev_r = numpyro.deterministic("dev_r", dev_q * dev_scale_r)
    age = lambda a: 0.3 * (s["age1"] * a + s["age2"] * a ** 2)
    sig_q = numpyro.sample("sig_q", dist.HalfNormal(0.5)); sig_r = numpyro.sample("sig_r", dist.HalfNormal(0.5))
    sig_q_wet = numpyro.sample("sig_q_wet", dist.HalfNormal(1)); sig_r_wet = numpyro.sample("sig_r_wet", dist.HalfNormal(1))
    mq = sk_q[q["d"]] + wet_q[q["d"]] * q["wet"] + car_q[q["t"]] + dr_q[q["t"]] * q["prog"] + dev_q[q["dt"]] + age(q["age"])
    mr = sk_r[r["d"]] + wet_r[r["d"]] * r["wet"] + car_r[r["t"]] + dr_r[r["t"]] * r["prog"] + dev_r[r["dt"]] + age(r["age"])
    # races and sessions are centred before this stage, so no session intercepts are needed
    numpyro.sample("q_obs", dist.StudentT(4, mq, jnp.sqrt(q["se"] ** 2 + (sig_q + sig_q_wet * q["wet"]) ** 2)), obs=q["adv"])
    numpyro.sample("r_obs", dist.StudentT(4, mr, jnp.sqrt(r["se"] ** 2 + (sig_r + sig_r_wet * r["wet"]) ** 2)), obs=r["adv"])
    # Racecraft: finishing order not explained by observed pace and starting position
    craft_d = nc("craft", numpyro.sample("craft_sd", dist.HalfNormal(0.5)), n["d"])
    craft_wet = nc("craft_wet", numpyro.sample("craft_wet_sd", dist.HalfNormal(0.3)), n["d"])
    strat = nc("strategy", numpyro.sample("strat_sd", dist.HalfNormal(0.5)), n["t"])
    b_pace = numpyro.sample("b_pace", dist.LogNormal(0, 1))
    g0 = numpyro.sample("grid_era", dist.Normal(-2, 2).expand([3]))
    g_c = nc("grid_c", numpyro.sample("grid_c_sd", dist.HalfNormal(1)), n["c"])
    u = (craft_d[craft["d"]] + craft_wet[craft["d"]] * craft["wet"] + strat[craft["t"]] + b_pace * craft["adv"]
         + (g0[craft["era"]] + g_c[craft["c"]]) * craft["grid"])
    numpyro.factor("craft_pl", _pl_loglik(u, craft["mask"]))
    _error_model(err, n, nc)

def fit(model, *args, warmup=500, samples=500, chains=2, seed=0, progress=False, tree=8):
    kernel = NUTS(model, target_accept_prob=0.85, max_tree_depth=tree, init_strategy=init_to_median(num_samples=10))
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=chains,
                chain_method="parallel", progress_bar=progress)
    mcmc.run(jax.random.PRNGKey(seed), *args)
    return mcmc
