"""Middle option for collisions: no race-by-race blame. Each driver gets a collision tendency, measured against
drivers starting from similar grid slots in the same decade and pulled towards average when evidence is thin.
This block shares no parameters with pace, so it is fitted alone and joined to the B80 draws."""
import sys, pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from scipy.special import expit
from scipy.stats import spearmanr
from f1model.config import PROCESSED, ROOT
from f1model.v2_inputs import apply_team_orders, build_wet
from f1model.v2_model import Data
numpyro.set_host_device_count(2)
OUT = ROOT / "outputs_v2"
def block(data):
    t = data.train; e = t[t.started & ~t.no_race & (t.session == "gp")].copy()
    size = e.groupby("race_id").driver.transform("size"); e["gr"] = ((e.grid.fillna(size / 2) - 1) / (size - 1).clip(lower=1)).clip(0, 1) - .5
    e["wet"] = e.wet_r + .5 * e.chg_r
    return e
def model(d, dec, wet, gr, nd, ndec, y=None):
    a = numpyro.sample("col_dec", dist.Normal(-3, 1).expand([ndec]))
    w = numpyro.sample("col_wet", dist.Normal(.5, 1)); b1 = numpyro.sample("col_g1", dist.Normal(0, 1)); b2 = numpyro.sample("col_g2", dist.Normal(0, 2))
    sd = numpyro.sample("col_sd", dist.HalfNormal(.5))
    z = numpyro.sample("col_z", dist.Normal(0, 1).expand([nd])); e = numpyro.deterministic("col_d", sd * z)
    numpyro.sample("obs", dist.Bernoulli(logits=a[dec] + w * wet + b1 * gr + b2 * gr ** 2 + e[d]), obs=y)
def fit(e, nd, ndec, y, W=1000, S=1000, seed=0):
    m = MCMC(NUTS(model, target_accept_prob=.85), num_warmup=W, num_samples=S, num_chains=2, chain_method="parallel", progress_bar=False)
    m.run(jax.random.PRNGKey(seed), jnp.asarray(e.d.values), jnp.asarray(e.decade.values), jnp.asarray(e.wet.values, dtype=float), jnp.asarray(e.gr.values, dtype=float), nd, ndec, jnp.asarray(y, dtype=float))
    return {k: np.asarray(v) for k, v in m.get_samples(group_by_chain=True).items() if k != "col_z"}
def lp(s, e, driver=True):
    a = s["col_dec"].reshape(-1, s["col_dec"].shape[-1])[:, e.decade.values]; f = lambda k: s[k].reshape(-1, 1)
    x = a + f("col_wet") * e.wet.values + f("col_g1") * e.gr.values + f("col_g2") * e.gr.values ** 2
    if driver: x = x + s["col_d"].reshape(-1, s["col_d"].shape[-1])[:, e.d.values]
    return expit(x).mean(0)
ll = lambda p, y: -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
res = apply_team_orders(pd.read_pickle(PROCESSED / "results_v2.pkl")); data = Data(res, build_wet(), 1966)
e = block(data); nd = data.n["d"]; ndec = data.n["dec"]; y = e.collision.values.astype(float)
print("rows", len(e), "collisions", int(y.sum()), flush=True)
s = fit(e, nd, ndec, y); np.savez_compressed(OUT / "D80_collision_draws.npz", **s)
sd = s["col_sd"]; print("spread", round(float(sd.mean()), 3), np.quantile(sd, [.05, .95]).round(3), flush=True)
from numpyro.diagnostics import gelman_rubin as gr_
print("rhat", round(float(max(np.max(gr_(s[k])) for k in s)), 3), flush=True)
# 1. held out: fit on odd seasons, predict even, and the reverse
tot = {"driver": [], "none": []}; ytrue = []
for r in (0, 1):
    a, b = e[e.season % 2 == r], e[e.season % 2 != r]
    h = fit(a, nd, ndec, a.collision.values.astype(float), 500, 500)
    tot["driver"].append(lp(h, b)); tot["none"].append(lp(h, b, False)); ytrue.append(b.collision.values.astype(float))
yt = np.concatenate(ytrue); L = {k: ll(np.concatenate(v), yt) for k, v in tot.items()}; print("heldout logloss", {k: round(v, 5) for k, v in L.items()}, "n", len(yt), flush=True)
# 2. known answers at the fitted spread
st = e.groupby("d").size(); big = st[st >= 100].index.values; rows = []
base = s["col_dec"].mean((0, 1))[e.decade.values] + s["col_wet"].mean() * e.wet.values + s["col_g1"].mean() * e.gr.values + s["col_g2"].mean() * e.gr.values ** 2
for seed, tr_sd in [(11, float(sd.mean())), (12, float(sd.mean())), (13, float(sd.mean()))]:
    rng = np.random.default_rng(seed); tr = rng.normal(0, tr_sd, nd); ys = rng.random(len(e)) < expit(base + tr[e.d.values])
    h = fit(e, nd, ndec, ys.astype(float), 500, 500, seed); dd = h["col_d"].reshape(-1, nd); lo, hi = np.quantile(dd, [.05, .95], axis=0)
    rows.append((round(spearmanr(dd.mean(0)[big], tr[big])[0], 3), round(float(((lo[big] <= tr[big]) & (tr[big] <= hi[big])).mean()), 3))); print("known answer", rows[-1], flush=True)
# 3. driver table
nm = res.drop_duplicates("driver").set_index("driver").name; dn = pd.Series(data.drivers if hasattr(data, "drivers") else None)
f = s["col_d"].reshape(-1, nd); tb = pd.DataFrame(dict(d=np.arange(nd), eff=f.mean(0), lo=np.quantile(f, .05, 0), hi=np.quantile(f, .95, 0))).merge(e.groupby("d").agg(driver=("driver", "first"), n=("collision", "size"), c=("collision", "sum")).reset_index(), on="d")
tb["name"] = tb.driver.map(nm); tb.to_csv(OUT / "D80_collision_tendency.csv", index=False)
print(tb[tb.n >= 100].sort_values("eff")[["name", "n", "c", "eff", "lo", "hi"]].round(2).pipe(lambda x: pd.concat([x.head(8), x.tail(8)])).to_string(index=False))
print(tb[tb.name.isin(["Michael Schumacher", "Ayrton Senna", "Fernando Alonso", "Max Verstappen", "Lewis Hamilton", "Alain Prost", "Niki Lauda", "Jackie Stewart"])][["name", "n", "c", "eff", "lo", "hi"]].round(2).to_string(index=False))
