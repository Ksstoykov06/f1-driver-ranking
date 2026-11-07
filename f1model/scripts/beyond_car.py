"""How much did each driver do beyond their car?
For every season, puts the real drivers in their real cars. A driver's gain is their simulated average finish
compared with an average driver in the same car, against the same rivals. Also ranks each car in its season."""
import pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from f1model.config import PROCESSED, ROOT
from f1model.v2_inputs import apply_team_orders, build_wet
from f1model.v2_model import Data
from f1model.v2_score import driver_seasons

OUT = ROOT / "outputs_v2"
def run(name, start, kind, draws=120, weekends=40, seed=9):
    res = apply_team_orders(pd.read_pickle(PROCESSED / "results_v2.pkl")); wet = build_wet(); data = Data(res, wet, start)
    st = pickle.load(open(OUT / "ckpt" / f"{name}.pkl", "rb"))
    s = {k: np.concatenate([c[k] for c in st["chunks"]], axis=1).reshape((-1,) + st["chunks"][0][k].shape[2:]) for k in ["skill_r", "skill_q", "form", "car_r", "car_q", "ad_t", "ad_r", "ad_t_mu", "ad_r_mu", "r_scale"] if k in st["chunks"][0]}
    rng = np.random.default_rng(seed); ix = rng.choice(len(s["r_scale"]), min(draws, len(s["r_scale"])), replace=False); s = {k: v[ix] for k, v in s.items()}
    g = driver_seasons(data); ts = {k: i for i, k in enumerate(data.ts)}
    g["t"] = [ts[f"{a}|{b}"] for a, b in zip(g.team, g.season)]
    # car ranks use the whole season's entries
    ranks = {}
    for season, h in g.groupby("season"):
        cars = h.t.unique(); m = (s["car_q"][:, cars] + s["car_r"][:, cars]).mean(0) / 2
        ranks.update({c: r + 1 for r, c in enumerate(cars[np.argsort(-m)])}); ranks[f"n{season}"] = len(cars)
    rows = []
    for season, h in g.groupby("season"):
        n = len(h); d, t = h.d.values, h.t.values
        lvl = s["skill_r"][:, d] + s["r_scale"][:, None] * (s["form"][:, h.dsi.values] + (s["ad_t"][:, d] - s["ad_t_mu"][:, None]) * h.x_t.values + (s["ad_r"][:, d] - s["ad_r_mu"][:, None]) * h.x_r.values)
        lvl = lvl if kind == "B" else lvl * 3.0
        car = s["car_r"][:, t] * (1.0 if kind == "B" else 3.0)
        gains = np.zeros((len(ix), n))
        for k in range(len(ix)):
            for _ in range(weekends):
                e = rng.gumbel(size=n); u = lvl[k] + car[k] + e
                pos = (-u).argsort().argsort() + 1
                for j in range(n):            # same luck, but an average driver in this seat
                    v = u.copy(); v[j] = car[k, j] + e[j]
                    gains[k, j] += (v > v[j]).sum() + 1 - pos[j]
        gains /= weekends
        for j, r in enumerate(h.itertuples()):
            rows.append(dict(driver=r.driver, name=data.names[r.driver], season=season, team=r.team, races=r.races, car_rank=ranks[r.t], cars=ranks[f"n{season}"], gain=gains[:, j].mean(), lo=np.quantile(gains[:, j], .05), hi=np.quantile(gains[:, j], .95)))
    t = pd.DataFrame(rows); t.to_csv(OUT / f"{name}_beyond_car.csv", index=False); return t

if __name__ == "__main__":
    import sys
    NAME, START = (sys.argv[1], int(sys.argv[2])) if len(sys.argv) > 2 else ("B80", 1966)
    t = run(NAME, START, "B")
    print(t.sort_values("gain", ascending=False).head(15).round(2).to_string())
    print(t[t.car_rank >= 3].sort_values("gain", ascending=False).head(15).round(2).to_string())
    print(t[(t.driver == "fernando-alonso") & t.season.isin([2010, 2012, 2013, 2022, 2023])].round(2).to_string())
    print(t[(t.season == 2012)].sort_values("car_rank")[["name", "team", "car_rank", "gain"]].round(2).head(10).to_string())
