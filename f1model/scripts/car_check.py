"""Independent check: do the model's car rankings resemble the real constructors' championship?"""
import yaml, numpy as np, pandas as pd
from scipy.stats import spearmanr
from f1model.config import RAW
def standings():
    rows = []
    for y in range(1966, 2027):
        p = RAW / "f1db/src/data/seasons" / str(y) / "constructor-standings.yml"
        for r in (yaml.safe_load(p.read_text()) or []) if p.exists() else []:
            rows.append(dict(season=y, team=r["constructorId"], wcc=r["position"] if isinstance(r["position"], int) else None, points=r.get("points")))
    return pd.DataFrame(rows).drop_duplicates(["season", "team"])
def check(s, ts):
    """s: dict with car_q and car_r draws."""
    m = (np.asarray(s["car_q"]).mean(tuple(range(np.asarray(s["car_q"]).ndim - 1))) + np.asarray(s["car_r"]).mean(tuple(range(np.asarray(s["car_r"]).ndim - 1)))) / 2
    d = pd.DataFrame(dict(team=[k.split("|")[0] for k in ts], season=[int(k.split("|")[1]) for k in ts], car=m)).merge(standings(), on=["season", "team"], how="left").dropna(subset=["wcc"])
    d["car_rank"] = d.groupby("season").car.rank(ascending=False)
    per = d.groupby("season")[["car", "wcc"]].apply(lambda x: spearmanr(-x.car, x.wcc)[0] if len(x) > 3 else np.nan)
    return d, per, (d[d.car_rank == 1].wcc <= 2).mean()
