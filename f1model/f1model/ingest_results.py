"""Pipeline B input: official results only, parsed from F1DB's source YAML."""
import pandas as pd, yaml
from .config import RAW, PROCESSED, FIRST_SEASON, LAST_SEASON

def _load(p):
    return yaml.safe_load(p.read_text(encoding="utf8")) if p.exists() else None

def _num(x):
    return int(x) if isinstance(x, int) or (isinstance(x, str) and x.isdigit()) else None

def build_results():
    base = RAW / "f1db" / "src" / "data"
    drivers = {}
    for p in (base / "drivers").glob("*.yml"):
        d = _load(p)
        drivers[d["id"]] = dict(name=d["name"], code=d.get("abbreviation"), dob=d.get("dateOfBirth"))
    rows = []
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        for rd in sorted((base / "seasons" / str(season) / "races").iterdir()):
            race, rr = _load(rd / "race.yml"), _load(rd / "race-results.yml")
            if not rr:                       # scheduled, not yet run
                continue
            quali = {q["driverId"]: q for q in (_load(rd / "qualifying-results.yml") or [])}
            for r in rr:
                q = quali.get(r["driverId"], {})
                grid = r.get("gridPosition")
                rows.append(dict(
                    season=season, round=race["round"], race_id=race["id"], gp=rd.name.split("-", 1)[1],
                    circuit=race["circuitId"], date=str(race["date"]), quali_date=str(race.get("qualifyingDate")),
                    driver=r["driverId"], number=r.get("driverNumber"), team=r["constructorId"],
                    pos_text=str(r["position"]), finish=_num(r["position"]), reason=r.get("reasonRetired"),
                    grid=_num(grid), pit_start=(grid == "PL"), quali_pos=_num(q.get("position")),
                    laps=r.get("laps"), race_laps=race.get("laps"),
                ))
    df = pd.DataFrame(rows)
    d = pd.DataFrame.from_dict(drivers, orient="index").rename_axis("driver").reset_index()
    df = df.merge(d, on="driver", how="left")
    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = (pd.to_datetime(df["date"]) - df["dob"]).dt.days / 365.25
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_pickle(PROCESSED / "results.pkl")
    return df

if __name__ == "__main__":
    df = build_results()
    print(df.groupby("season").agg(races=("race_id", "nunique"), drivers=("driver", "nunique"), rows=("driver", "size")))
    print(df[df.age.isna()].driver.unique(), df.pos_text.value_counts().to_dict())
