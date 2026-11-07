"""Results from 1950, kept in a separate file so B80 stays reproducible. Adds 1950s-60s rule changes."""
import f1model.v2_inputs as V
V.FIRST = 1950; V.REG_SEASONS = (1952, 1954, 1961, 1966) + V.REG_SEASONS
r = V.build_results("results_v2_50.pkl")
print(r[r.season < 1970].groupby("season").agg(races=("race_id", "nunique"), rows=("driver", "size"), drivers=("driver", "nunique"), teams=("team", "nunique"), shared=("shared_car", "sum")))
