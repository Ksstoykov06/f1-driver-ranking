"""Every modelling assumption that a person might want to argue with lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

FIRST_SEASON, LAST_SEASON = 2018, 2026
REG_CHANGE_SEASONS = (2022, 2026)      # major regulation resets inside the pilot window
MIN_RACES_TO_RANK = 40                 # drivers below this are modelled but not ranked

# ---- Retirement classification (F1DB `reasonRetired`) -----------------------------
# Solo errors count against the driver. Collisions cannot be attributed from results
# alone, so the base case ignores them; `collisions_as_errors` tests that assumption.
SOLO_ERROR = {"Accident", "Accident damage", "Spun off", "Accident in qualifying"}
COLLISION = {"Collision", "Collision damage"}

# ---- Lap filters -----------------------------------------------------------------
PUSH_WINDOW_DRY = 1.035    # a qualifying lap is "push" if within 3.5% of the driver's best in that segment
PUSH_WINDOW_WET = 1.06     # wet laps vary more
SEGMENT_WINDOW_DRY = 1.07  # and within 7% of the fastest lap of the segment, so a lone slow lap never counts
SEGMENT_WINDOW_WET = 1.15
RACE_LAP_WINDOW = 1.07     # race laps slower than 107% of a driver's median are not representative
TRAFFIC_GAP_S = 1.5        # race laps within this gap of the car ahead are treated as traffic
WET_LAP_SHARE = 0.25       # session counts as wet if ≥ this share of laps are on inter/wet tyres
MIN_RACE_LAPS = 8          # clean laps a driver needs for a usable race-pace estimate
MIN_PUSH_LAPS = 1

# ---- Simulation -----------------------------------------------------------------
SIM_WEEKENDS = 400
FIELD_SIZE = 20
