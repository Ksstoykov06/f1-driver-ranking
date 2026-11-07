# F1 driver model, version 2

Ranks F1 drivers since 1970 relative to their cars and teammates.

- `B80`: results only, fitted on 1966-2026, scores reported from 1970.
- `B18`: results only, 2018+.
- `A18`: lap data, 2018+.

## Run
`pip install -r requirements.txt`, `bash scripts/download_data.sh`, build inputs with `f1model/v2_inputs.py`, then `bash scripts/run_all_v2.sh`. Stages restart where they stopped. `python3 scripts/build_v2_report.py` builds the page.

## Files
- `f1model/v2_inputs.py`, `lowgrip.py`: results, low-grip laps, wet labels, team orders.
- `f1model/v2_model.py`: the models. Each car follows from the team's previous car (AR1).
- `f1model/v2_score.py`, `scripts/beyond_car.py`: peak, career, adaptability, gain beyond the car.
- `run_v2.py`: fit, held-out test, simulated test, sensitivity.
- `scripts/car_check.py`: car rank against the constructors' table.
- `inputs/`: hand-written lists to review: `wet_evidence.csv` (share of each race on wet tyres, with sources), `wet_leadin_1966_69.csv`, `team_orders.csv`.
- `outputs_v2/`: current results. `outputs_v2_third/`: the first full run. `outputs_v2_fourth/`: the 1980+ run with the corrected cars.

## Long laps run and collisions (5 Nov 2026)
A18 now uses 5,000 warm-up and 20,000 kept steps per chain (worst R-hat 1.0003, lowest effective samples about 6,600). The earlier short run is in outputs_v2_fifth/A18_short. A18_holdout.csv and A18_synth.csv still come from the shorter fits of 4 Nov. scripts/collisions.py fits the separate collision tendency, and `run_v2.py colscore` joins it to B80 (D80_* outputs); it stays out of the main scores.

## Final 1970+ run (5 Nov 2026)
B80 is fitted 1966-2026 with customer-team links (LINEAGE in f1model/v2_inputs.py) and checked 1966-68 wet races, reported from 1970. The main combined ranking is 50/50 peak and career. C66 and C61 in outputs_v2 are the warm-up comparison; the previous B80 is archived in outputs_v2_fifth. scripts/run_final_b80.sh and scripts/run_final_tests.sh rerun the fit and tests, skipping finished stages.

## 1950 run (C50)

`python scripts/build_1950.py` writes data/processed/results_v2_50.pkl, then `python run_v2.py C50`, `holdout C50` and `synth C50`. It excludes Indianapolis, leaves shared-car results unranked, and has its own pre-1966 noise and grid settings. Only peak is compared across all eras; career and combined stay from 1970. Some champion cars are rated too low (Mercedes 1954, Lotus 1962-63), which probably flatters Fangio and Clark.

## Improvement tests and qualifying times (6 Nov 2026)
Run with `MALLOC_ARENA_MAX=1 MALLOC_TRIM_THRESHOLD_=0`, otherwise memory grows about 90 MB a minute and long fits die at the 6 GB limit. Dominant car: scripts/dominant_test.py. Held-out comparisons: scripts/link_test.py (off/on/base/gap/age/qt), scripts/compare_pairs.py. Rankings with vs without qualifying times: scripts/compare_qt.py, scripts/why_qt.py. Results in outputs_v2/gap, outputs_v2/link, outputs_v2/dominant. None of these switches is on in the main model.

## Final improvements and tiers (7 Nov 2026)
- Dominant cars: link_test.py modes cart3, cartest, rho3 and form18. None was adopted.
- Teammate removal: `bash scripts/run_tm.sh x y` (env TM_DROP), then scripts/compare_tm.py → outputs_v2/gap/tm_summary.csv.
- Lap check: scripts/lap_check.py.
- Tiers: scripts/tiers_approx.py builds the tiers from the published 90% intervals of C50 and B80 and writes outputs_v2/tiers_*.csv.
- scripts/tiers.py does the same from full posterior draws. It needs `SAVE_DRAWS=1 LINK_FULL=1 python3 scripts/link_test.py base`, which takes longer than 2h on this machine.
