# Who was the greatest F1 driver once you take the car away?

A Bayesian model that ranks every Formula 1 driver from 1950 to 2026 by how well he drove relative to his car and his teammates, not by what he won.

## Headline results

**Peak, best three straight seasons, all-time from 1950.** Drivers in the same tier can't be reliably separated.

| Tier | Drivers |
|---|---|
| 1 | Fangio, Senna, Clark, Michael Schumacher, Stewart |
| 2 | Graham Hill, Verstappen, Lauda, Rindt, Alonso, Mansell, Piquet, Ascari |
| 3 | Surtees, Moss, Hamilton, Prost, Andretti, Vettel, McLaren, Rosberg, Hunt, Farina, Fittipaldi, Ickx, Scheckter |

**Combined, 50% peak and 50% career, from 1970**

| Tier | Drivers |
|---|---|
| 1 | Schumacher, alone, with a 70% chance of ranking first |
| 2 | Alonso, Verstappen, Senna, Hamilton |
| 3 | Lauda, Piquet, Mansell |

**Strongest single season:** Schumacher, 1994 (Benetton). It's a narrow lead over Fangio's 1953 and 1954.

**Drivers without the best car:** in 2012 Alonso gained more places over an average driver in the same seat than anyone else, in only the third-best car.

## How it works

1. **Data.** Every qualifying session and race from 1950 to 2026 comes from [F1DB](https://github.com/f1db/f1db). Lap times from 2018 come from [TracingInsights](https://github.com/TracingInsights). Lists of wet races, team orders and collisions were added by hand.
2. **Model.** Each result is treated as a Plackett–Luce ranking. A driver's performance is his skill plus his car's speed that season, adjusted for:
   - season-to-season form
   - wet and changeable conditions
   - a dip after changing team or after a major rules change
   - grid position
   - driver errors

   Teammates share a car, so the gap between them measures the drivers. Drivers move between teams, which links every era together.
3. **Fitting.** The model is fitted in numpyro with NUTS: 2 chains, with R-hat between 1.001 and 1.013.
4. **Scores.** Every driver-season from 1950 to 2026 is put into one shared field of identical cars, and thousands of weekends are simulated. A score is the driver's average finish in a 20-car field, so lower is better.
   - **Peak** is the best three-season average.
   - **Career** adds up places gained over an average driver, weighted by races.
   - **Tiers** group drivers who can't be told apart.
5. **Checks.**
   - Hidden races: about 70% of teammate battles predicted correctly, better than grid order or head-to-head record.
   - Known-answer simulations: qualifying skill recovered at 0.91–0.94 and race skill at 0.85–0.87.
   - Car ratings against the constructors' championship: 0.88.
   - Agreement with a separate lap-time model from 2018: 0.90.
6. **Stress tests.** The full list is in `f1model/outputs/f1-model-test-register.xlsx`. There are 48 tests, each marked as in the model, left out, assumption test, or check. They include:
   - letting dominant cars be rated more extreme
   - refitting without a driver's main teammate, 15 times
   - adjusting for how spread out each era's field was
   - warnings for early drivers whose car may be rated too low

## Limitations

- Results show the finishing order, not the margins between drivers.
- Comparisons across distant eras depend on chains of teammates.
- Drivers in slow cars are measured poorly.
- Early cars are sometimes rated too low, which probably flatters Fangio and Graham Hill.
- Wet-weather and adaptability scores are rough.

## Repository layout

- `f1model/f1model/` is the model, simulation and scoring code.
- `f1model/scripts/` holds the scripts for each test, check and report.
- `f1model/run_v2.py` runs the main models (C50, B80, B18 and A18).
- `f1model/outputs_v2/` has the scores, season scores, tiers and diagnostics as CSV and JSON.
- `f1model/outputs/f1-model-walkthrough.html` is the interactive results page; open it in a browser.
- `f1model/inputs/` holds the hand-built lists: wet races, team orders and car lineage.

The raw data isn't included. `scripts/download_data.sh` fetches it. The large posterior files are left out too.

Built by Kaloyan Stoykov, 2026.
