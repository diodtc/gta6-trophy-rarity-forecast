# GTA 6 trophy rarity forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/diodtc/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/diodtc/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn
rate on Exophase at 2026-11-26T00:00:00Z — five quantiles per trophy, for
make.fun's `gta6-trophy-rarity-forecast-01` bounty.

## What it does

`main.py` (Python 3, standard library only, no network) reads a case file —
every read the house made of the game's Exophase trophy page before the
case's checkpoint — and writes one forecast file per case.

The key observation is that an earn rate is a ratio of two moving
quantities: earners creep up steadily as players progress, while
players-tracked jumps in steps at Exophase refreshes, so rates usually
fall at a refresh (dilution) except for very early trophies whose earners
outgrow players. The program therefore:

- deduplicates history reads into distinct snapshots (identical reads
  between refreshes count once),
- recovers earners per snapshot as `E = round(rate * N / 100)`,
- fits, per trophy, an ensemble (median in logit space) of a concave
  quadratic of logit(rate) against `ln(1 + hours since release)` — rates
  saturate, so the curve bends down toward a ceiling — a linear fit on
  the last 24h of history, and the recent level (mean logit of the last
  five snapshots),
- falls back, on a flat history with fewer than 3 distinct snapshots, to
  frozen earners with players assumed to jump 3x by the target (a refresh
  is assumed before it), diluting every trophy including the 100% one,
- widens the spread for trophies nobody has earned yet (median 0.01%,
  sigma 1.5), since endgame trophies can pick up their first earners by
  the target.

## Run it locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory and writes `<id>.out` for each
to the output directory. Deterministic: two runs produce byte-identical
output. Runtime is ~0.05 s for the two practice cases, far under the
8 s limit.

## Local validation

- Practice cases p00/p01 against the hypothesized answers (frozen
  earners, N = 3868 at the refresh): pinball loss **0.364** (typical miss
  0.45 points, 90% coverage 1.00), vs **1.883** for the starter/persistence
  baseline.
- Synthetic final-like cases (known truth: N(t) growing with ~6h refresh
  steps, per-trophy regimes — fast-rising early, diluting late, saturating
  mid, late-blooming, zero, 100% — history cut at the deadline, target
  4.5 days later): losses **0.64 / 0.68 / 0.71 / 0.66 / 0.32**, beating a
  persistence baseline (**0.90 / 0.86 / 0.95 / 0.91 / 0.57**) on every case.
