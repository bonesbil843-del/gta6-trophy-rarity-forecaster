# gta6-trophy-rarity-forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bonesbil843-del/gta6-trophy-rarity-forecaster/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bonesbil843-del/gta6-trophy-rarity-forecaster/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, from the recorded history of Grand Theft Auto VI's PlayStation 5
trophy earn rates on Exophase, each trophy's earn rate one week after launch as
five quantiles — the entry for [make.fun's "Forecast how many GTA VI players
earn each trophy in week one" bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## How it works

A trophy's earn rate is `e / P * 100` — the players who earned it over the
players Exophase tracks — so in logit space `logit(r) = ln(e / (P - e))`
exactly. The forecaster models the two counts separately:

1. **Players P** — a recency-weighted fit of `ln P` on `ln(1 + hours since
   launch)`, its slope blended toward the most recent window and decayed with
   the extrapolation distance, since tracked-player adoption slows as it works
   through the cohort.
2. **Earned counts e** — the same fit per trophy on `ln(e + 0.5)`, rebuilt from
   each read's rate and tracked count.

Both extrapolate to the target and combine into the structural logit, blended
with a direct fit of `logit(rate)` on the same scale and with persistence, in
weights set by how the trophy has been moving: a falling rate leans on the
structural dilution view, a flat one on persistence, a rising one on the local
trend. The five quantiles take the standard normal's spread around the blended
median, widened by the fits' residuals, the slopes' standard errors over the
extrapolation distance, the predictors' disagreement and a horizon floor. A
launch-night page that has not refreshed for hours shifts the forecast down for
the dilution its pending refresh will bring.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only. Reads every `*.in` case in the input
directory and writes one `<id>.out` forecast per case into the output
directory.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
