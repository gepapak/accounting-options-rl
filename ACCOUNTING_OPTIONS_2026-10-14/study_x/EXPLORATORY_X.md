# Study X exploratory outputs (not registered)

Runs: 32.

| arm | runs | mean X | agents X>0 | gamblers (X>=0.25) | min E | median E | max E | mean E from zero equity | placebo reported | placebo forgiven |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAIN | 8 | 0.836 | 8 | 8 | -27.14 | -14.77 | -1.85 | -25.83 | -14.71 | 81.85 |
| ORACLE | 8 | 0.000 | 0 | 0 | +14.12 | +18.91 | +22.03 | +18.56 | -6.62 | 1.91 |
| SPR | 8 | 0.000 | 0 | 0 | +12.45 | +21.82 | +22.01 | +21.37 | -7.33 | 2.74 |
| TIR | 8 | 0.000 | 0 | 0 | +10.87 | +20.54 | +22.01 | +18.50 | -6.41 | 2.10 |

Booked edge, TIR minus SPR: -2.46 points, 95% bootstrap interval [-6.15, +1.29] (percent of the allocation per week; 10,000 resamples of runs).
Booked edge, TIR minus ledger: -0.27 [-4.25, +3.52].
Booked edge, SPR minus ledger: +2.20 [-1.22, +5.20].

## Audit errors (absolute, points of the allocation per week; the mean over 100 test weeks of each agent)

| group | agents | estimate | median error, edge | max error, edge | median error, placebo | max error, placebo |
|---|---|---|---|---|---|---|
| gamblers (X>=0.25) | 8 | reported | 27.303 | 35.730 | 93.747 | 107.295 |
| gamblers (X>=0.25) | 8 | SPR | 3.375 | 4.046 | 1.741 | 2.834 |
| gamblers (X>=0.25) | 8 | TIR | 1.357 | 1.566 | 0.054 | 0.189 |
| non-gamblers | 24 | reported | 0.000 | 0.139 | 2.854 | 2.854 |
| non-gamblers | 24 | SPR | 0.085 | 0.085 | 0.108 | 0.132 |
| non-gamblers | 24 | TIR | 0.000 | 0.001 | 0.003 | 0.006 |

Study V's single-floor audit (128 agents): largest TIR error 0.385 point (edge market) and 0.129 (placebo market); median reported-return error 68.25 points for the 28 gamblers.
