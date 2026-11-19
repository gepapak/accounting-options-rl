# Second set of predictions (compounded prices), written after Tests A and B were scored and before Tests A2, A3, C and D ran

Generated 2026-11-17T01:50:40 by `predict_v2.py` from inputs only. Exploratory, not registered.

Why a second set: in Test A the additive forms were 12% high at L = 20 in all three rules, TERM included, because the simulated prices compound. The time-step content passed (slopes -0.47 to -0.49; ZF / TERM within 1%). The forms below use the log price.

Overshoot constant of the log price, by far-barrier Monte Carlo: EURUSD 0.6151, crypto 1.0008 (simple returns: 0.6180 and 0.8176).

## C. Unmodified backtesting.py 0.6.6 at finer time steps (BLIND: never run in any form)

| Fee per side | L | m | P(ruin) | Forgiven (% of equity) |
|---|---|---|---|---|
| 0 bp | 20 | 1 | 0.0907 | 0.338 |
| 0 bp | 20 | 4 | 0.0968 | 0.181 |
| 0 bp | 20 | 16 | 0.1000 | 0.093 |
| 0 bp | 50 | 1 | 0.4834 | 4.650 |
| 0 bp | 50 | 4 | 0.5031 | 2.420 |
| 0 bp | 50 | 16 | 0.5131 | 1.234 |
| 0 bp | 200 | 1 | 0.8262 | 32.278 |
| 0 bp | 200 | 4 | 0.8504 | 16.611 |
| 0 bp | 200 | 16 | 0.8626 | 8.424 |
| 2 bp | 20 | 1 | 0.0919 | 0.378 |
| 2 bp | 20 | 4 | 0.0981 | 0.220 |
| 2 bp | 20 | 16 | 0.1014 | 0.133 |
| 2 bp | 50 | 1 | 0.4874 | 5.166 |
| 2 bp | 50 | 4 | 0.5072 | 2.936 |
| 2 bp | 50 | 16 | 0.5172 | 1.751 |
| 2 bp | 200 | 1 | 0.8311 | 35.777 |
| 2 bp | 200 | 4 | 0.8554 | 20.112 |
| 2 bp | 200 | 16 | 0.8675 | 11.925 |

Criteria: in every cell, the engine's forgiven amount within 10% of the prediction, or the prediction inside the 95% bootstrap interval; ruin share within 0.03; fee-free log-log slope against m between -0.56 and -0.42.

## A2. Additive profit and loss (the paper's ledger model), additive forms: the table of PREDICTIONS.md section A, unchanged

Criterion: every LL, TERM and ZF cell within 5% of the simulated value.

## A3 and B3. Compounded prices, the cells of Tests A and B (post hoc: their simulated values were known)

| Case | L | m | P(ruin) | LL | TERM | ZF |
|---|---|---|---|---|---|---|
| Gaussian | 20 | 1 | 0.0907 | 0.338 | 1.284 | 2.195 |
| Gaussian | 20 | 4 | 0.0969 | 0.181 | 1.284 | 2.368 |
| Gaussian | 20 | 16 | 0.1001 | 0.093 | 1.284 | 2.459 |
| Gaussian | 20 | 64 | 0.1017 | 0.047 | 1.284 | 2.505 |
| Gaussian | 50 | 1 | 0.4835 | 4.651 | 24.175 | 43.219 |
| Gaussian | 50 | 4 | 0.5031 | 2.420 | 24.175 | 45.548 |
| Gaussian | 50 | 16 | 0.5131 | 1.234 | 24.175 | 46.748 |
| Gaussian | 50 | 64 | 0.5182 | 0.623 | 24.175 | 47.356 |
| Gaussian | 200 | 1 | 0.8262 | 32.281 | 201.914 | 367.639 |
| Gaussian | 200 | 4 | 0.8504 | 16.613 | 201.914 | 383.647 |
| Gaussian | 200 | 16 | 0.8626 | 8.425 | 201.914 | 391.823 |
| Gaussian | 200 | 64 | 0.8687 | 4.242 | 201.914 | 395.954 |
| eurusd | 20 | 1 | 0.0901 | 0.355 | 1.284 | 2.177 |
| eurusd | 50 | 1 | 0.4813 | 4.889 | 24.175 | 42.965 |
| eurusd | 100 | 1 | 0.7023 | 14.413 | 80.409 | 144.815 |
| eurusd | 200 | 1 | 0.8235 | 33.972 | 201.914 | 365.881 |
| crypto | 2 | 1 | 0.2657 | 1.639 | 4.304 | 6.323 |
| crypto | 5 | 1 | 0.7038 | 17.373 | 58.643 | 88.832 |
| crypto | 10 | 1 | 0.8303 | 46.114 | 167.312 | 254.508 |

Criteria as before: 5% (Gaussian), 8% (EURUSD), 25% (crypto).

## D1. The registered Study XE census, compounded-price forms with the fee term (measured values were known)

| L | P(ruin) | Forgiven (% of equity) |
|---|---|---|
| 20 | 0.0913 | 0.394 |
| 50 | 0.4852 | 5.403 |
| 100 | 0.7070 | 15.907 |
| 200 | 0.8284 | 37.468 |
