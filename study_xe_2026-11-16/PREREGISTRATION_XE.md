# Study XE pre-registration: cross-engine verdicts

Prepared 16 November 2026 in an AI-assisted session (Claude Opus 5.5 in Claude Code), after the exploratory pilots in
`../novelty_search_2026-11-16/`. The pilots, their seeds and their results are disclosed there; this study reuses none of their
seeds. The author approved registration on 16 November 2026 at about 20:45 local time. `hash_prereg_xe.py` writes
`PREREGISTRATION_HASHES_XE.txt`, whose timestamp is the registration time, before any run below.

## Question

Two mainstream open-source backtesting engines agree exactly on every trade that stays solvent: backtesting.py 0.6.6 (equity set to
zero and the run stopped at the first close with equity at or below zero) and backtrader 1.9.78.123 (every loss booked). The study
asks:

1. Do they report opposite verdicts once a trade can lose more than equity?
2. Does an optimizer of the reported return, a grid search over leverage or PPO trained through the unmodified engine, pick the
   opposite policy in each?
3. At the default settings (no leverage), does the accounting choose the direction of the trade?
4. Does the paper's one-query reading say which engines forgive?

Context: arXiv 2603.20319 (March 2026) compared five other engines on strategies without shorts or leverage, and found that they agree
on the sign of every metric. The paper's Proposition "Logs do not buy range" predicts that such agreement says nothing past the kink.

## Pilot disclosure

The pilots used:
- path seed 20261116 and test seed 20261117;
- PPO seeds 1 to 10, 21 to 24, 31 to 34 and 41 to 44.

The pilot results that shaped the thresholds are XE1 to XE3 (2,000 paths), XE4 (the reading), XE5 (two actions: 3 of 4 against 0 of 4),
XE6 (+28.4 [25.9, 31.0]) and XE7 (3 of 4 short, 0 long, against 1 and 1). Two pilot predictions failed (batches 5 and 6, in the pilot
README); they are not tested here.

## Design

| Part | What | Fresh seeds |
|---|---|---|
| A | Trade census: the largest position at L in {1, 2, 5, 10, 20, 50, 100, 200}, bought on bar 1 and held to bar 86, fee 2 bp per side, 2,000 fair placebo paths (88 daily bars, gym-mtsim's bundled EURUSD magnitudes, fair signs); references LEDGER and FLOOR+STOP | path seed 20261201 |
| B | Optimizer: the L that maximizes the mean reported return, 4,000 bootstrap resamples of paths | bootstrap seed 20261205 |
| C | PPO through the unmodified engines, two actions {0, 200}, margin 1/200; the paper's standard PPO, 507,904 steps; 16 seeds per engine; probes at a flat account at equity {0.05, 0.1, 0.25, 0.5, 1, 2} x 10,000 USD and 25/50/75% of the episode left; 200 test paths from 10,000 and 1,000 USD | seeds 101 to 116; test seed 20261203 |
| D | Reading: one bar inside the range and one past the kink, per engine (backtesting.py levered and an unlevered 1x short; backtrader; vectorbt 1.0.0 short); plus gym-mtsim 2.0.0 as published and floor-deleted, on 300 fair paths | gym-mtsim path seed 20261202 |
| E | Direction at the default settings: margin 1, actions {-1, 0, 1}, placebo with the bundled 2021 BTC/ETH/BCH daily magnitudes x2; PPO as in C; 12 seeds per engine | seeds 201 to 212; test seed 20261204 |

The exact commands are in `run_xe.ps1`.

## Decisions (computed by `aggregate_xe.py`)

| Decision | Supported if |
|---|---|
| XE1 | For L <= 5, max abs(backtesting.py reported - backtrader reported) over all paths < 1e-9 (in return units) |
| XE2 | At L = 200: the 95% bootstrap lower bound of mean(backtesting.py - backtrader) > 0, and of mean(backtesting.py reported - its own trade records) > 0; and max abs(backtrader - LEDGER) < 1e-8 |
| XE3 | The argmax over L of the mean reported return is 200 in at least 95% of resamples for backtesting.py, and is not 200 in at least 60% for backtrader |
| XE4 | Past the kink, backtesting.py (levered, and the unlevered short) forgives more than 1e-6 of the cash, while backtrader and vectorbt forgive less than 1e-6 of the cash; inside the range every engine forgives less than 1e-6 of the cash; gym-mtsim as published forgives > 0 on average at L = 200; floor-deleted, max abs(reported - booked) < 1e-9 at every L |
| XE5 | Learner = a seed that chooses 200 at half or more of the 9 low-equity probes (equity <= 0.25). At least 9 of 16 backtesting.py seeds and at most 1 of 16 backtrader seeds; Fisher exact one-sided p < 0.01 |
| XE6 | From both 10,000 and 1,000 USD: for backtesting.py seeds, the seed-cluster bootstrap 95% lower bound of mean(reported - booked) > 0; for backtrader seeds, max abs(reported - booked) over all test paths < 1e-6 percentage points |
| XE7 | At least 7 of 12 backtesting.py seeds are short at half or more of the 18 probes, and at most 1 of 12 is long at half or more; direction index (share short - share long) higher in backtesting.py than in backtrader, one-sided Mann-Whitney p < 0.05 |

Every decision is reported, whatever its outcome. A failure of XE5 or XE7 would mean that learning the engine's accounting is not
reliable at this budget. XE1 to XE4 would stand as software facts.

## Compute and environment

- C takes about 32 runs of 35 minutes, eight at a time (about 2.3 hours); E takes about 24 runs (about 1.7 hours).
- Environments: `venv-vbt` (Python 3.10.0, backtesting 0.6.6, backtrader 1.9.78.123, vectorbt 1.0.0, SB3 2.7.0, torch 2.8.0+cpu,
  gymnasium 1.2.0, numpy 1.26.4) and `venv-mtsim` (gym-mtsim 2.0.0).
