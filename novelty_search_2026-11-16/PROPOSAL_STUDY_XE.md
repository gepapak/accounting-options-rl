# Proposal: Study XE, cross-engine verdicts (draft for the author's approval, 16 November 2026)

**NOT registered. Nothing is hashed, and nothing has run under this protocol.** Thresholds were set from the exploratory pilots in
`README.md` (batches 1 to 4, the cross-engine pilot, the census, the reading). No seed, path seed or test seed used in those pilots may
be reused.

## Question

Two mainstream open-source backtesting engines agree exactly on every trade that stays solvent: backtesting.py (equity set to zero
and the run stopped at ruin) and backtrader (every loss booked). The study asks:

1. Do they report opposite verdicts once a trade can lose more than equity?
2. Does an optimizer of the reported return, a grid search over leverage or PPO trained through the unmodified engine, pick the
   opposite policy in each?
3. At the default settings (no leverage), does the accounting choose the direction of the trade?
4. Does the paper's reading, one query past the widest move, made before any optimization, say which engine forgives?

Why it matters: arXiv 2603.20319 (March 2026) validated five engines by comparison and concluded that "all engines agree on the
sign of every performance metric". Its engines were its own reference implementation, bt, vectorbt, Backtrader and cvxportfolio, not
backtesting.py. As described, its strategies take no short positions and no leverage. The paper's Proposition "Logs do not buy range"
says that such agreement cannot reach a kink the compared strategies never visit.

## Design

- **Engines.**
  - backtesting.py 0.6.6 and backtrader 1.9.78.123, unmodified, in `venv-vbt`.
  - gym-mtsim 2.0.0 (as published and with the floor lines deleted) and vectorbt 1.0.0 enter only A and D.
- **Market.** Fair placebo paths: 88 daily bars, gym-mtsim's bundled EURUSD magnitudes (A to C) or its 2021 BTC/ETH/BCH magnitudes x2
  (E), fair signs, fresh seeds.
- **A. Trade census.**
  - The trade: the largest position at leverage L in {1, 2, 5, 10, 20, 50, 100, 200}, bought on bar 1 and held to bar 86, with a fee
    of 2 bp per side.
  - Paths: 2,000 fresh ones.
  - References: LEDGER and FLOOR+STOP.
- **B. Optimizer.** The L that maximizes the mean reported return, with 4,000 bootstrap resamples.
- **C. PPO through the engine, two actions (flat or 200x).**
  - The wrapper `rl_engine_env.py` exchanges each bar with the engine running in a thread. The reward is the change in the engine's
    reported equity / 10,000.
  - Episodes start at equity log-uniform in [0.05, 2] x 10,000 USD. PPO as in the paper, 507,904 steps, **16 fresh seeds per engine**.
- **D. Reading.** One bar past the kink and one inside the range, per engine, plus backtesting.py's unlevered short.
- **E. Direction at the default settings.**
  - Margin 1, actions {short, flat, long}, crypto magnitudes x2.
  - PPO as in C, **12 fresh seeds per engine**.

## Decisions (thresholds from the pilots)

| Decision | Prediction | Pilot evidence |
|---|---|---|
| XE1 Agreement inside the range | For L <= 5, the two engines' reported returns are identical on every path (abs. difference below 1e-9 of the cash) | identical on 2,000 paths at L 1, 2, 5 |
| XE2 Divergence past the kink | At L = 200: backtesting.py's mean reported return exceeds backtrader's, and exceeds its own trade records (95% bootstrap lower bounds > 0), and backtrader equals the ledger (abs. difference below 1e-6 points) | +26.0 [11.7, 40.9] against -7.4; forgiven +37.4 |
| XE3 Optimizer choice | The mean-reported-return optimizer picks L = 200 in backtesting.py in at least 95% of resamples, and does not pick L = 200 in backtrader in at least 60% | 100%; 70% |
| XE4 Reading | One query past the kink classifies the engines (backtesting.py, including the unlevered short, and gym-mtsim as published forgive; backtrader, vectorbt and gym-mtsim without the floor book), and every query inside the range agrees | as predicted |
| XE5 PPO learns the engine | At least 9 of 16 backtesting.py seeds choose 200x at half or more of the low-equity probes; at most 1 of 16 backtrader seeds do; Fisher exact one-sided p < 0.01 | 3 of 4 against 0 of 4 (two actions) |
| XE6 Reported against booked for the learned policies | On 200 fresh test paths from 10,000 and from 1,000 USD: for backtesting.py-trained agents, reported minus booked > 0 (seed-cluster bootstrap lower bound > 0); for backtrader-trained agents, abs(reported - booked) < 1e-6 points on every path | +28.4 [25.9, 31.0]; 0 |
| XE7 Direction chosen by the accounting | At least 7 of 12 backtesting.py seeds are short at half or more of all probes, and at most 1 of 12 is long; the direction index (share short minus share long) is higher in backtesting.py than in backtrader (one-sided Mann-Whitney p < 0.05) | 3 of 4 short, 0 long; backtrader 1 short, 1 long |

## Lessons from the pilots that shaped the thresholds

- **Use two actions at L = 200 for XE5.** With six actions only 4 of 10 seeds learned, because the losing intermediate levels hide the
  payoff. With two actions 3 of 4 did.
- **Compare the engines at the same leverage; do not predict from the sign of the net edge.** In batches 5 and 6, backtesting.py agents
  took 20x and 50x in 2 of 4 seeds although the net edge at 20x was slightly negative, while backtrader agents stayed flat. Only the
  contrast was robust (11 of 16 against 1 of 16 over four leverages).
- **An optional XE8, a dose-response over L in {20, 50, 100, 200} with backtrader controls at each L,** would need about 64 more runs
  (about 3 to 4 hours), so it does not fit the same night as XE5 and XE7.

## Compute

- A, B and D take minutes.
- C: 32 runs of about 35 minutes, eight at a time, about 2.3 hours.
- E: 24 runs, about 1.7 hours.
- In all, about 4 to 4.5 hours, so it fits one night.

## What would change if it fails

- **XE5 or XE7 fails:** learning the engine's accounting is not reliable at this budget. The cross-engine verdicts (XE1 to XE4) still
  stand as software facts.
- **XE1 to XE4 fail:** that would contradict the pilots, which reproduce exactly, so it would point to a setup error.
