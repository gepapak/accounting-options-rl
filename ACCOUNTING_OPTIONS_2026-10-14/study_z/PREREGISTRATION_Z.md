# Study Z pre-registration: does splitting the account blind the cheap remedy, and do the twin rewards still work?

Written 3 November 2026, before any full run of this study. Internal pre-registration: the files listed in `PREREGISTRATION_HASHES_Z.txt`
are hashed (SHA-256, UTC time) before launch. No public registry is claimed. Every run and every hypothesis is reported, whatever the result.

## Question and what precedes it
A lump-sum ruin penalty (1, 5 or 25 percent of the allocation on every step into zero equity) is the strongest cheap competitor of the twin
rewards: strength 5 removed the exploit and kept the edge in all four one-account settings of Studies V and W. Simulators and settlement
systems, however, often keep several accounts with their own floors (sub-accounts per asset or market, isolated margin, separate legal entities,
several balance responsible parties in one group, each with its own collateral). A trainer that sees the portfolio but not its account structure
would measure ruin on the portfolio. Study Z asks whether that remedy still works when the same losses are floored per account, and whether the
twin rewards, which read only the portfolio's reward channel, a sign-reversed clone and the own cost, still remove the exploit.

**Theory (`THEORY_Z.md`, exact checks in `test_envs_z.py`, both written before the pilot results were read).** Split floors forgive at least
what a pooled floor forgives (a portfolio of options is worth at least an option on the portfolio). From two empty accounts, opposite positions
report the winning leg's net gain for either sign of the move and book minus both costs. A step after which some account keeps positive equity
never takes the portfolio to zero, and a step that trades only from empty accounts has a non-negative portfolio reward, so a policy that keeps
one account funded and flat never incurs a portfolio ruin penalty, and one that trades only from empty accounts no loss or drawdown penalty
either, whatever the strength. The
portfolio accounting stays forgiving, so the twin bracket holds, and under conditional sign fairness SPR and TIR have expectation equal to minus
the own cost for every action, whatever the account structure. 244,000 exact checks of these statements passed. A scripted witness
(`witness_z.py`, no learning, run before any pilot result was read) shows the incentive on the registered test weeks: from zero equity a
"park and gamble" policy (a hedge from two empty accounts, otherwise the richer account flat and the other at maximal leverage) incurs no
portfolio ruin, reports +33.0% of the allocation per week in the placebo market while booking -164.1%, and keeps its reward under the
portfolio penalty at any strength, while the account penalty at 5 gives it -282.1; in the edge market from zero equity the portfolio penalty at
25 scores it +53.5 against -52.8 for the honest optimum. From funded starts the honest optimum scores higher, so the incentive lies at empty
and low accounts, where the probes are taken.

**Development (pilot seed 99, exploratory; not part of this study).** One run per arm for eight arms (10^6 steps, `pilot_results/`,
16:37-17:00, 21-22 minutes per run with 8 workers). Reported-reward PPO gambled (`X` 0.91; maximal leverage at every both-low probe and in
the empty account at 81% of the one-empty probes, the funded account's leg small) and booked -17.4% of the allocation per week against +22.9%
for ledger training; in the placebo market it reported +0.6% and booked -140.7%. The portfolio ruin penalty did not remove it: `X` 0.94 at
strength 5 and 0.59 at strength 25, where the agent incurred no portfolio ruin in the placebo test (20.2 account ruins per week), gambled
from the empty account and otherwise stayed nearly flat (booked -14.0%). The account penalty at 5 removed it (`X` 0, booked +28.2%), and so did
SPR (`X` 0, +29.7%). Pooled control: reported reward `X` 1.00 (both legs at maximal leverage in the same direction), penalty 5 `X` 0 (+29.6%).
The learned form was the park-and-gamble of the theory, not the hedge from two empty accounts (hedge share 0 in every run). The twin bracket
contained the booked P&L of every audited step; the portfolio-level TIR audit of the reported-reward gambler was within 0.03 points of its
booked placebo return (SPR 2.4, reported return 141). The pilot changed the design in three ways, all before hashing: the portfolio penalty at
strength 1 was dropped (it did not remove the exploit even in one account, Study V Part C, so it cannot test splitting; 72 runs instead of 80);
the mechanism hypothesis ZB2, first written as "portfolio ruins at most a tenth of account ruins" (the strength-5 pilot run had 6.7 and 51.0),
was replaced by a comparison with the reported-reward arm; and the exploit's form was added as hypothesis ZB3.

## Design (`run_z.py`, `envs_z.py`, `rewards_z.py`)
- **Environment.** The Study T edge-market sleeve (`S_DK1`: DK1, `p_edge` 0.60, kappa 4, the Study T market draw) with two accounts A and B
  that trade the same spread, each with its own zero floor (`SplitEnv`), and a pooled control in which the same two legs are written to one
  account with one floor (`PooledEnv`). The agent chooses a leverage per account (`MultiDiscrete([9, 9])` over the sleeve's grid); each leg is
  sized as in the sleeve (fixed sizing, liquidity cap per leg) and pays its own cost `fee*|q| + kappa*q^2`; the ledger books both legs. The
  agent observes both accounts' reported equities, the signal and the share of the episode left (the pooled account shows its equity as two
  equal halves). Half of the training episodes start with every account at zero; the others with a total drawn log-uniformly between 0.05 and
  2 allocations, split equally.
- **Learner.** PPO as in Studies V and X: 8 environments, n_steps 1024, batch 256, gamma 0.99, 64x64, reward normalization, 10^6 steps; the
  multi-discrete policy has one categorical head per account.
- **Arms (split).** REP (the portfolio's reported reward), LEDGER (booked reward; reads the ledger), RUINP5 and RUINP25 (Study V's ruin
  penalty at 5 and 25 percent of the allocation, the strengths that removed the exploit in one account, measured on the portfolio's reported
  equity), RUINA5 (the same penalty at 5, measured on every account; needs the account structure), SPR and TIR (portfolio reward channel,
  sign-reversed clone of the whole portfolio, own cost; never the ledger or the account structure). **Arms (pooled control).** P_REP
  (reported reward) and P_RUIN5 (ruin penalty 5 on the one account, which is the portfolio).
- **Seeds.** 210-217 per arm: 9 arms x 8 seeds = 72 runs. A missing or failed run is reported and not replaced.
- **Measures per run (greedy policy).** Probes: (a) both accounts low, total equity in {0, 0.02, 0.05, 0.1} allocations split equally, both
  signals, share of the episode left 0.05 and 0.5 (16 probes); (b) one account empty and the other at 0.5 or 2 allocations, either account
  empty, both signals, both times (16 probes). `X_sym` = share of probes (a) at which some leg is at maximal leverage (16); `X_empty` = share of
  probes (b) at which the empty account's leg is at maximal leverage; the exploit share `X` = mean of `X_sym` and `X_empty` for split arms and
  `X_sym` for the pooled arms (their probes (b) are unreachable). `E` = booked P&L of both legs in the edge market on the test split, percent of
  K0 per one-week episode, mean of 100 episodes (start equity log-uniform between 0.05 and 2 allocations, split equally). Placebo test: reported,
  booked and forgiven returns and the numbers of portfolio and account ruin events per episode. Ledger-free audit as in Study X (sums of the
  reported, SPR and TIR step rewards and of the twin bracket's lower end over the 100 test episodes, both markets, twin queries at evaluation
  time only), with the bracket checked on every step and episode.

## Hypotheses and decision rules (`aggregate_z.py`)
"Present": `X(arm) > X(LEDGER)` by a one-sided Mann-Whitney test, `p < 0.05`, and `mean X(arm) >= mean X(LEDGER) + 0.10`. "Removes the
exploit": `X(arm) < X(REP)` by a one-sided Mann-Whitney test, `p < 0.05`, and `mean X(arm) <= mean X(LEDGER) + 0.10`. "Keeps the edge": mean `E`
of the arm over mean `E` of LEDGER at least 0.80, with a one-sided 95% bootstrap lower bound (10,000 resamples of runs within each arm, generator
seed 20261104) of at least 0.60.
- **ZA1.** The exploit is present in the split sleeve: REP present.
- **ZA2, ZA3.** The portfolio ruin penalty does not remove it at strength 5 or 25: RUINP5 present, RUINP25 present (two decisions).
- **ZA4.** The same penalty measured on every account removes it: RUINA5 removes the exploit.
- **ZA5.** SPR removes it. **ZA6.** SPR keeps the edge.
- **ZA7.** TIR removes it. **ZA8.** TIR keeps the edge.
- **ZA9.** Pooled control: the exploit is present with one floor, `X_sym(P_REP)` present against `X_sym(LEDGER)`.
- **ZA10.** Pooled control: the ruin penalty removes it there, `X_sym(P_RUIN5) < X_sym(P_REP)` (one-sided Mann-Whitney, `p < 0.05`) and
  `mean X_sym(P_RUIN5) <= mean X_sym(LEDGER) + 0.10`.
- **ZB1.** For all 72 runs, both markets, the twin bracket contains the booked P&L of every audited step and the booked return of every audited
  episode (Proposition 10 applied to split floors).
- **ZB2.** Mechanism of ZA2 and ZA3: per run, the placebo test's portfolio ruins per account ruin (runs with at least one account ruin) are
  lower in the RUINP5 and RUINP25 runs, pooled, than in the REP runs (one-sided Mann-Whitney, `p < 0.05`): the portfolio penalty teaches the
  agent to avoid ruining the portfolio, not to avoid ruining an account.
- **ZB3.** The exploit's form: among the REP, RUINP5 and RUINP25 runs with `X >= 0.25`, the mean park share (one-empty probes at which the empty
  account's leg is at maximal leverage and the funded account's is not) is at least 0.50.
If ZA1 fails, ZA2-ZA5 and ZA7 are reported as not testable; if ZA9 fails, ZA10 is not testable. Thirteen hypotheses, each with its own
decision; no multiplicity correction.

*Prior belief (written before the full runs, after the pilot):* ZA1 holds (pilot `X` 0.91; 0.91 in Study V's one-account sleeve with zero
starts). ZA2 very probably holds (pilot 0.94): by part 3 of the proposition the penalty can be avoided entirely, and the pilot agent still
incurred some portfolio ruins without losing the exploit. ZA3 is the least certain: the pilot agent kept the exploit at empty accounts
(`X` 0.59) and stopped trading elsewhere, but other seeds may stop trading altogether, as strength 25 did from zero equity in one account
(Study V Part C). ZA4 probably holds (pilot 0; Study V's strength 5 at the account level). ZA5-ZA8 probably hold (pilot SPR `X` 0 with
+29.7% against +22.9%; TIR was not piloted, but its portfolio-level identification is exact when one account trades and fair-unbiased
otherwise); ledger-trained PPO here learned well below the honest optimum (+44% per week for leverage 4 with the signal in both accounts,
`witness_z.py`), so edge ratios above one would reflect that, not an advantage. ZA9 and ZA10 hold (pilot 1.00 and 0). ZB1 holds by the
proposition; a failure would be an implementation error. ZB2 and ZB3 probably hold (pilot ratios 0.29 for REP against 0.13 and 0 for the
penalties; park shares 0.81, 0.88 and 0.69).

## Exploratory outputs (not registered)
The hedge share (both legs at maximal leverage in opposite directions at the both-low probes) and the same-direction share; booked return
from zero equity; placebo reported, booked and forgiven returns; the audit errors of SPR, TIR and the reported return and the bracket's width
for the gambling runs (`X >= 0.25`).

## Software validation, integrity and failures
`test_envs_z.py` (244,000 exact checks, no failure) and `run_z.py --smoke` (20,000 steps, one seed per arm; output excluded) ran before the
pilot. The greedy NumPy forward pass is checked against `model.predict` for every run and the audit asserts that it replays the test episodes.
Files hashed before launch: this file, `THEORY_Z.md`, `RUN_LOG_Z.md`, `envs_z.py`, `rewards_z.py`, `run_z.py`, `aggregate_z.py`,
`test_envs_z.py`, `witness_z.py`, `witness_z.json`, and (reused unchanged) `../study_t/envs_t.py`. The run log is append-only after
"## After hashing".
