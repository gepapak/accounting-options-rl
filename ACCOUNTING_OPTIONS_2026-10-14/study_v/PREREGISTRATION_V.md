# Study V pre-registration: cheap fixes, the need for a clone, and a ledger-free audit

Written 31 October 2026, before any full training run of this study. Internal pre-registration: the files listed in
`PREREGISTRATION_HASHES_V.txt` are hashed (SHA-256, UTC time) before launch. No public registry is claimed. Every run and every
hypothesis is reported, whatever the result.

## Question and what precedes it
Studies S, T and U show that sign-twin rewards (SPR, TIR) remove the zero-floor exploit without the ledger and keep a legitimate edge. Three
objections remain after a fresh review of the manuscript:

1. **Cheaper remedies were not tested.** A trainer that sees only the reported equity can penalize ruin events (the usual "death penalty"), or
   train on a risk-adjusted (mean-variance) reward. Study S and T compared the twin rewards only with regularization toward a safe policy.
2. **The twin needs a clone of the simulator and a sign override.** Is the clone necessary?
3. **In the simple sleeve a trainer could compute the booked profit itself.** The identification result should also let an auditor
   estimate a trained agent's booked return from the reward channel alone, in a simulator whose ledger is not exposed.

This study tests (i) a **ruin-event penalty** (`RUIN<lambda>`) and a **mean-variance reward** (`MV<lambda>`), (ii) a **pooled-magnitude reward**
(PMR), a reward that needs no clone, and (iii) the **accuracy of a ledger-free audit** of every trained agent.

**Theory (exploratory, `exploratory_claude_2026-10-31/study_v_development/THEORY_PMR.md`).** Under Assumption C the reported reward is
`r = max(x, tau)`, `x = sigma*|q|*m - C`, with `m >= 0` the exogenous per-unit move. On a favourable step (`sigma = +1`) whose favourable branch
is not censored, `(r + C)/|q| = m` exactly. If the law of `m` does not depend on the agent's equity, action or observation and `m` is independent of the sign, the pooled
mean `m_bar` of these draws gives the reward `sigma*|q|*m_bar - C`, whose expectation given the state and action equals the booked one in every
state (and equals `-C` in a fair market, in every state, including those where a credit is certain). It is not exact step by step
(`|q||m_bar - m|`). A ruin-event penalty is a lump sum on the transition from positive equity to zero; it cannot be sized to the loss that is
forgiven (which grows with the position), so it is either too small, and the gamble pays, or large enough to stop risk-taking altogether, the
legitimate edge included, and at zero equity it has no direct hold. On random-action rollouts (`pmr_accuracy.txt`) the mean reward given the state and action of PMR tracks the booked one at low
equity (booked -0.312, reported +0.117, SPR -0.288, TIR -0.312, PMR -0.271 percent of K0 per quarter-hour at equity <= 0.1 K0 and leverage 16).

**Development (pilot seeds 90-92, `exploratory_claude_2026-10-31/study_v_development/`; not part of this study).** 24 pilot runs in the Study T
Part A setting (DK1) with the arms below: PMR reached the ledger-trained policy in 3 of 3 runs (booked +22.0%, +22.0%, +22.0% of K0 per
episode, ORACLE +22.0%, +22.0%, +19.9%); mean-variance rewards with weight 0.2 or more removed the exploit and almost the whole edge (booked
+2.9%, +1.0%, -7.9% at weights 0.2, 1, 4); the ruin penalty was sensitive to its strength (booked +15.1%, +16.2%, +19.4% at 1, 5 and 25, the weakest
strength still gambling in one of two runs); reported-reward PPO collapsed to a gambling policy in only one of three pilot runs (booked +2.0% against +22.0% for the other two). The ledger-free audit with the twin
estimates was within 0.10 point of the booked return of every pilot agent in both markets (largest error of the reported return 45.6 points
in the placebo market). The weight grid of the mean-variance reward was moved down to {0.05, 0.2, 1}, because weight 4 only destroys the
agent (booked -23% in one run). A pilot of Part C (one batch of 8 runs, seeds 90-91; `pilot_results_c/`): reported-reward PPO gambled at every
probe in 2 of 2 runs (booked +5.3% and -3.0% of K0 per episode, and -18.1% and -25.6% from zero equity); ORACLE, TIR and PMR trained to the same
policy (booked +22.0%, also from zero equity); the ruin penalty at strength 25 stopped the gambling at the probes but also the trading (booked
+1.8% and +11.9%, from zero equity 0.0% and +11.4%). Strengths 1 and 5 were not piloted in Part C and were added to the registered arms.
The ledger-free audit was within 0.28 point of the booked return of every pilot agent (largest error of the reported return 78.1).

## Design (`run_v.py`)
- **Markets.** The Study T edge-market sleeve (`envs_t.py`, `S_DK1`: DK1, p_edge 0.60, kappa 4), with the new market draw of Study T. Part B
  uses `T_DK2` (DK2, 0.60, 6) and `T_DK1M` (DK1, 0.58, 4). Part C (`S_DK1+Z`) changes only the training distribution of the start equity:
  half of the training episodes start at zero reported equity (an account that has already hit the floor and keeps trading), the other half
  as before (log-uniform in [0.05, 2] allocations). Evaluation windows and start equities are those of Part A.
- **Learner.** PPO as in Study T: 8 environments, n_steps 1024, batch 256, gamma 0.99, 64x64, reward normalization, 10^6 steps.
- **Arms.** PLAIN (reported reward); ORACLE (booked reward; needs the ledger); TIR (Part C only; as Study T); PMR (pooled-magnitude reward, no
  clone: reward `sigma*|q|*m_bar - own cost` with `m_bar` the running mean of `(r + cost)/|q|` over favourable steps; `rewards_v.py`);
  RUIN<lambda>: the reported reward minus lambda (percent of K0) on every step that takes the reported equity from a positive value to zero,
  lambda in {1, 5, 25}; MV<lambda>: the reported step reward `r` minus `lambda r^2` (r in percent of K0), lambda in {0.05, 0.2, 1}. None reads the
  ledger, the accounting rule or the hidden threshold; PMR uses the action, the exogenous sign of the step, the own cost and the reward;
  RUIN uses the public reported equity; MV uses the reward.
- **Part A** (`S_DK1`, seeds 150-157): PLAIN, ORACLE, PMR, RUIN1, RUIN5, RUIN25, MV0.05, MV0.2, MV1: 72 runs.
- **Part B** (`T_DK2` and `T_DK1M`, seeds 160-167): PMR: 16 runs, compared with Study T's PLAIN and ORACLE runs in the same settings
  (`study_t/runs_t.csv`, hashed before Study T ran; same test windows, so differences reflect the policies).
- **Part C** (`S_DK1+Z`, seeds 170-177): PLAIN, ORACLE, TIR, PMR, RUIN1, RUIN5, RUIN25: 56 runs.
- **Measures per run.** `X` = share of the 16 reachable low-equity probes (equity in {0, 0.02, 0.05, 0.1} allocations, both signals, share of
  the episode remaining 0.05 and 0.5) at maximal leverage (greedy policy); `E` = booked P&L in the edge market on the test split, percent of K0
  per one-week episode, mean of 100 episodes (fixed evaluation seeds 9000 and 9500); `Ez` = the same from zero start equity (exploratory).
- **Ledger-free audit (every run).** On the same 100 test episodes in the edge market and in the placebo market, the booked return of the
  trained agent is estimated from the reward channel by the sum over steps of the reported reward, SPR, TIR and PMR rewards. The twin queries
  of SPR and TIR are made at evaluation time (clone of the simulator with the exogenous sign reversed); PMR pools the favourable-step moves of
  the evaluation episodes themselves. The audit replays exactly the episodes of the test (asserted in the code against the test's booked and
  reported returns). The error of an estimate is |estimated mean - booked mean| per episode, percent of K0.

## Hypotheses and decision rules (`aggregate_v.py`)
"Edge ratio" is the mean `E` of an arm over the mean `E` of ORACLE of the same part (Part B: Study T's ORACLE runs in the setting), with a
one-sided 95% bootstrap lower bound (10,000 resamples of runs within each arm, generator seed 20261101). "Removes the exploit": `X(arm) <
X(PLAIN)` by a one-sided Mann-Whitney test, `p < 0.05`, and `mean X(arm) <= mean X(ORACLE) + 0.10`. "Both" for an arm means `mean X <= mean
X(ORACLE) + 0.10` and `mean E >= 0.80 * mean E(ORACLE)`.
- **Part A.** VA1: the exploit is present, `X(PLAIN) > X(ORACLE)`, one-sided Mann-Whitney `p < 0.05`. VA2: PMR removes it. VA3: PMR keeps the
  edge (ratio >= 0.80, lower bound >= 0.60). VA4: no tested mean-variance weight both removes the exploit and keeps the edge (the weights
  tested being 0.05, 0.2, 1). VA5: no tested ruin-penalty strength does both (1, 5, 25). *Prior belief:* VA4 holds (pilot); VA5 is uncertain, the
  strongest penalty did both on average in the pilot, so a failure of VA5 would be a finding, not an error.
- **Part B**, separately in `T_DK2` and `T_DK1M`: VB-1, PMR removes the exploit (its `X` against Study T's PLAIN and ORACLE in the setting); VB-2,
  PMR keeps the edge against Study T's ORACLE.
- **Part C.** VC1: the exploit is present when half of the training episodes start at zero equity (`X(PLAIN) > X(ORACLE)`, `p < 0.05`). VC2: PMR
  removes it. VC3: PMR keeps the edge. VC4: TIR removes it. VC5: TIR keeps the edge (ratio >= 0.80, lower bound >= 0.60). VC6: no tested ruin-penalty
  strength (1, 5, 25) does both when half of the training episodes start at zero equity. *Prior belief:* VC6 holds (pilot).
- **Part D (audit)**, over all Part A and Part C agents. VD1: the largest error of the TIR estimate is at most 1.0 point (percent of K0 per
  episode), in both the edge and the placebo market. (The threshold was set from the pilots, whose largest TIR errors were 0.10 and 0.28
  point for fully trained agents and 0.57 for 20,000-step smoke agents, against reported-return errors of up to 78 points.) VD2: among the agents with `X >= 0.25` (at least four probes at maximal leverage; needs at
  least five such agents), the median error of the reported return in the placebo market is at least 5 points and the median ratio of the TIR
  error to the reported-return error is at most 0.05.
If VA1 (VC1) fails, the other Part A (C) hypotheses are reported as not testable, the exploit not having been acquired. Seventeen hypotheses
(five in A, four in B, six in C, two in D), each with its own decision; no multiplicity correction. A missing or failed run
is reported and not replaced.

## Exploratory outputs (not registered)
`Ez` (booked return from zero equity); placebo-market reported and forgiven returns; the SPR and PMR audit errors; the audit's PMR bias (it
pools its own evaluation sample); per-run probe maps; the magnitude pool's final mean per PMR run against the training sample's true mean.

## Software validation, integrity, completion and failures
`run_v.py --smoke` (20,000 steps; every arm of Part A, Part B and Part C, one seed) checked that every arm runs end to end and that the audit
replays the test episodes; smoke results are excluded. The greedy NumPy forward pass is checked against `model.predict` for every run.
Files hashed before launch: this file, `RUN_LOG_V.md`, `rewards_v.py`, `run_v.py`, `aggregate_v.py`, and (reused unchanged from Study T)
`../study_t/envs_t.py` and `../study_t/spr_t.py`. The run log is append-only after "## After hashing".
