# Study Y pre-registration: do the twin rewards transfer to a second learner (A2C)?

Written 3 November 2026, before any full run of this study. Internal pre-registration: the files listed in `PREREGISTRATION_HASHES_Y.txt`
are hashed (SHA-256, UTC time) before launch. No public registry is claimed. Every run and every hypothesis is reported, whatever the result.

## Question and what precedes it
Every edge-market test of SPR and TIR so far trained PPO (Studies S, T, V, W, X); Study S also trained SPR with A2C, but only in a fair market,
where there is no edge to keep. A referee can therefore ask whether "removes the exploit and keeps the edge" is a property of the rewards or
of PPO. This study trains a second on-policy learner, A2C, with the configuration already used in Studies I, Q and S, in the two sleeves where
PPO was tested with zero-equity starts: the single-floor sleeve (Study V Part C) and the two-write sleeve (Study X), with the same arms, probes,
test windows, start equities, audit and decision rules as Study X.

## Development (pilot seed 99, exploratory; not part of this study)
`run_y.py --pilot` (A2C, 10^6 steps, one run per arm; `pilot_results/`, 07:34-07:50). In the single-floor sleeve *without* zero starts
(`S_DK1`), all four A2C runs learned the same policy (leverage 4 in the direction of the signal at every probe; X = 0; booked +22.0% per
episode): the reported-reward run did not gamble, unlike PPO in Study T (24 of 30 reported-reward agents with X > 0). In the two-write sleeve
with zero starts (`X_DK1+Z`) the reported-reward run gambled (X = 0.50; booked -0.9% per episode, -6.9% from zero equity; 50% of the allocation
forgiven per placebo episode; TIR audit error 0.45 point), while ORACLE, SPR and TIR all booked +22.0% with no probe at maximal leverage
(TIR audit error at most 0.003 point in both markets). The pilot changed the design: the
single-floor main setting uses zero starts (Study V Part C's `EdgeEnvZ`) so that insolvent accounts are visited, and the no-zero-start
sleeve is kept as an acquisition check (PLAIN and ORACLE only). No decision rule was changed; they are Study X's.

## Design (`run_y.py`)
- **Learner.** A2C (stable-baselines3) "A2C-n64" as in Studies I, Q and S: rollout 64 steps, GAE 0.95, advantage normalization, other
  hyperparameters at their defaults; 8 environments, reward normalization, gamma 0.99, 64x64, 10^6 steps. The greedy NumPy forward pass is
  checked against `model.predict` for every run.
- **Settings.** `S_DK1+Z`: Study T's single-floor edge-market sleeve (DK1, p_edge 0.60, kappa 4, the Study T market draw), half of the training
  episodes at zero reported equity (`../study_v/rewards_v.EdgeEnvZ`), probes and evaluation in `../study_t/envs_t.EdgeEnvT`. `X_DK1+Z`: the
  two-write sleeve of Study X (`../study_x/envs_x.EdgeEnvMFZ` in training, `EdgeEnvMF` in probes and evaluation). `S_DK1`: the single-floor
  sleeve without zero starts (`EdgeEnvT` throughout), PLAIN and ORACLE only.
- **Arms.** PLAIN (reported reward), ORACLE (booked reward; reads the ledger), SPR and TIR (`../study_t/spr_t.py`; the reported rewards of the
  real step and of a sign-reversed clone and the own cost, never the ledger).
- **Seeds.** 200-207: 2 settings x 4 arms x 8 seeds + 2 arms x 8 seeds = 80 runs. A missing or failed run is reported and not replaced.
- **Measures per run.** As Study X: `X` = share of the 16 reachable low-equity probes at maximal leverage (greedy policy); `E` = booked P&L in the
  edge market on the test split, percent of K0 per one-week episode, mean of 100 episodes; ledger-free audit (sum of the reported, SPR and TIR
  step rewards over the same 100 test episodes, twin queries at evaluation time only).

## Hypotheses and decision rules (`aggregate_y.py`)
Study X's definitions. "Removes the exploit": `X(arm) < X(PLAIN)` by a one-sided Mann-Whitney test, `p < 0.05`, and `mean X(arm) <=
mean X(ORACLE) + 0.10`. "Keeps the edge": mean `E` of the arm over mean `E` of ORACLE at least 0.80, with a one-sided 95% bootstrap lower bound
(10,000 resamples of runs within each arm, generator seed 20261103) of at least 0.60. Separately in `S_DK1+Z` (suffix -S) and `X_DK1+Z` (-X):
- **YA1.** The exploit is acquired by A2C: `X(PLAIN) > X(ORACLE)`, one-sided Mann-Whitney `p < 0.05`.
- **YA2.** TIR removes it. **YA3.** TIR keeps the edge. **YA4.** SPR removes it. **YA5.** SPR keeps the edge.
- **YB1.** For the ORACLE, SPR and TIR agents (24 per setting), the largest error of the TIR estimate of the 100-episode booked return is at
  most 1.0 point (percent of K0 per episode) in the edge market and in the placebo market.
If YA1 fails in a setting, YA2-YA5 of that setting are reported as not testable. In `S_DK1`:
- **YC1.** A2C acquires the exploit without zero starts: `X(PLAIN) > X(ORACLE)`, one-sided Mann-Whitney `p < 0.05`.
Thirteen hypotheses, each with its own decision; no multiplicity correction (a Holm sensitivity analysis is reported as exploratory).

*Prior belief (written before the runs):* YA1-X holds (the pilot's reported-reward run gambled); YA1-S probably holds (zero starts made PPO
gamble in Study V Part C). YA2-YA5 probably hold in both settings: SPR and TIR remove the credit that makes gambling pay, whatever the
learner, and the honest A2C runs of the pilot all reached the same booked return. YB1 probably holds (TIR is exact in the single-floor sleeve;
in the two-write sleeve honest agents rarely reach an account that cannot pay the fee). YC1 probably fails (the pilot's reported-reward run did
not gamble without zero starts), which would mean that acquisition by A2C depends on visiting insolvent accounts in training.

## Exploratory outputs (not registered)
Booked edge from zero equity; placebo-market reported and forgiven returns; audit errors of the gambling agents (`X >= 0.25`); the A2C results
next to PPO's in the same sleeves (Study V Part C, Study X); a Holm adjustment of the thirteen p-values and decisions.

## Software validation, integrity and failures
`run_y.py --smoke` (20,000 steps; PLAIN and TIR, seed 99, two sleeves) ran end to end and its output is excluded; the pilot is exploratory.
Files hashed before launch: this file, `RUN_LOG_Y.md`, `run_y.py`, `aggregate_y.py`, and (reused unchanged) `../study_t/envs_t.py`,
`../study_t/spr_t.py`, `../study_x/envs_x.py` and `../study_v/rewards_v.py`. The run log is append-only after "## After hashing".
