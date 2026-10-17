# Study O pre-registration: can a quantity computed before training predict whether RL agents learn a reward exploit?

Written 14 October 2026, before any full training run of this study. Internal pre-registration: the files listed in
`PREREGISTRATION_HASHES_O.txt` are hashed (SHA-256, UTC time) before the full run is launched.

## Question
Studies B–N show that PPO agents learn to exploit the zero floor when the exploit is large, and do not when it is
small (Study H's costs, southern Norway). Can the outcome be predicted before training? Study O tests a learnability
index computed from the dynamic program (DP) alone, on new conditions in which the exploit is shrunk continuously by
booking a share alpha of every forgiven loss (partial liability, Corollary 3 of the paper).

## The learnability index (fixed before training)
For a condition, solve the DP of `dp_full.py` (full training distribution, 2,000 equal-probability bins per leverage
level, gamma 0.99, equity grid to 6 allocations). Let A(e) = Q(e, 16) - Q(e, 0), the action gap of maximal leverage
over staying flat, read at e in {0.02, 0.1, 0.25, 0.5, 1, 1.5, 2} allocations and interpolated linearly in log e. Then

    index = 100 * mean_{e in 400 log-spaced points of [0.05, 2]} max(A(e), 0) / sigma16   (percent)

where sigma16 is the standard deviation of the binned one-quarter-hour P&L at leverage 16 (the reward noise), and
[0.05, 2] with log-uniform weights is the training-start distribution of equity. Code: `dp_o.learnability_index`.

**Disclosure.** The index formula was chosen after the retrospective analysis of Studies F, H, J and K (14 October
2026) had been seen: those studies are the calibration set, not a test. The test is the prospective set below.

## Calibration (completed studies; `calibrate_o.py`, `calibration_o.json`)
Ten conditions with equity as the only input and a matched full-liability control (DK1/DK2 at gamma 0.95, 0.99, 0.995;
DK1/DK2 under Study H's costs; FI and NO2). Learned = the floor-trained agents' mean |leverage| at 2% equity exceeds the
controls' by at least 4 units (G >= 4).
- Learned: index 1.539–1.821% (7 conditions, G = 8.2 to 14.8).
- Not learned: index 0.138–0.171% (3 conditions, G = -0.05 to 0.9).
- Registered threshold: c* = sqrt(0.171 x 1.539) = **0.513%**.
Study K trained with initial equity in [0.05, 6] allocations; the index uses [0.05, 2] for every condition.

## Design
- **Environment** (`envs_o.py`): Study F's placebo sleeve (sign-randomized Danish spreads, fixed sizing, liquidity cap,
  fee 0.10 EUR/MWh, allocation 20,000 EUR, leverage levels ±{0, 0.25, 1, 4, 16} x 0.25 MWh, training equity
  log-uniform in [0.05, 2] allocations). The account keeps trading under the zero floor; a share alpha of every
  forgiven loss is charged in the reward: reward = [(1 - alpha)(e' - e) + alpha * pnl] / K0 * 100.
- **Arms**: alpha in {0, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95} and the matched control alpha = 1 (every loss charged,
  continued trading, as Study L's full charge).
- **Learner** (`run_o.py`): PPO, Stable-Baselines3 2.7.0, equity as the only input, 8 parallel environments, rollout
  1,024, batch 256, network 64x64, reward normalization, gamma 0.99, 10^6 steps; seeds 30–34; DK1 and DK2.
  80 runs in total.
- **Measurements**: greedy |leverage| at equity probes {0, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3}
  allocations; 20 placebo sign draws of the TEST period from one allocation (reported and booked returns).

## Predicted indices (`dp_o.py`, `dp_o_DK1.json`, `dp_o_DK2.json`; mean of the two zones)

| alpha | index (%) | DP region where leverage 16 is optimal (allocations) | prediction |
|---|---|---|---|
| 0 | 1.805 | 1.75 | learned (confident) |
| 0.25 | 1.345 | 1.65 | learned (confident) |
| 0.5 | 0.886 | 1.52 | learned (confident) |
| 0.7 | 0.518 | 1.35 | learned (transition arm: within a factor 1.5 of c*) |
| 0.8 | 0.335 | 1.21 | not learned (confident) |
| 0.9 | 0.154 | 0.97–1.00 | not learned (confident) |
| 0.95 | 0.064 | 0.73–0.77 | not learned (confident) |
| 1 (control) | 0 | 0 | — |

Confident arms: index >= 1.5 c* (predicted learned) or index <= c*/1.5 (predicted not learned).

## Hypotheses and decision rules (`aggregate_o.py`)
Outcome per run: L(0.02), the greedy |leverage| at 2% equity. Arm gap G(alpha) = mean L(0.02) of the arm's 10 runs
minus that of the control's 10 runs. An arm is **learned** if G >= 4.

- **O1 (primary: prediction).** Among the confident arms, the index's prediction (learned iff index >= c*) is correct
  in all but at most one arm.
- **O2 (dose-response).** Across all 80 runs, the Spearman correlation between the run's zone index and L(0.02) is at
  least 0.4, with a one-sided permutation p-value below 0.01 (10,000 permutations).
- **O3 (optimum versus learnability).** At alpha = 0.9 the DP still prescribes maximal leverage up to at least
  0.5 allocations in both zones, yet the agents do not learn it: G(0.9) < 4 and fewer than 5 of 10 runs choose
  leverage 16 at 2% equity.
- **O4 (replication of Study L).** alpha = 0 against the control: G(0) >= 4 and the one-sided 95% bootstrap lower bound
  of the gap (10,000 resamples of runs) is above 0.

## Exploratory (reported, not tested)
- Logistic fit of P(L(0.02) = 16) on log10(index) over the treatment runs: the transition point c50 with a bootstrap
  interval, compared with c*.
- The transition arm alpha = 0.7.
- Learned gambling region (Study K's measure) against the DP region, by arm.
- Reported against booked returns by arm.

## Disclosures
- A smoke test (16,384 steps, TRAIN split, alpha 0 and 1, DK1, seed 30) checked the code; its output is in
  `results_o/jobs_smoke/` and is excluded from all results.
- The design calculation was first launched with a longer list of alphas, stopped after alpha = 0.8, and relaunched with
  the final arm set; the registered `dp_o_*.json` are from the relaunch.
- `aggregate_o.py` was tested on synthetic run files before registration.
- Failed runs are rerun with the same seed; every run is reported. Any deviation is logged in `RUN_LOG_O.md`.
