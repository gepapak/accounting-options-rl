# Study P pre-registration: does the learnability threshold scale with the training budget?

Written 14 October 2026 after Study O was complete (21:28) and before any full training run of this study. Internal
pre-registration: the files listed in `PREREGISTRATION_HASHES_P.txt` are hashed (SHA-256, UTC time) before launch.

## Question
Study O found that PPO learns the partial-forgiveness exploit when the learnability index exceeds a threshold
(fitted c50 = 0.429% at 10^6 training steps, 95% interval [0.302, 0.573]%; registered c* = 0.513%). Gap-dependent
sample-complexity theory says that the number of samples needed to distinguish an action with advantage Delta from one
with noise sigma grows as (sigma / Delta)^2. If learnability is governed by this law, the threshold index should fall
as the square root of the training budget N: **c50(N) = c50(10^6) (N / 10^6)^(-1/2)**. Training four times longer
should halve the threshold and reveal exploits that Study O's agents did not learn; training a quarter as long should
double it and hide exploits they did learn.

## Design (`run_p.py`)
Study O's environment and learner unchanged (`study_o/envs_o.py`: a share alpha of every forgiven loss charged in the
reward, continued trading; PPO, equity as the only input, 8 parallel environments, rollout 1,024, batch 256, network
64x64, reward normalization, gamma 0.99; placebo market; DK1/DK2; probes and evaluation as in Study O). New seeds 40–44.

| Budget | Arms (alpha) | Runs |
|---|---|---|
| 250,000 steps | 0, 0.25, 0.5, 0.7, 0.8, and the control 1 | 60 |
| 4,000,000 steps | 0.8, 0.9, and the control 1 | 30 |

Study O's 80 runs at 10^6 steps are the third budget. The learnability index of every arm is Study O's
(`study_o/dp_o_*.json`); it does not depend on the budget.

## Predictions (`design_p.py`, `design_p.json`)
Study O's run-level logistic fit (slope 5.20, c50 = 0.429%) shifted by the law gives:

| Budget | predicted c50 | P(run learns the exploit) by alpha |
|---|---|---|
| 250,000 | 0.859% | 0: 0.84, 0.25: 0.73, 0.5: 0.52, 0.7: 0.24, 0.8: 0.11 |
| 4,000,000 | 0.215% | 0.8: 0.73, 0.9: 0.32 |

## Hypotheses and decision rules (`aggregate_p.py`)
A run learns the exploit when it chooses leverage 16 at 2% equity (L(0.02) = 16). Joint logistic model over the
treatment runs of Studies O and P: P(learned) = sigmoid(s [log10 index - log10 c50(N)]), with
log10 c50(N) = log10 c0 + beta log10(N / 10^6). 95% interval for beta: 2,000 bootstrap replicates, runs resampled within
each (budget, alpha) cell.
- **P1 (the budget shifts the threshold).** The upper bound of the 95% interval for beta is below 0.
- **P2 (the shift follows the sample-complexity law).** The 95% interval for beta contains -0.5.
- **P3 (more training reveals the exploit).** The share of runs at leverage 16 for alpha in {0.8, 0.9} is larger at
  4M steps than in Study O at 10^6 steps (one-sided Fisher exact test, p < 0.05).
- **P4 (less training hides it).** The share of runs at leverage 16 for alpha in {0.5, 0.7, 0.8} is larger in Study O
  at 10^6 steps than at 250,000 steps (one-sided Fisher exact test, p < 0.05).

A power simulation under the predicted model (150 replicates) gave a sampling standard deviation of about 0.10 for
beta with this design, so P1 has power near 1 if the law holds and fails if the budget does not shift the threshold.

## Exploratory
The gap over the control by budget and arm; the fitted slope s; reported against booked returns.

## Disclosures
- Study O's outcomes were known when this study was designed; they are its calibration, and the predictions above
  follow from them by the stated law alone.
- The arms were chosen with a power simulation on Study O's provisional fit (56 of 70 treatment runs, 21:00) and the
  pooled arms of P4 were widened from {0.5, 0.7} to {0.5, 0.7, 0.8} after the final fit, for power; no Study P run had
  started.
- A smoke test (16,384 steps, TRAIN split, the control arm at both budget labels, DK1, seed 40) checked the code; its
  output is in `results_p/jobs_smoke/` and is excluded.
- Failed runs are rerun with the same seed; every run is reported. Deviations are logged in `RUN_LOG_P.md`.
