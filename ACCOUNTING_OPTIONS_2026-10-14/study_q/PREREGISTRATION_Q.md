# Study Q pre-registration: does the learnability index transfer from PPO to other learners?

Written 15 October 2026, after Study O was complete and while Study P was running, before any full training run of this
study. Internal pre-registration: the files listed in `PREREGISTRATION_HASHES_Q.txt` are hashed (SHA-256, UTC time)
before launch.

## Question
Study O showed, for PPO, that the learnability index (the exploit's action gap relative to the reward noise over the
states training visits; `study_o/dp_o.py`) predicts whether the partial-forgiveness exploit is learned. The index is a
property of the market and the accounting rule, not of the learner. Does it order learnability in the same way for
learners that differ from PPO in kind? Study I found that A2C and DQN learn the zero floor's exploit but often gamble at
every equity level, so their thresholds may differ from PPO's; this study tests the direction and the extremes, and
estimates each learner's transition as an exploratory quantity.

## Design (`run_q.py`)
- **Environment:** Study O's (`study_o/envs_o.py`): placebo market, fixed sizing, liquidity cap, fee 0.10 EUR/MWh, a
  share alpha of every forgiven loss charged in the reward, continued trading under the zero floor, equity as the only
  input, training equity log-uniform in [0.05, 2] allocations.
- **Learners:** Study I's configurations: A2C with a rollout of 64 steps, generalized advantage estimation (0.95) and
  advantage normalization (`A2C-n64`); DQN at its Stable-Baselines3 defaults (`DQN-default`). Network 64x64, 8 parallel
  environments, reward normalization, gamma 0.99, 10^6 steps.
- **Arms:** alpha in {0, 0.5, 0.8, 0.9} and the matched control alpha = 1. Indices (Study O, mean of DK1 and DK2):
  1.805%, 0.886%, 0.335%, 0.154% and 0.
- **Seeds and zones:** seeds 50–54, DK1 and DK2. 2 learners x 5 arms x 10 = 100 runs.
- **Measurements:** as Study O (greedy |leverage| at equity probes; 20 placebo test draws from one allocation).

## Hypotheses and decision rules (`aggregate_q.py`), separately for each learner
Outcome per run: L(0.02), the greedy |leverage| at 2% equity. Gap G(alpha) = mean L(0.02) of the arm (10 runs) minus that
of the control (10 runs).
- **Q1 (the index orders learnability).** The run-level Spearman correlation between the run's index and L(0.02) is
  positive with a one-sided permutation p-value below 0.05 (10,000 permutations).
- **Q2 (the full exploit is learned).** alpha = 0 against the control: G(0) >= 4 and the one-sided 95% bootstrap lower bound
  above 0 (10,000 resamples).
- **Q3 (an exploit far below PPO's threshold is not learned).** At alpha = 0.9 (index 0.154%, a third of PPO's fitted
  transition): G(0.9) < 4.
Six verdicts in total. No correction across them.

## Exploratory
- Each learner's logistic transition c50 against Study O's PPO value (0.429%).
- Indiscriminate gambling: mean L(2) by arm (Study I found A2C and DQN often gamble at every equity level).
- Reported against booked returns.

## Disclosures
- Studies I and O, and the partial results of Study P (4M-step runs found the alpha = 0.9 exploit in 8 of 10 runs), were
  known when this study was designed. Study Q uses the registered training budget of Study O (10^6 steps).
- A smoke test (16,384 steps, TRAIN split, the control arm, DK1, seed 50, both learners) checked the code; its output is
  in `results_q/jobs_smoke/` and is excluded. `aggregate_q.py` was tested on synthetic run files.
- The full run is launched automatically once Study P has finished, to avoid sharing the CPU.
- Failed runs are rerun with the same seed; every run is reported. Deviations are logged in `RUN_LOG_Q.md`.
