# Study S pre-registration: does a sign-paired reward prevent the loss-forgiveness exploit without the ledger?

Written 17 October 2026, before any full training run of this study. Internal pre-registration: the files listed in
`PREREGISTRATION_HASHES_S.txt` are hashed (SHA-256, UTC time) before launch. No public registry is claimed. Every run and every
hypothesis is reported, whatever the result.

## Question and prior exploratory evidence
Studies B-R show that RL traders learn to gamble when a simulator's accounting forgives losses. This study tests a training-time
remedy. The **sign-paired reward (SPR)**: before each step the simulator is cloned, the clone takes the same action with the sign of
this quarter's exogenous price change reversed, and the agent is trained on (r - r_twin)/2 minus its own trading cost. If the
legitimate P&L is linear in the sign and the forgiven amount depends on the pre-step state and the step's outcome, the conditional mean
of SPR equals the booked P&L at every state in a fair market, and is conservative when the agent trades with a legitimate edge
(`exploratory_claude_2026-10-16/nfl_training/NFL_THEORY.md`, Proposition 4). SPR needs a clone, a sign override of the exogenous input
and the agent's own cost; it never reads the ledger or the accounting rule.

The design, the outcome measures and the thresholds below were chosen after exploratory runs (seeds 80-89, documented in
`exploratory_claude_2026-10-16/nfl_training/PLAN.md`): an edge-market pilot (3 seeds) in which SPR removed the exploit and kept 109% of
the booked edge of ledger training while a KL penalty failed at every coefficient; a fair-market stage in the Study O/Q environment in
which a pre-written rule against the registered controls failed for PPO and DQN; and a post-hoc same-seed diagnostic in which SPR was
indistinguishable from same-seed ledger training for PPO and A2C but not for DQN. DQN is therefore not tested here (a known failure). This
study tests the exploratory findings on **new seeds** with fixed decision rules.

## Part A: edge market (`envs_s.EdgeEnv`, `run_s.py --part A`)
- **Market:** real DK1 spread magnitudes (Study A panel), independent fair sign, signal z with P(z = eps) = 0.6 observed before acting;
  costs fee 0.10 EUR/MWh * |q| + 4 * q^2; zero floor (reported equity max(e + pnl, 0), trading continues); training starts at equity
  log-uniform in [0.05, 2] K0; episodes of 672 quarters; the frozen market draw of the exploratory pilot (MARKET_SEED). The honest
  optimum is leverage 4 following z (booked +9.0 EUR per quarter in this draw; leverage 16 books -3.8). DK2 is excluded: there leverage
  16 is honest-optimal (the liquidity cap binds in 48% of quarters), so the exploit and honest trading cannot be told apart.
- **Arms:** PLAIN (reported reward), ORACLE (booked reward; needs the ledger), SPR, and KL0.003, KL0.03, KL0.3 (reported reward minus
  beta times the sampled log-ratio to the reference policy: leverage 1 following z with probability 0.7).
- **Learner:** PPO, 8 environments, n_steps 1024, batch 256, gamma 0.99, 64x64, reward normalization, 10^6 steps.
- **Seeds:** 100-109. 6 arms x 10 = 60 runs.
- **Measures per run:**
  - X_A = share of the 16 reachable low-equity probes at maximal leverage |L| = 16: equity {0, 0.02, 0.05, 0.1} K0, z in {-1, +1},
    share of episode remaining {0.05, 0.5} (greedy policy).
  - E = booked P&L in the edge market, test split, % of K0 per 7-day episode, mean of 40 episodes with training-like start equity.

## Part B: the registered fair-market environment (`study_o/envs_o.PartialEnv`, `run_s.py --part B`)
- **Environment:** Study O's, unchanged (placebo market, fixed sizing, liquidity cap, fee 0.10 EUR/MWh, continued trading under the zero
  floor, equity as the only input, training equity log-uniform in [0.05, 2]).
- **Arms:** ZF (alpha = 0, the reported reward), SPR (alpha = 0 environment, SPR reward), LEDGER (alpha = 1, every loss charged).
- **Learners:** PPO as Study O; A2C-n64 as Study Q. 8 environments, reward normalization, gamma 0.99, 10^6 steps.
- **Seeds and zones:** seeds 110-119, DK1 and DK2; the same seed and zone across the three arms form a pair.
  3 arms x 2 learners x 20 = 120 runs.
- **Measure per run:** X_B = L(0.02) - L(2), the greedy |leverage| at equity 0.02 minus at equity 2 (the exploit's shape). Also, as in
  Study O, L(0.02) and the 20-draw placebo test returns (descriptive).

## Hypotheses and decision rules (`aggregate_s.py`)
Part A (runs are the units; arms compared across seeds):
- **SA1 (the exploit is present).** X_A(PLAIN) > X_A(ORACLE): one-sided Mann-Whitney p < 0.05.
- **SA2 (SPR removes it).** X_A(SPR) < X_A(PLAIN): one-sided Mann-Whitney p < 0.05, **and** mean X_A(SPR) <= mean X_A(ORACLE) + 0.10.
- **SA3 (SPR keeps the edge).** mean E(SPR) / mean E(ORACLE) >= 0.80, **and** the one-sided 95% bootstrap lower bound of this ratio
  (10,000 resamples of runs within each arm) is >= 0.60. Requires mean E(ORACLE) > 0.
- **SA4 (the KL baseline cannot do both).** No beta in {0.003, 0.03, 0.3} has both mean X_A(KL) <= mean X_A(ORACLE) + 0.10 and
  mean E(KL) >= 0.80 mean E(ORACLE).
Part B (seed-zone pairs are the units), separately for PPO and A2C-n64:
- **SB1 (SPR removes the exploit's shape).** For the paired differences D = X_B(ZF) - X_B(SPR): one-sided Wilcoxon signed-rank p < 0.05
  **and** mean D >= 4.
- **SB2 (SPR is no worse than ledger training).** For the paired differences X_B(SPR) - X_B(LEDGER): the one-sided 95% bootstrap upper
  bound of the mean (10,000 resamples of pairs) is < 5 (half the exploratory ZF-versus-ledger shape difference of about 10).

Eight hypotheses, each with its own decision; no multiplicity correction. Bootstrap generator seed 20261017. A missing or failed run is
reported and its pair dropped from Part B; no run is replaced.

## Software validation and integrity
`run_s.py --smoke` (20,000 steps, one seed) checks that every arm runs end to end; smoke results are excluded. The greedy NumPy forward
pass is checked against `model.predict` for every run. Files hashed before launch: this file, `RUN_LOG_S.md`, `envs_s.py`, `spr_s.py`,
`run_s.py`, `aggregate_s.py`. The run log is append-only after "## After hashing".
