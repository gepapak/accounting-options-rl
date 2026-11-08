# Study RE pre-registration: does learned exploitation of the floor vanish where the market cannot reach it?

Written 7 November 2026, before any full training run of this study. Internal pre-registration: the files listed in
`PREREGISTRATION_HASHES_RE.txt` are hashed (SHA-256, UTC time) before launch. No public registry is claimed. Every run and every
hypothesis is reported, whatever the result.

## Question and prior evidence
Two accountings that agree on an interval of market moves write the same reward log at every length inside it (manuscript,
Proposition "Logs do not buy range"). Read on the learning side: a learner whose training never crosses a floor's kink generates
the same experience under the floored and the unfloored simulator, so its output has the same law, and nothing about the floor can
be learned, for good or ill. Studies E, F, H and J are consistent with this (acquisition where low equities and high leverage put the
floor inside the visited region; none where calm spreads, costs or equity-scaled sizing left it outside), but none was designed to
test it: in every study the floor's distance and other things changed together. This study moves only the distance.

## Design (`envs_re.ScaledLiabilityEnv`, `run_re.py`)
- **Sleeve:** Study F's, unchanged: placebo DK1 market (real spread magnitudes, independent fair signs, the Study B training draw),
  fixed sizing from K0 = 20,000 EUR (Q1 = 0.25 MWh at leverage 1, grid {0, ±0.25, ±1, ±4, ±16}, liquidity cap), fee 0.10 EUR/MWh,
  episodes of 672 quarters, training equity log-uniform in [0.05, 2] K0, equity as the only input.
- **Scale c:** every spread and the fee are multiplied by c in {1, 0.3, 0.1, 0.03, 0.01}. Scaling both leaves cost per unit of
  spread, the leverage grid and the cap unchanged and moves only how far an episode's P&L can travel relative to the equity: the
  floor's distance in units of the market's reach.
- **Rules:** FL (every loss booked; the account closes at non-positive equity, overshoot booked) and ZF (reported equity floored at
  zero, trading continues). Same seed and scale across the two rules form a pair; the two runs of a pair see identical market draws
  and identical start equities until the first step on which the rules differ.
- **Learner:** PPO (stable-baselines3), 8 environments, n_steps 1024, batch 256, gamma 0.99, 64x64, reward normalization, 10^6
  steps, CPU, one torch thread. Seeds 0-4. 2 rules x 5 scales x 5 seeds = 50 runs.
- **Predicted reach (`reach_re.py`, run before this file was hashed, output `reach_re.json`):** the expected number of training
  episodes per run whose cumulative P&L would reach the floor under maximal leverage at every quarter: c = 1: 856; 0.3: 443;
  0.1: 149; 0.03: 5.7; 0.01: 0.0 (0 of 20,000 simulated episodes; 0 of 20,000 from the lowest training equity). This is the
  registered predictor of how often training can meet the floor. It is not a pathwise bound on a learner (a policy can lose faster
  by luck); Kolmogorov's inequality gives the rigorous per-episode bound, 4.7% from the lowest equity at c = 0.01, which is loose.
- **Measures per run:** the greedy |leverage| L(e) at equity e in {0, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 3} K0; X = L(0.02) - L(2),
  the exploit's shape (Studies O, S); a SHA-256 digest of every policy parameter; the number of training episodes in which the
  floor was reached (ZF) or the account closed (FL), and of training steps on which the floored write differed from the booked one;
  the 20-draw placebo test from equity K0 at the same scale (reported and booked return, share of draws with the floor bound).
- **Pair measure:** D = X(ZF) - X(FL) for the same seed and scale.

## Hypotheses and decision rules (`aggregate_re.py`)
- **RE1 (reach is necessary).** At c = 0.01 every pair is identical on the whole probe grid (all nine L(e) equal) and no ZF run
  reached the floor in training. The digests are reported; identical parameters are expected but not required, since the two
  processes are only guaranteed the same experience, not the same floating-point reduction order.
- **RE2 (the exploit where the reach is large).** At c = 1: mean D >= 4 and D > 0 in at least 4 of the 5 pairs.
- **RE3 (the exploit follows the reach).** Mean D at each of c = 1, 0.3, 0.1 (predicted reach >= 100) exceeds mean D at each of
  c = 0.03 and c = 0.01 by at least 2 units of leverage.
- **Descriptive.** The observed floor-reaching ZF training episodes per scale, against the predicted reach.

## What this study does not test
It does not test whether reach is sufficient: with a reach of 5.7 (c = 0.03) the theory predicts nothing about acquisition, and
the registered rules do not score that cell on its own. It uses one learner, one sleeve and one zone. A non-identical pair at
c = 0.01 with zero floor episodes would indicate non-deterministic training, not a failure of the invariance, and would be reported
as such with the digests.

## Smoke test (excluded)
`run_re.py --smoke`: 16,384 steps, seed 0, c in {1, 0.01}, train split; results in `results_re/jobs_smoke/`. The c = 0.01 pair had
identical parameter digests and probes after 16,384 steps and zero floor episodes; the c = 1 pair differed.
