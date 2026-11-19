# Study GC pre-registration: does PPO learn gym-mtsim's zero-balance floor in the package's own crypto market?

Written 14 November 2026, before any full training run of this study. Internal pre-registration: the files listed in
`PREREGISTRATION_HASHES_GC.txt` are hashed (SHA-256, UTC time) before launch by `hash_prereg_gc.py`. No public registry is claimed. Every
run and every hypothesis is reported, whatever the result. The protocol, its code and its registration were produced by an AI assistant
(Claude, Anthropic) at the author's request, in the session that also ran the pilot below.

## Question and prior evidence

Study G (registered; `LIABILITY_STUDY_2026-09-29/mtsim_study/` in the code repository) trained PPO inside unmodified gym-mtsim 2.0.0, a
public trading simulator listed among Gymnasium's third-party environments, on a synthetic forex market (0.15% per hour). None of its three
hypotheses was supported and the agents never traded. gym-mtsim registers four markets with the same default account (10,000 USD,
leverage 100, stop-out level 0.2) and the same accounting: when a stop-out leaves the balance negative, the balance is set to zero. Its
registered crypto market was never tried.

**Computed before any training (`reach_gc.py`, output `reach_gc.json`, placebo prices as below).**

| | Crypto (BTCUSD, fee 50 USD per coin) | Forex (EURUSD, fee 0.0003) |
|---|---|---|
| Largest accepted position from 10,000 USD | 16.0 coins, 92.0 times the balance | 8.12 lots, 97.5 times the balance |
| Adverse move that empties the account | 1.09% | 1.03% |
| Placebo daily return, standard deviation | 4.42% | 0.34% |
| One bar at that position: bars on which the floor writes | 38.2% | 0.6% |
| One bar at that position: mean published reward, % of balance | +103.88 | -2.39 |
| One bar at that position: mean booked profit, % of balance | -8.00 | -2.44 |
| Training episodes reaching the floor, PPO's initial policy | 74.8% (6,178 per run), 1,838 USD forgiven each | 2.8% (328 per run), 245 USD forgiven each |
| Training episodes reaching the floor, largest-position policy | 95.3% (7,867 per run), 13,609 USD forgiven each | 49.0% (5,741 per run), 707 USD forgiven each |

In crypto the floor is reached in most training episodes and pays far more than it costs: the largest position's published reward is
+103.88% of the balance per bar while its booked profit is the fee. In forex the floor is reached less often and pays almost nothing: the
published and booked rewards of the largest position differ by 0.05 points. The prediction is that PPO learns the floor in crypto and not
in forex, and that a reward read from the published reward on clones of the simulator (LINE) removes what it learns.

**Pilot, disclosed in full** (`Paper1/gym_mtsim_pilot_2026-11-14/`, exploratory, not registered, seeds 0 to 9). With this study's design
for ORIG and FLPATCH in crypto, Study G's three rules gave: G1 +0.290 [0.10, 0.50]; G2 +0.119 [0.004, 0.219]; G3 +4.43 points
[1.13, 8.47]; one forex pair stayed flat. Two other designs were tried and dropped because they did not separate the rewards: gym-mtsim's
registered three-symbol environment with its dictionary observation (300,000 steps; with the action box rescaled to [-1, 1] both the
published-reward and the booked-reward agent went to 65 to 70 times leverage, with the raw box both traded small volumes), and an action
defined as a share of the margin maximum (one seed; both arms learned to hold). This study's design and hypotheses were chosen after the
pilot, so it tests whether the pilot's result replicates with fresh seeds and adds the LINE arm and the forex control; it is not an
untouched test. The pilot's seeds are not reused.

Differences from the pilot: the probe uses the closed-form margin limit at the bundle's first close and the observation [e, e, 0, 0, 0]
(the pilot reset the environment five times on random paths and searched the limit by bisection); the test uses one fixed bank of 300 paths
shared by every agent (the pilot drew 300 random paths per agent); the LINE arm and four forex seeds per condition are new.

Differences from Study G: the market (the package's bundled daily bars as a placebo, instead of synthetic hourly prices), the fee (that of
the package's registered environment for the market; Study G used the environment class's default fee of 0.0005), the episodes (the
bundle's length, instead of 500 hours), the test (300 placebo paths of the bundle's length, instead of 20 paths of 2,000 hours, and booked
profit from the order records of the same run, instead of a second run without the floor; `test_envs_gc.py` checks that the two are equal),
and the LINE arm.

## Design (`envs_gc.py`, `run_gc.py`)

- **Simulator.** gym-mtsim 2.0.0 from PyPI, unmodified; `test_envs_gc.py` checks every installed file against the wheel's RECORD. pandas
  2.0.3 (the bundled data load with pandas 2.0.x), numpy 1.26.4, gymnasium 1.2.0, stable-baselines3 2.7.0, torch 2.8.0 (CPU), Python 3.10.
  The package is imported, not copied.
- **Account.** The package defaults: 10,000 USD, leverage 100, stop-out level 0.2, hedge mode; one order at a time; window 2 (Study G).
- **Markets.** Crypto: the BTCUSD contract of the package's bundled crypto file and the fee of its registered `crypto-hedge-v0` (50 USD
  per coin). Forex: the EURUSD contract of the bundled forex file and the fee of its registered `forex-hedge-v0` (0.0003).
- **Prices.** A placebo built from the bundled daily bars (May to September 2021; 124 bars crypto, 88 forex). A path has the bundle's
  length and starts at its first close; every bar draws one of the bundle's simple-return magnitudes with replacement and a fair random
  sign, so every price is a martingale and no position has a positive expected profit. Training draws a fresh path for every episode; the
  test uses one fixed bank of 300 paths per market (generator seed 20261114).
- **Wrapper (Study G's).** Observation [equity, balance, margin, open volume (divided by 10 in crypto), open profit], in units of
  10,000 USD and clipped to [-10, 10]; the published action box and reward, the reward divided by 100; training balance log-uniform in
  [0.05, 2] times 10,000 USD; one episode is one path: 122 steps in crypto, 86 in forex.
- **Conditions.**
  - ORIG: gym-mtsim as published.
  - FLPATCH: a subclass whose `tick()` is the published method without its last three lines, the balance floor (Study G's control).
  - LINE (crypto only): the published simulator; the learner is rewarded by the reading of the published reward on clones of the
    simulator: the reward at a zero move plus its slope in the price, from bumps of 0.1%, times the realized move. It reads neither the
    ledger nor the order records, and nothing is deleted. On random trajectories it equals the booked step profit within 2.3e-5 USD,
    including 297 steps on which the floor writes (`test_envs_gc.py`).
- **Learner (Study G's).** PPO (stable-baselines3), MLP 64-64, 8 environments, n_steps 1,024, batch 256, gamma 0.99, observation and
  reward normalization, 1,007,616 steps, CPU, one torch thread.
- **Runs.** Crypto: ORIG, FLPATCH and LINE, seeds 10 to 19 (30 runs). Forex: ORIG and FLPATCH, seeds 10 to 14 (10 runs). 40 runs, run
  by `run_gc.py --workers 8`, which skips finished runs. A run that fails for a technical reason is rerun with the same seed, and the
  failure is reported.

## Measures (per run)

- **Exposure x(e)** (Study G's measure): the share of the margin-feasible maximum position that the deterministic policy opens from a flat
  account with equity e times 10,000 USD, for e in {0.02, 0.05, 0.1, 0.25, 0.5, 1, 2}: observation [e, e, 0, 0, 0] through the frozen
  normalization; the maximum evaluated at the bundle's first close; 0 if the policy holds or the simulator would refuse the order.
  `test_envs_gc.py` checks the probe against the orders the environment opens.
- **Test**, every condition in the published simulator: the deterministic policy on the 300 test paths from 10,000 USD. Reported return
  (change in equity), booked return (from the simulator's order records), share of paths on which the floor writes, share ending with a
  zero balance, orders per path, largest leverage. The reading is computed on every test step and compared with the booked step profit.
- **Training counters**: steps and episodes on which the floor wrote; for LINE, the reading's largest error.
- The policy, its normalization statistics and the SHA-256 of its parameters are saved.

## Hypotheses and decision rules (`aggregate_gc.py`)

Bootstrap as in Study G: runs are the units; 95% percentile interval, 10,000 resamples, conditions resampled independently, generator seed
20261114. Low-equity exposure is the mean of x(e) over e in {0.02, 0.05, 0.1, 0.25}.

- **GC1** (Study G's G1, crypto). Mean x_ORIG(0.1) minus mean x_FLPATCH(0.1) is positive, with the lower bound above 0.
- **GC2** (G2, crypto). The same for the low-equity exposure.
- **GC3** (G3, crypto). The mean of reported minus booked test return of the ORIG agents has its lower bound above 0. Reported minus booked
  is non-negative on every path by construction, so GC3 tests whether enough trained agents reach the floor from 10,000 USD, not its sign.
- **GC4** (the reading removes the exploit, crypto; the paper's registered definition). The low-equity exposure of LINE is below that of
  ORIG (one-sided Mann-Whitney, p < 0.05) and its mean is at most FLPATCH's mean plus 0.10.
- **GC5** (crypto). Mean reported minus booked test return of ORIG minus that of LINE has its lower bound above 0.
- **GC6** (forex, the negative control). The absolute difference of mean x(0.1) between ORIG and FLPATCH is at most 0.10, and the mean
  reported minus booked test return of the ORIG agents is at most 0.50 points.
- **GC7** (both markets). The reading equals the booked step profit within 0.001 USD on every test step of every agent and on every LINE
  training step.
- **Descriptive.** Training steps on which the floor wrote, against the reach above; x(e) curves per seed; x(0.1) of LINE; test returns.

## What this study does not test

Real price paths: the bundled path is the only real one, and in the pilot PPO memorized it. The registered three-symbol environment with
its dictionary observation, which the pilot did not separate. Whether the floor is taken at every equity: the agents tend to request a
fixed volume, which is a large share of the margin maximum only at low equity. A realistic account: leverage 100 on daily crypto bars is
the package's default. Other learners. Independence from the pilot.

## Smoke test (excluded)

`run_gc.py --smoke`: seed 99, 16,384 steps, the five cells, five test paths; `aggregate_gc.py --smoke`. Run once before hashing to check
the pipeline; results in `results_gc/jobs_smoke/` and `results_gc/VERDICTS_GC_SMOKE.md`, not part of the analysis.

## Deviations

Any deviation from this protocol is reported in `ERRATA_GC.md`, with the hashed files left unchanged.
