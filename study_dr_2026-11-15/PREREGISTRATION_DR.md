# Study DR pre-registration: does a wider training distribution switch on gym-mtsim's dormant forex floor, and does the edge computed before training say when?

Written 15 November 2026, before any full training run of this study. Internal pre-registration: the files listed in
`PREREGISTRATION_HASHES_DR.txt` are hashed (SHA-256, UTC time) before launch by `hash_prereg_dr.py`. No public registry is claimed. Every
run and every hypothesis is reported, whatever the result. The protocol, its code and its registration were produced by an AI assistant
(Claude, Anthropic) at the author's request, in the session that also ran the pilot below.

## Question and prior evidence

Study GC (registered, `../study_gc_2026-11-14/`) trained PPO inside unmodified gym-mtsim 2.0.0. In the package's crypto market the floor
was learned (all ten floor-trained agents took risk at low equity). In its forex market, at the package defaults (leverage 100,
nominal volatility), none of the five floor-trained agents traded: one bar at the largest position loses even under the published
reward (-2.39% of the balance), because the floor is rarely reached and forgives almost nothing.

Trading agents are increasingly trained on randomized market parameters or synthetic stress scenarios to make them robust (for example
SAiFE-gym, arXiv 2609.17788; domain-randomized market making, TU Delft repository; adversarial synthetic markets, arXiv 2601.17008;
diffusion stress scenarios, arXiv 2510.07099). A literature search on 15 November 2026 found no work on what such training does to a
simulator's loss accounting. This study asks two questions. Does widening the training distribution switch on a floor that nominal
training never learns? Does one number computed before training, the published edge of the largest position over the training
distribution, say when? The training distribution is widened along two levers: randomized volatility, hidden from the agent, and the
account's leverage.

**Pilot, disclosed in full** (`../gym_mtsim_dr_pilot_2026-11-15/`, exploratory, not registered). It used this study's code path for the V
arms: Study GC's wrapper, with training magnitudes times k ~ log-uniform[1, k_hi]. Seeds were 400 to 403 (k_hi 4, ORIG and FLPATCH),
410 to 412 (k_hi 8, ORIG), 420 to 422 (k_hi 16, ORIG), 420 and 421 (k_hi 16, FLPATCH), and 499 (smoke). It counted agents with
low-equity exposure of at least 0.10 among floor-trained agents:

- k_hi 4: 1 of 4 (xlow 0.11, from one probe point; the others 0)
- k_hi 8: 2 of 3 (0.25, 0.13, 0)
- k_hi 16: 2 of 3 (0.35, 0.23, 0)
- floor-deleted controls: none of the 6 pilot controls, and none of Study GC's 5 forex controls.

The leverage arms below were not trained in the pilot. This study's switch-on threshold and its V4, V8 and V16 arms were chosen after
seeing the pilot. It therefore tests whether the pilot's ordering replicates with fresh seeds, and whether the same rule predicts a
second lever and two new arms (V2, L30, L500). It is not an untouched test. No pilot seed is reused.

## Computed before any training (`before_dr.py`, output `before_dr.json`)

The quantities in the table:

- **One bar at the largest position:** from 10,000 USD at the bundle's first close and the arm's leverage, the largest buy the margin
  check accepts, held one bar, over every bundled magnitude times k (capped at 0.9) with both signs.
- **V arms:** averaged over k ~ log-uniform[1, k_hi], with the trapezoid rule on 33 points in log k.
- **Published edge C:** the mean published reward, % of balance.
- **Credit:** published minus booked.
- **Reach:** the share of 1,000 training episodes of the arm's wrapper in which the floor writes. It is computed under PPO's initial policy (each action dimension standard normal) and under the largest-position policy.

| Arm | Training distribution | Position / balance | **Published edge C** | Booked | Credit | Bars on which the floor writes | Reach, initial policy (per run; forgiven per episode) | Reach, largest-position policy |
|---|---|---|---|---|---|---|---|---|
| (Study GC forex, nominal) | k = 1, leverage 100 | 97.5 | -2.39% | -2.44% | +0.05 | 0.6% | 2.8% (GC) | 49.0% (GC) |
| V2 | k ~ logU[1, 2], leverage 100 | 97.5 | **-1.77%** | -2.44% | +0.67 | 2.7% | 9.1% (1,066; 289 USD) | 59.1% |
| V4 | k ~ logU[1, 4], leverage 100 | 97.5 | **+2.06%** | -2.44% | +4.50 | 8.4% | 22.3% (2,613; 589 USD) | 70.1% |
| V8 | k ~ logU[1, 8], leverage 100 | 97.5 | **+12.53%** | -2.44% | +14.97 | 14.4% | 36.4% (4,265; 1,014 USD) | 76.3% |
| V16 | k ~ logU[1, 16], leverage 100 | 97.5 | **+34.42%** | -2.44% | +36.86 | 20.0% | 46.8% (5,483; 1,607 USD) | 81.5% |
| L30 | k = 1, leverage 30 | 29.8 | **-0.74%** | -0.74% | 0.00 | 0.0% | 0.0% (0) | 0.8% |
| L500 | k = 1, leverage 500 | 444.4 | **+13.94%** | -11.11% | +25.04 | 25.3% | 31.7% (3,714; 307 USD) | 89.3% |

**Predictions fixed from C.**

- **Switch-on arms** (C of at least +10% of the balance): V8, V16, L500.
- **Dormant arms** (C not positive): V2, L30. At leverage 30 the floor cannot be reached in one bar: the largest position needs a 3.4% adverse move, and no bundled bar has one.
- **Intermediate arm:** V4. It enters only the ordering in DR4.

The +10% threshold is set from the pilot, which saw learning at +12.5% and +34.4% and almost none at +2.1%.

**Change from the design first proposed to the author: the leverage arms are fixed, not randomized.** The first proposal drew leverage
log-uniformly in [100, 1000] (predicted to switch on) and in [30, 500] (predicted dormant), hidden from the agent. Its means of +9.41% and
-0.16% are the mean over L of the largest position's published reward, which only an agent that knows L can collect. From a flat account
the observation [e, e, 0, 0, 0] carries no leverage, and gym-mtsim refuses an order that needs more margin than the account has. So a request
sized for leverage L* is refused whenever the drawn leverage is below L*. The best single request earns:

- [100, 1000]: +4.63% per bar (sized for leverage 590), between V4 and V8, so the arm has no clear prediction.
- [30, 500]: +0.58% (sized for 396), not clearly dormant.

A trader knows the broker's leverage, so fixed leverage is also the realistic case: 30 is the EU retail cap for major currency pairs
(ESMA, 2018) and 500 is common at offshore retail brokers. A second change from the proposal is the run count (below): 64 runs, to fit the
seven to eight hours agreed with the author.

## Design (`envs_dr.py`, `run_dr.py`)

- **Simulator and versions.** gym-mtsim 2.0.0 from PyPI, unmodified. `test_envs_dr.py` runs Study GC's check of every installed file
  against the wheel's RECORD and of the control's source. Other versions: pandas 2.0.3, numpy 1.26.4, gymnasium 1.2.0, stable-baselines3
  2.7.0, torch 2.8.0 (CPU), scipy 1.15.3, Python 3.10, in Study GC's virtual environment.
- **Everything not listed below is Study GC's forex arm** (`envs_gc.py`, imported read-only):
  - EURUSD from the bundled forex file, with the fee of the registered `forex-hedge-v0` (0.0003).
  - 10,000 USD account, stop-out level 0.2, hedge mode, one order at a time, window 2.
  - The flat five-number observation; the published action box and reward, the reward divided by 100.
  - Training balances log-uniform in [0.05, 2] times 10,000 USD.
  - One episode is one placebo path of the bundle's length (86 steps): every bar draws a bundled magnitude with replacement and a fair sign.
- **Arms.**
  - V2, V4, V8 and V16: leverage 100. In training, each episode's magnitudes are multiplied by k ~ log-uniform[1, k_hi], capped at 0.9, and the agent does not observe k.
  - L30 and L500: nominal magnitudes (Study GC's training paths, draw for draw) and leverage 30 or 500, fixed for the run and the tests.
- **Conditions.** ORIG: gym-mtsim as published, in which a balance left negative by a stop-out is set to zero. FLPATCH: a subclass whose `tick()` is the published method without its last three lines, the floor (Study GC's control).
- **Learner** (Study G's and GC's): PPO (stable-baselines3), MLP 64-64, 8 environments, n_steps 1,024, batch 256, gamma 0.99,
  observation and reward normalization, 1,007,616 steps, CPU, one torch thread. The PPO seed is the run's seed, and environment i has
  seed 100 times the run's seed plus i.
- **Runs (64).**
  - ORIG, seeds 600 to 609: V4, V8, V16, L500.
  - ORIG, seeds 600 to 605: V2, L30.
  - FLPATCH, seeds 600 to 605: V16 and L500, the strongest arm of each lever. These are the controls.

  `run_dr.py --workers 8` starts the runs expected to trade first and skips finished runs. A run that fails for a technical reason is
  rerun with the same seed, and the failure is reported. At about 55 minutes per run with eight at a time (the pilot), the run takes
  about 7.5 hours. Standby pauses it without changing it (Study GC).

## Measures (per run)

- **Exposure x(e)** (Study GC's probe at the arm's leverage; `test_envs_dr.py` checks that at leverage 100 it is Study GC's probe): the
  share of the margin-feasible maximum position that the deterministic policy opens from a flat account with equity e times 10,000 USD,
  for e in {0.02, 0.05, 0.1, 0.25, 0.5, 1, 2}. It uses the observation [e, e, 0, 0, 0] through the frozen normalization and the maximum
  at the bundle's first close. It is 0 if the policy holds or the simulator would refuse the order. `test_envs_dr.py` checks the probe
  against the orders the environment opens, in every arm.
- **Low-equity exposure (xlow):** the mean of x(e) over e in {0.02, 0.05, 0.1, 0.25} (Study GC's GC2 measure). A floor-trained agent with xlow of at least 0.10 is counted as a **learner**.
- **Tests**, descriptive only: the deterministic policy from 10,000 USD at the arm's leverage in the published simulator, on two banks:
  - the nominal bank: Study GC's 300 forex test paths;
  - the stressed bank: 300 paths with magnitudes times 4, generator seed 20261118 (the pilot's).

  For each bank: reported return (change in equity), booked return (from the simulator's order records), the share of paths on which the
  floor writes, the share ending with a zero balance, orders per path, and the largest leverage.
- **Training counters:** steps and episodes on which the floor wrote.
- **Saved:** the policy, its normalization statistics and the SHA-256 of its parameters.

## Hypotheses and decision rules (`aggregate_dr.py`)

Runs are the units. Bootstrap as in Studies G and GC: a 95% percentile interval of a difference of means, 10,000 resamples, groups
resampled independently. Each decision is a single test at 5%, or an intersection in which every listed arm must pass. No multiplicity
correction is applied, as none is needed for an intersection. The four decisions are reported separately, and no overall verdict combines
them.

- **DR1 (volatility randomization switches the floor on).** In V8 and in V16, take the mean xlow of the floor-trained agents minus the mean
  xlow of the six V16 floor-deleted controls. The lower bound must be above 0 in both arms. The controls come from the widest volatility
  range, where trading without the floor is noisiest.
- **DR2 (leverage 500 switches it on at nominal volatility).** In L500, the mean xlow of the floor-trained agents minus that of the six L500
  floor-deleted controls has its lower bound above 0.
- **DR3 (dormant where the published edge is not positive).** In V2 and in L30, at most one of the six floor-trained agents is a learner.
- **DR4 (dose-response).** The floor-trained agents' xlow increases across V2, V4, V8 and V16, the order of C. The test is the one-sided
  Jonckheere-Terpstra statistic, with its p-value from 100,000 random permutations of the arm labels among the 36 agents, and needs
  p < 0.05.
- **Descriptive.** Learners per arm (the switch-on arms are expected to hold most of them). x(e) curves. Training episodes reaching the
  floor against the reach above. Control learners. The test returns on both banks.

## What this study does not test

- **Prices:** placebo paths only.
- **Unrealistic volatility:** volatility ranges of 8 or 16 times are not realistic for EURUSD (16 times is bitcoin-like); the leverage
  arms are the realistic lever.
- **Hidden k:** whether an agent that observes the volatility would condition on it; here k is hidden, so it learns one policy for all
  regimes.
- **Hidden randomized leverage** (see above).
- **Every equity:** whether the floor is taken at every equity. The agents tend to request a fixed volume, which is a large share of the
  maximum only at low equity.
- **Other learners, and gym-mtsim's registered three-symbol environment.**
- **Independence from the pilot** (see above).

## Smoke test (excluded)

`run_dr.py --smoke` uses seed 699, 16,384 steps, the eight cells and five test paths per bank; `aggregate_dr.py --smoke --perms 2000`
aggregates it. It is run once before hashing to check the pipeline. Results are in `results_dr/jobs_smoke/` and
`results_dr/VERDICTS_DR_SMOKE.md`, and are not part of the analysis.

## Deviations

Any deviation from this protocol is reported in `ERRATA_DR.md`, with the hashed files left unchanged.
