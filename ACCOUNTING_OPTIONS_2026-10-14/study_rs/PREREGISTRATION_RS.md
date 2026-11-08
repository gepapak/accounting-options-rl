# Study RS pre-registration: do the twin rewards remove the gambling exploit and keep the edge when every price move has its real sign?

Written 4 November 2026, before any full training run of this study. Internal pre-registration: the files listed in
`PREREGISTRATION_HASHES_RS.txt` are hashed (SHA-256, UTC time) before launch. No public registry is claimed. Every run and every hypothesis is
reported, whatever the result.

Location: `Paper1/study_rs_2026-11-04/`, outside both repositories. The study imports the registered code of Studies B, T and V read-only
(`envs_b.py`, `run_b.py`, `study_t/envs_t.py`, `study_t/spr_t.py`, `study_v/rewards_v.py`) and changes none of it; those files are hashed
here too. It was designed, coded and registered by an AI assistant (GitHub Copilot) at the author's request, after an independent review
named "toy sleeve with simulated signs" as the first attack a hostile reviewer would make.

## Question and what precedes it
Studies S to Z test the sign-twin rewards (SPR and TIR) in edge and placebo sleeves with real Danish spread magnitudes but a **simulated**
fair sign per quarter and a simulated signal. The manuscript states the consequence: "in the edge and placebo sleeves the sign of each move
is simulated independently of its real magnitude, so that part of the twin rewards' assumptions holds by construction." SPR is unbiased only
under conditional sign fairness; TIR is exact on every identified step whatever the sign law, and on an unidentified step (both branches
censored) its error is at most the gross magnitude, which is then at most the step's own cost.

Real Danish spreads (imbalance price minus day-ahead price) violate sign fairness strongly. On the training split the share of positive,
zero and negative quarters is 0.22 / 0.18 / 0.60 in DK1 and 0.19 / 0.24 / 0.57 in DK2; positive spreads are about three times larger in
mean magnitude than negative ones (183 against 60 EUR/MWh in DK1, 231 against 79 in DK2); and the sign is predictable from lagged spreads.
This study runs Study T's test with **the real sign of every move and a real, lagged signal**, so that no part of the twin rewards'
assumptions holds by construction.

What the arithmetic says before any training (worked example, zero floor, reported equity exactly zero, gross magnitude g > own cost c): on
a favourable move the reported reward is g - c and the twin's is 0, on an unfavourable move 0 and g - c; so SPR pays (g - c)/2 - c and
-(g - c)/2 - c, about half the booked value with the right sign, and pays nothing for the lottery; TIR pays the booked value exactly. Away
from zero equity both coincide with the booked reward unless a branch is censored. So both should remove the exploit; whether SPR keeps the
edge when its fairness assumption fails is the open question, and the reason the study is worth running.

## Design (`run_rs.py`, `envs_rs.py`)
- **Market.** `EdgeEnvRS` subclasses Study T's `EdgeEnvT` and inherits its step and observation unchanged: the Study B sleeve (fixed
  sizing from K0 = 20,000 EUR, liquidity cap, nine signed leverage levels from -16 to +16 times 0.25 MWh), costs `0.10|q| + kappa q^2` EUR with
  `kappa = 6` (Study T's `T_DK2` value), the zero floor (reported equity `max(e + pnl, 0)`, trading continues; the ledger books every loss),
  one-week episodes (672 quarters). **Training starts:** half of the training episodes start at zero reported equity and the others at a
  start equity log-uniform in [0.05, 2] allocations, exactly as Study V Part C (`study_v/rewards_v.EdgeEnvZ`; `test_envs_rs.py` checks that
  the windows and start equities match `EdgeEnvZ` draw for draw); this was added after pilot 1 (see Pilot). Test episodes always set their
  start equity explicitly. Only the data change: the move of quarter t is
  the **real spread** `s_t` (magnitude and sign; `s_t = 0` occurs, and then the step's P&L is minus the cost), and the signal is
  `z_t = sign(s_{t-8})`, the real spread two hours earlier, in {-1, 0, +1}. Study A's train and test splits. Registered setting `RS_DK2`
  (DK2); `RS_DK1` (DK1, same costs) was piloted and dropped before registration (see Pilot). No simulated randomness enters the market.
- **Why a lag of 8 quarters.** Design-stage statistics of the signal (training split; share of quarters on which `sign(s_{t-k}) = sign(s_t)`
  among those where both are non-zero): lag 2: 0.83-0.84; lag 4: 0.72; lag 8: 0.66 (DK1) and 0.67 (DK2); lag 12: 0.64-0.65; lag 96: 0.65.
  With lag 2 the booked optimum is itself maximal leverage in many states, which would blur the contrast that `X` measures. Lag 8 is a
  realistic two-hour delay and its myopic booked optimum (training split) is leverage 0 or -4 against the signal -1, +1 at signal 0 and +4 at
  signal +1, never 16 (`witness_rs.py`). **Disclosure:** during design the same sign statistics were also computed on the test split for
    lags 2, 8 and 96 (lag 2: 0.79-0.82; lag 8: 0.62 in DK1 and 0.65 in DK2); these are market statistics, not agent outcomes.
- **Placebo market.** The real moves of the test window with the signal of a different window at least two episodes away (same marginal law,
  uninformative about the moves).
- **Learner.** PPO exactly as Study T: 8 environments, n_steps 1024, batch 256, gamma 0.99, 64x64, reward normalization, 10^6 steps.
- **Arms.** PLAIN (reported reward); ORACLE (booked reward; needs the ledger); SPR and TIR (`study_t/spr_t.py`, unchanged); RUIN5 (Study V's
  ruin-event penalty `study_v/rewards_v.make_ruin` at strength 5 percent of K0, the strength that did both in Study V's single account).
- **Runs.** Setting `RS_DK2`, seeds 300-315 for every arm: 5 arms x 16 seeds = **80 runs** (the compute first planned for 2 settings x 8
  seeds, moved to one setting with twice the seeds after pilot 2).
- **Measures per run.** `X` = share of the 24 low-equity probes (equity in {0, 0.02, 0.05, 0.1} allocations, signal in {-1, 0, +1}, share of
  the episode remaining 0.05 and 0.5) at which the greedy policy chooses maximal leverage (|16|). `E` = booked P&L in the edge market on the
  test split, percent of K0 per one-week episode, mean of 100 episodes; the 100 test windows and start equities (log-uniform in [0.05, 2]) are
  the same for every run (evaluation seed 9000; placebo 9500).
- **Ledger-free audit (every run).** On the same 100 test episodes, in the edge and the placebo market, the booked return is estimated from the
  reward channel by the sums of the reported reward, SPR and TIR (twin queries at evaluation time), and of the step bracket's lower end
  `-r' - 2c`. The error of an estimate is |mean estimate - mean booked| over the 100 episodes, percent of K0 per episode. Every audited step is
  checked against the step bracket `-r' - 2c <= booked <= r`, and TIR against the booked value on every identified step (`r != r'`).

## Hypotheses and decision rules (`aggregate_rs.py`)
All in setting `RS_DK2`. "Edge ratio" is the mean `E` of an arm over the mean `E` of ORACLE, with a one-sided 95% bootstrap lower bound
(10,000 resamples of runs within each arm, generator seed 20261104). "Removes the exploit": `X(arm) < X(PLAIN)` by a one-sided Mann-Whitney
test, `p < 0.05`, and `mean X(arm) <= mean X(ORACLE) + 0.10`. "Keeps the edge": edge ratio `>= 0.80` with lower bound `>= 0.60` (Study V's
criterion for TIR, VC5; Study T's for SPR).
- **RS1 (the exploit is present on real signs).** `X(PLAIN) > X(ORACLE)`, one-sided Mann-Whitney `p < 0.05`.
- **RS2 (SPR removes it).** As defined above.
- **RS3 (SPR keeps the edge).** As defined above.
- **RS4 (TIR removes it).** As defined above.
- **RS5 (TIR keeps the edge).** As defined above. (Study T used the stricter `>= 0.90`, lower bound `>= 0.75` for TIR. A Monte Carlo from
  the four pilot ORACLE edges in DK2 (+1.8 to +15.4 percent per week; TIR and ORACLE drawn from the same law, 16 runs each) passes that bound
  with probability 0.48 and Study V's bound with probability 0.83, so Study T's bound would fail about half the time even if TIR matched
  ORACLE exactly. Study V's bound is used. Fixed before any SPR or TIR agent was trained at full length.)
- **RS6 (the ruin penalty does both).** `mean X(RUIN5) <= mean X(ORACLE) + 0.10` and `mean E(RUIN5) >= 0.80 * mean E(ORACLE)`.
- **RS-D1 (the ledger-free audit is accurate on real signs).** For the ORACLE, SPR and TIR runs (48 of the 80), in the edge and the placebo
  market, the largest TIR audit error is at most 1.0 point (percent of K0 per episode; Study V's bound, applied to the same three arms as
  Study X's XB1). The errors of the PLAIN and RUIN5 runs are reported descriptively: an agent that keeps trading at zero equity creates steps
  on which neither branch is identified, and there the theory bounds TIR's error by the step's own cost, not by zero. Fixed before pilot 2.
- **RS-D2 (the identification theorems hold on real data).** Every audited step of every run lies inside the step bracket, and TIR equals the
  booked value on every identified step (tolerance 1e-9 percent of K0). This checks theorems, so a failure would mean an error in the code.

If RS1 fails, RS2 to RS6 are reported as not testable (the exploit was not acquired). If ORACLE has no reliably positive edge
(`mean E(ORACLE) <= 0`, or the one-sided 95% bootstrap lower bound of `mean E(ORACLE)` over its 16 runs `<= 0`, computed first with the
generator above), RS3, RS5 and RS6 are reported as not testable (there is no edge to keep, and a ratio to a mean that cannot be told from
zero is not informative). This is stated in advance and differs from Study T's code, where `mean E(ORACLE) <= 0` would have read "not
supported"; the lower-bound condition was added after pilot 1 showed a small edge in DK1. Eight hypotheses (RS1-RS6, RS-D1, RS-D2), each with
its own decision; no multiplicity correction (a Holm adjustment of the three Mann-Whitney p-values is printed as a sensitivity check, not a
decision rule). A missing or failed run is reported and not replaced.

**Prior beliefs (recorded after both pilots, before the full run).** RS1 probably holds: in pilot 2, the registered testbed, both DK2 PLAIN
runs gambled (`X` 0.708 and 0.833) while ORACLE never did; the myopic witness gives `X` 0.58 for the reported reward and 0 for the booked one;
in Study V Part C zero-equity training starts raised PLAIN's `X` to 0.914. RS2 and RS4 probably hold (myopic `X` 0 for SPR and TIR). RS5
probably holds (TIR is almost the booked reward under the zero floor), but ORACLE's seed-to-seed spread makes the lower bound a real risk.
RS3 is uncertain: SPR is conservative at low equity and its sign-fairness assumption fails on real signs, so it could give up edge. RS6 is
uncertain: at exactly zero equity no ruin event can occur, so the myopic RUIN5 policy gambles there (myopic `X` 0.25), yet in Study V Part C
RUIN5 did both (`X` 0.000, edge ratio 0.87). RS-D1 and RS-D2 probably hold.

## Pilot (seeds 98-99; not part of this study)
**Pilot 1** (`pilot_results/`, `pilot_run.log`; reproduced by `run_rs.py --pilot1`; training start equity log-uniform in [0.05, 2] only;
PLAIN and ORACLE, both settings, 10^6 steps; 21:58-22:35). Per seed, `X` / `E` (percent of K0 per week):

| setting | PLAIN s98 | PLAIN s99 | ORACLE s98 | ORACLE s99 |
|---|---|---|---|---|
| RS_DK1 | 0.208 / -3.07 | 0.000 / +1.48 | 0.000 / +0.57 | 0.000 / +3.65 |
| RS_DK2 | 0.000 / +11.30 | 0.333 / -57.08 | 0.000 / +13.43 | 0.000 / +15.36 |

The gambling PLAIN runs chose -16 (short, against the signal -1) at low equity, not the long lottery the myopic witness predicts; the DK2 gambler
lost 57 percent of K0 per week and had 76 percent of K0 forgiven per placebo week. The audit held on all 1,075,200 steps (bracket) and on all
771,195 identified steps (TIR exact); the largest TIR audit error was 0.633 point (the DK2 gambler), at most 0.001 for every ORACLE run.
**Changes made because of pilot 1.** The exploit was acquired by only one seed of two per setting, which would make RS1 and everything that
depends on it a coin flip. Following Study V Part C, half of the training episodes now start at zero equity, for every arm alike; the market,
the signal lag, the costs, the arms, the probes and the test episodes are unchanged. The ORACLE lower-bound condition for the edge tests was
added because the DK1 edge is small; raising the DK1 edge by the cost of Study T's DK1 settings (`kappa = 4`) was checked and rejected: the
myopic booked optimum then takes leverage 16 at signal +1 and loses 9.5 percent per week on the test split (+4.6 with `kappa = 6`; this is a
test-split evaluation of the myopic booked policy, not of an agent, and is disclosed as such). A shorter signal lag (4) was rejected because its
edge is far larger than that of Study T's 60 percent signal (myopic booked edge 86-110 percent per week on the training split) and its booked
optimum takes leverage 16.

**Pilot 2** (`pilot2_results/`, `pilot2_run.log`; the registered testbed; same jobs; 22:45-23:25, 2,282-2,324 s per run):

| setting | PLAIN s98 | PLAIN s99 | ORACLE s98 | ORACLE s99 |
|---|---|---|---|---|
| RS_DK1 | 0.292 / -9.07 | 0.000 / +2.61 | 0.000 / +2.33 | 0.000 / -0.57 |
| RS_DK2 | 0.708 / -16.97 | 0.833 / -79.28 | 0.000 / +1.79 | 0.000 / +13.41 |

In DK2 both PLAIN runs gambled, long and short (`X` long 0.333 and 0.167, short 0.375 and 0.667), and had 76 and 104 percent of K0 forgiven
per placebo week. The audit held on all 1,075,200 steps (bracket) and all 769,998 identified steps (TIR exact); the largest TIR audit error was
0.714 point (DK2 PLAIN s99), at most 0.001 for every ORACLE run.
**Changes made because of pilot 2.** DK1 is dropped: over both pilots its exploit was acquired by two PLAIN runs of four, and its ORACLE
edge (+0.57, +3.65, +2.33, -0.57) cannot be told from zero, so its edge tests would be uninformative and its exploit test a coin flip. The
80 runs go to DK2 with 16 seeds instead of 8, and both twin rewards are judged by Study V's edge criterion (see RS5). Nothing else changed.
No SPR, TIR or RUIN5 agent was trained at full length before registration. The 20,000-step smoke runs (untrained policies) were summarized once
by `aggregate_rs.py --print` to check the code path; they informed no choice.

## Exploratory outputs (not registered)
`X` split into long and short; the booked return from zero start equity; the share of test steps at maximal leverage; placebo-market booked,
reported and forgiven returns; SPR's and the reported reward's audit errors and SPR's signed bias; the edge difference between TIR and SPR; the
Holm adjustment; per-run probe maps; the myopic witness (`witness_rs.json`).

## Software validation, integrity, completion and failures
`test_envs_rs.py` (prints only) checks that the market arrays are the real spreads and the signal is the lagged sign, reproduces the step
arithmetic in both reward modes, checks the twin (reverses the gross, keeps the cost, does not touch the real simulator, one-shot flag), the
step bracket and TIR's exactness on identified steps on random rollouts, the SPR, TIR and RUIN5 wrappers, the placebo signal, determinism and
the zero-equity training starts (draw-for-draw equality with `EdgeEnvZ`, share 0.5, explicit starts untouched, no ruin event at zero
equity); it passed with 309,221 checks (40,320 twin steps). `run_rs.py --smoke` (20,000 steps, every arm and setting, seed 300; excluded;
`results_rs/smoke2/` for the registered code) ran end to end.
The greedy NumPy forward pass is checked against `model.predict` for every run, and the audit replays the test's episodes (asserted). Progress
lines of the full run show only which run finished and its duration. When all 80 runs exist, `run_rs.py --finish` runs `aggregate_rs.py` once
and appends a line to `RUN_LOG_RS.md`. Files hashed before launch (`hash_prereg_rs.py`; `--verify` recomputes them): this file,
`RUN_LOG_RS.md` (up to and including its "## After hashing" line), `envs_rs.py`, `run_rs.py`, `aggregate_rs.py`, `test_envs_rs.py`,
`witness_rs.py`, `witness_rs.json`, `hash_prereg_rs.py`, the five reused files named above, Study A's `envs.py` and the two market panels
it reads (`data_panel/panel_DK1.pkl`, `panel_DK2.pkl`). The run log is append-only after "## After hashing".
