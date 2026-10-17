# Study N pre-registration: forgiven versus booked deficits under settlement lag and terminal limited liability

Internal pre-registration, written and hashed (`PREREGISTRATION_HASHES_N.txt`) before the full run and before any test-period evaluation. The disclosed pilot used only the TRAIN split and is described below.

## Question

Theorem 1 of `THEORY_ACCOUNTING_OPTIONS.md` orders accounting rules by the option they grant. Two rungs have not been tested with learning agents:
- a settlement lag (a "zombie window");
- terminal limited liability (Merton's call).

We ask whether RL agents learn the exercise policy of each option. The comparison is against a matched control that books the same deficit: identical states, dynamics, observations, rewards before closure and training starts. The only difference is the bonus (−L)^+ at closure.

## Design (`envs_n.py`, `run_n.py`)

**Settings**
- **Z (zombie window):** a breach of zero is detected W = 96 quarter-hours later. Trading continues meanwhile, then the sleeve closes. The arms are:
  - ZFG, deficit forgiven: once insolvent, the simulator reports the owner's claim max(L, 0) and rewards its change;
  - ZBK, deficit booked: reported equity is L throughout.

  Observation: [L/K0, k/W].
- **T (terminal limited liability):** the horizon is H = 192 quarter-hours. The ledger may go negative. The arms are TFG (deficit forgiven at the horizon) and TBK (deficit booked). Observation: [L/K0, τ/H].

**Market and sleeve.** As in Study F:
- the placebo market (sign-randomized Danish spreads);
- K0 = 20,000 EUR; leverage ±{0, 0.25, 1, 4, 16} × 0.25 MWh;
- liquidity cap 25 % of balancing volume;
- fee 1.0 EUR/MWh: the Study H level, chosen after pilot 1 (see Pilot);
- reward = change in reported equity in % of K0.

**Training starts.**
- Half the episodes start solvent, with equity log-uniform in [0.05, 2]·K0.
- Half start insolvent, with deficit log-uniform in [0.01, 1]·K0.
- Setting Z: insolvent starts begin inside the window, with k uniform on 1..W. Episodes last at most 672 quarter-hours.
- Setting T: the starting time-to-go is uniform on 1..H.
- Time limits are passed to the learner as truncation (bootstrapped), not as termination.

**Learner.** PPO with Study F's configuration:
- 8 parallel environments, rollout 1,024, batch 256, network 64×64;
- reward normalization (VecNormalize, discount 0.99);
- 1,000,000 steps.

**Runs.** Seeds 0–4 × zones DK1, DK2 × 4 arms = 40 runs, 10 per arm.

**Probes.** The probes are greedy |leverage| at fixed inputs:
- Z, insolvent: n ∈ {−0.5, −0.25, −0.1, −0.02}·K0 × k ∈ {96, 48, 12};
- Z, solvent (k = 0): e ∈ {0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 3}·K0;
- T: L ∈ {−0.25, −0.1, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2}·K0 × τ ∈ {1, 4, 16, 48, 96, 192}.

**Evaluation (for X1 and exploratory outcomes).** Placebo test split (2026-01-01 .. 2026-08-17), 20 independent sign draws, starting equity K0.
- Z: one run per draw, from the start of the test period until closure or its end.
- T: consecutive non-overlapping 192-quarter-hour horizons, each starting at K0.

## Theory benchmark (`dp_n2.py`, registered design; `dp_n2_DK1.json`, `dp_n2_DK2.json`)

The risk-neutral optimum on the quantile-binned training distribution, at fee 1.0 EUR/MWh and discount 0.99:
- **ZFG:** |leverage| 16 at every insolvent probe; the action gaps over flat are 2.4–43.5 EUR. When solvent it is 16 up to 0.5·K0, with gaps of 2.3–19.6 EUR.
- **ZBK:** flat everywhere.
- **TFG:**
  - 16 at every probe with L ≤ 0.25·K0 and τ ≤ 96 (gaps 2.3–55 EUR);
  - **flat at τ = 192** for every L: waiting is optimal far from the horizon;
  - at L = 0.5·K0, 16 only for τ ∈ {48, 96}.
- **TBK:** flat everywhere.

The first DP (`dp_n.py`, fee 0.10, lump-sum forgiveness) is retained for the record.

## Hypotheses and decision rules

Intervals are 95 % percentile bootstrap intervals over the 10 runs per arm, with 10,000 resamples per arm, independently.

| ID | Prediction | Decision rule |
|---|---|---|
| N1 (primary) | Forgiveness makes insolvent-but-undetected agents gamble | Mean over the 12 insolvent probes, ZFG − ZBK ≥ 8 units, lower bound > 4 |
| N2 | Maximal zombie gambling, run by run | ZFG |lev| = 16 at (n = −0.1·K0, k = 48) in ≥ 8 of 10 runs |
| N3 | The window's option raises solvent risk-taking near zero | Solvent |lev| at e = 0.1·K0, ZFG − ZBK, lower bound > 0 |
| N4 (finer) | Booked insolvent agents stay flat | ZBK mean insolvent-probe |lev| ≤ 1 in ≥ 8 of 10 runs |
| T1 (primary) | Terminal limited liability makes low-equity agents gamble | Mean over L ∈ {0.02, 0.1, 0.25}·K0 × τ ∈ {1, 4, 16, 48, 96}, TFG − TBK ≥ 8, lower bound > 4 |
| T2 | Run by run | TFG |lev| = 16 at (L = 0.1·K0, τ = 16) in ≥ 8 of 10 runs |
| T3 | Insolvent agents gamble before the horizon | Mean over L ∈ {−0.25, −0.1}·K0 × all τ, TFG − TBK, lower bound > 0 |
| T4 (finer) | Gambling is timed to the horizon | TFG |lev| at (L = 0.1·K0, τ = 16) > at (L = 0.1·K0, τ = 192) in ≥ 8 of 10 runs |
| X1 | The forgiven arms report profits they did not book | Draw-mean reported − booked > 0 in ≥ 8 of 10 runs, in both ZFG and TFG |

N4 and T4 are finer predictions. N4 relies on booked agents learning to stay flat, which needed the higher fee in Study H. T4 relies on a small action gap far from the horizon.

**Exploratory outcomes (no decision rule):**
- share of actual insolvent-window evaluation steps at maximal leverage;
- forgiven deficit (the loss shifted to the settlement body) in % of K0;
- per-zone results;
- comparison with Study F.

## Pilot (disclosed)

Two pilots were run on the TRAIN split, seed 0, DK1 (`RUN_LOG_N.md`).

- **Pilot 1:** fee 0.10, lump-sum forgiveness at detection, 300,000 steps; archived in `pilot1_archive/`.
  - The terminal setting learned the DP policy.
  - In the zombie setting, ZFG did not gamble, and ZBK gambled in most states.
  - Two changes were made before registration:
    - the fee was raised to 1.0 EUR/MWh in all arms;
    - ZFG reports the owner's claim max(L, 0) instead of paying the forgiveness as a lump sum at detection. The terminal payoff is unchanged; only the reward timing changes.
- **Pilot 2:** the registered design at the full 1,000,000-step budget. Its outcome is recorded in `RUN_LOG_N.md` before the hashes are taken.

## Analysis

`aggregate_n.py` writes `VERDICTS_N.md` and `runs_n.csv`. All hypotheses are reported, whatever the outcome.
