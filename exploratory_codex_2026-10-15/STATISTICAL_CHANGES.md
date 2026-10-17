# Exploratory statistical audit, 15 October 2026

This folder is a new, explicitly exploratory analysis of existing conditions.
It does not amend a pre-registration, registered script, verdict, raw result or
`runs_*.csv`. The original core audit reads no Study Q file; the separate
guarded Q diagnostic reads only the completed cohort. Registered analyses remain
the source of confirmatory verdicts.

Run from the repository root:

```
python exploratory_codex_2026-10-15/statistical_audit.py --seed-fit-bootstrap 199
```

`statistical_audit.json` records every input's SHA-256 and the numerical output.
The latest core execution used one numerical-library thread, took 4.33 seconds and
performed no training or dynamic-programming solve. The 199-replicate sensitivity
has limited Monte Carlo precision; it is not a new confirmatory confidence claim.

## Count correction

The pre-Q Table 1 cohort comprised **824 distinct trained agents**:
B 120, C 24, E 120, F 60, G 20, H 40, I 90, J 60, K 60, L 20, N 40,
O 80 and P 90. Studies B--K comprise 594 agents. Study C's 120 snapshot rows
come from 24 training paths. Study M evaluates Study L's 20 checkpoints without
retraining, and Study P's 170-row CSV includes Study O's 80 reused runs.
Study A is a separate cohort of 280 agents. Smoke tests and pilots are excluded.
Study Q subsequently completed all 100 registered runs, raising the Table 1
cohort to **924**. The core `statistical_audit.json` deliberately preserves its
pre-Q count breakdown; completed Q has a separate guarded diagnostic output
and is included in `tools/verify_studies.py`.

Thus the manuscript's old 644-agent abstract count (and an anticipated 744 after
Q) is stale. Existing Table 1 agent rows already sum to 824; their constituent
counts agree with the shipped final per-run results.

Primary registered verdict counts independently read from frozen verdicts:
B 3/5, C 4/5, E 5/5, F 5/6, G 0/3, H 2/5, I 2/9, J 3/5 and K 0/3,
totalling 24/46; N 6/9; O 4/4; P 3/4. B2's secondary comparison is excluded
from the primary scorecard. These are decision-rule counts, not independent
replications or a family-wise significance statement.

## Recomputed endpoints

An independent convex parameterization of the same unpenalized logistic
likelihood reproduces Study O's exploratory c50 = 0.429488946% and slope
5.203156717. The joint O/P model gives baseline c50 = 0.330330901%, slope
4.624233862 and beta = -0.429725147. The O-only and joint baseline estimates
are different fits and must be identified as such. Fisher exact one-sided
p-values are 0.001820514 (P3, 14/20 versus 4/20) and 0.302903587 (P4, 17/30
versus 14/30).

Including -0.5 in the interval is the registered P2 compatibility decision;
it does not establish the exponent or prove a sample-complexity law for PPO.
The joint model assumes a shared logistic slope across three budgets whose
alpha-arm coverage is unequal. Inference is conditional on the two market
panels, the registered learners and their normalization procedure.

## Dependence sensitivity

The original N/O/P procedures resample runs and/or permute run outcomes.
The same seed is reused across arms and zones. As an exploratory sensitivity,
we retain both zones and all arms for each seed. In P, a selected seed retains
both its budgets; O and P seed sets are resampled separately.

The 199 seed-cluster logistic bootstrap replicates were all numerically valid;
the percentile interval for beta was [-0.671166, -0.228860]. Leaving one seed
out at a time from O or P gave beta from -0.517381 to -0.325040. Only five seeds
per study are available, so these are small-cluster descriptive checks.

Exact enumeration of all 5^5 empirical paired seed bootstrap draws gave:

| Contrast | Mean | Exploratory 95% interval |
|---|---:|---:|
| N1: ZFG - ZBK zombie-probe leverage | 11.908333 | [8.243750, 14.385417] |
| T1: TFG - TBK terminal-probe leverage | 9.125000 | [4.615000, 13.465000] |
| O4: alpha 0 - alpha 1 leverage | 14.400000 | [13.725000, 15.075000] |

A common arm-label permutation applies one alpha relabeling jointly to every
seed and zone, preserving their dependence. Enumerating all 8! relabelings
gives Spearman rho = 0.745986294 and one-sided p = 0.000297619. This is an
exploratory reference under arm-label exchangeability, not a claim that the
eight alpha conditions are independently sampled markets. None of these
checks changes the registered hypothesis decisions.

## Index definition and baseline scope

`study_o/dp_o.py` and `LIABILITY_STUDY_2026-09-29/action_gap.py` use the same
Q*(e,16) - Q*(e,0), in euros, and binned leverage-16 one-period P&L noise.
`calibrate_o.py` calls the same `learnability_index` helper. The index averages
the interpolated positive gap over a fixed log-uniform *initial-equity proxy*
in [0.05,2], not actual state occupancy. Study K starts in [0.05,6], a mismatch
already disclosed in the O pre-registration. The threshold is 0.513336055%
(rounded 0.513%) and its formula, calibration JSON and DP inputs are among
Study O's files hashed before the full run (2026-10-14 15:03:37 UTC).

Independently rebuilding all baseline comparisons from per-run CSVs, DP JSON
and action-gap CSV agrees with `baselines_o.csv` within 1.2e-16. The
exploratory baseline analysis reports pooled AUC 0.962622549, but the
prospective Study O AUC is **0.915296053 for the index and both local gaps**.
The optimal-region AUC is 0.911184211. The index's 58/70 (82.857%) transfer
accuracy is tied by the gap at 10% equity; the 2%-equity gap gives 52/70 and
optimal region 38/70. The pooled advantage includes retrospective calibration
conditions and does not establish prospective superiority of the integrated
index. Baseline fitting uses 80 floor runs across 12 calibration conditions,
including gamma 0.98; the registered c* uses 10 conditions with matched controls.

An independent O treatment-only ranking check gives **AUC(1−alpha) =
0.919407895**, against 0.915296053 for the index (38 maximal-leverage versus
32 other runs). The dose is a ranking-only comparator: alpha is zero in the
historical calibration, so no transferable dose cutoff is calibrated. Within
the shared O/P/Q dose family, near-monotone index reparameterization does not
establish a need for dynamic programming.

## Verification changes

`tools/verify_studies.py` now checks unique-agent counts, primary support counts,
the O threshold identity, O/P logistic point fits, P3/P4 Fisher p-values, the
exploratory baseline endpoints, and completed Q completeness/headline agreement,
including all six supported criteria and DQN control leverage. Before the
verdict existed it left Q headlines pending and read no partial outcome CSV.
The final completed-Q verifier run ended with `all checks pass`. Hash checks retain
the original registered-prefix convention and do not write study files.

## Completed-Q exploratory diagnostics

`diagnose_completed_q.py` was executed only after Q's unchanged registered
aggregation and all 100 jobs were complete. It writes
`exploratory_completed_q_diagnostics.json` and
`exploratory_completed_q_probe_runs.csv`, including input SHA-256 records.
The frozen runner's serialization statement (`run_q.py:93-100`)
contains greedy probe choices, return summaries and floor incidence per
evaluation draw, but no training visitation, network weights or Q-values.
The requested visitation and critic-gap diagnosis therefore cannot be recovered
retrospectively from the released summaries.

DQN controls averaged low-equity leverage 9.225, with 5/10 maximal choices;
the alpha-0.9 treatment had 6/10 maximal choices and a gap of 1.300. Thus Q3
is a small treatment-control contrast, not an absence of gambling. Seven of
ten controls chose maximal leverage at two allocations and three of ten at
every probe. These summaries do not identify the cause of noisy learning.

Applying one common alpha relabeling to all seeds and zones, enumeration of
5! = 120 permutations gives exploratory one-sided p = **0.025 for each
learner**. These are exchangeability-reference sensitivities, not the frozen
run-level permutation p-values or a six-test family adjustment.

The exact 5^5 paired-seed empirical bootstrap retains both zones and all arms:

| Q2 contrast | Mean | Exploratory 95% interval | One-sided 95% lower bound |
|---|---:|---:|---:|
| A2C alpha 0 minus control | 12.825 | [9.300, 15.150] | 9.750 |
| DQN alpha 0 minus control | 5.200 | [1.650, 7.575] | 2.050 |

Both stay positive under this sensitivity, with only five seed clusters.
Treatment-only index versus dose AUC is 0.812500 versus 0.800000 for A2C and
0.723214 versus 0.678571 for DQN. The original six registered decisions remain
unchanged. No Holm adjustment is presented because not all six criteria are
p-value tests and these exploratory reference tests have a different unit.

The DP-free rollout extension subsequently completed and is documented
separately in `model_free/EXPLORATORY_RESULTS.md`. It has useful pooled ranking
but weak legacy ranking, wide Monte Carlo uncertainty and failed longer-budget
transfer; no confirmatory R was created on this evidence.
