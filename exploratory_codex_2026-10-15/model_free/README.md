# Exploratory DP-free rollout surrogate

**Exploratory; no pre-registration or confirmatory claim. Written before executing this analysis, after the existing study outcomes were available.** No new RL training is performed. Run only after Study Q is finished: the script refuses analysis without both Q's verdict and aggregated run table. It writes only inside this subfolder and never edits registered scripts, results, hashes, or logs.

## Fixed design

The primary continuation policy is **leverage 16 when reflected equity is below one allocation, otherwise flat** (`threshold1`). The one-allocation threshold is a simple fixed reference, not fitted to DP policies or learning outcomes. Two predetermined diagnostics, always-flat (`flat`) and constant leverage 16 (`max16`), are also reported. Every reference remains in the report; selecting the best after seeing the comparisons would be optimistic and is not part of this design.

For each seven-probe start equity, take leverage 16 versus flat once, then follow the same reference policy in the two branches. Use common random numbers between branches and opposite-sign antithetic path pairs. At every step, sample a uniform full empirical TRAIN row and an independent fair sign. Magnitudes, liquidity caps and fees follow `envs_b.py`/`envs_j.py`; Study H additionally uses its existing 10% cap, fee 1 and impact coefficient 50. Skip the first 200 warm-up rows, as in `dp_full.py`. Keep reflection at zero and continued trading. Equity is not clipped at six allocations.

**FI source revision:** the local FI panel is rebuilt from revised eSett data and differs from the registered J panel. Registered SHA-256: `14de807360dd1916fe79bc5273cbfa0a8cac5b43c9788b3c4e8abab72d20b41e`; inspected local SHA-256: `b8af3388aabcdbff58f718793d2ae5a5d81f96581734682effc53d970642ee13`. NO2 matches its registered hash, `9df2ee4c2698d133205126dd2848e69ee7c9f591fc12abbf94041c429b3036e5`. `THIRD_PARTY_LICENSES.md` attributes the FI difference to revised EXP13 volumes; original J agents used the registered panel. The script reads the registration manifest and dynamically records actual/registered hashes, `REVISED_J_PANEL` status and a source-confound flag in every affected group/index row. It also reports ranking with revised J panels excluded and flags calibration that includes FI. The whole-panel mismatch does not quantify a change in TRAIN payoffs; without the original TRAIN panel, any discrepancy with stored DP values cannot be attributed solely to continuation policy, binning or truncation.

The estimand is the **finite-horizon fixed-policy contrast** `Q_pi(e,16)-Q_pi(e,0)`, not the optimal contrast `Q*(e,16)-Q*(e,0)`. It can disagree with the DP because of the reference, full empirical distribution versus binning, state-grid clipping, and horizon. Such disagreement is a substantive result, not an estimate of pure DP numerical error. Even an alpha-1 fully booked control can have a positive fixed-policy contrast if the first gamble avoids future fees imposed by a poor continuation policy; report this mismatch rather than censoring controls. The index functional uses the registered seven probes, log interpolation and 400-point [0.05,2] log-equity grid. Its noise denominator is the exact standard deviation of the **same full empirical fair-sign one-step payoffs**, not the binned DP denominator. Both denominators and their ratio are reported.

Since reflected trajectories and the fixed policy do not depend on partial-booking share alpha, store reported and booked return differences separately, and combine them as `(1-alpha)*reported + alpha*booked`. All alpha arms reuse the same paths. O, P and Q reuse the same conditions, so their shared indices are **not independent replications**. The script compares every stored legacy F/H/J/K action-gap row (including gamma .98 without matched calibration labels) and all O alpha-zone conditions. Coverage and outcome summaries for O/P/Q are separate; Study O rows copied into P's aggregate are excluded from P transfer summaries.

## Horizon and uncertainty

For constant-16 continuation the common reflected equity difference keeps its sign and contracts. Its omitted discounted reward contrast is bounded by `gamma**H * max_abs_one_step_pnl`; the future booked contrast is zero. Flat continuation has no omitted rewards. For threshold-1, the general bound is `2 * max_abs_one_step_pnl * gamma**H/(1-gamma)`. Horizons target a deterministic index error of at most 0.01 percentage points, capped at 6000; unmet targets remain explicitly flagged. Coupled paths, or threshold paths that are both permanently flat, stop exactly.

Independent batches preserve the dependence between probes, antithetic signs and alpha arms. Bootstrap complete batches and recompute the positive-part functional after averaging; report percentile Monte Carlo intervals separately from the deterministic truncation allowance and their conservative combination. Positive-part estimates can be upward biased near zero; a narrow finite-sample bootstrap is not proof of reliable rare-tail estimation. Intervals condition on the historical market sample and exclude market-sampling uncertainty. No rank significance test or confirmatory hypothesis is claimed.

Fit each reference's threshold **only on the ten original F/H/J/K calibration labels**. Use the geometric midpoint only if largest-unlearned and smallest-learned surrogate indices separate with positive endpoints. Otherwise report threshold transfer unavailable and do not optimize another decision boundary. Report whether separation survives the MC/truncation intervals. Apply the frozen calibration threshold to O/P/Q, including failures: both the raw run-level maximal-leverage endpoint and the arm-level gap over controls (at least 4) are reported. P/Q budget/learner changes are diagnostic transfer tests with no threshold refitting. Near-threshold MC ambiguity is marked. Reference selection, threshold fitting and model tuning based on O/P/Q would require a separate, explicitly exploratory analysis.

Study N is currently a **scope failure, not a numerical test**: its ledger/time or zombie-age state and terminal reward timing do not share the one-dimensional equity/start distribution. A new two-dimensional reference diagnostic and calibration would be needed. E/G likewise have different sizing, action scales or state spaces; no forced numeric mapping is made. These exclusions are written to the summary.

## Execution, resumption and cost

From the code repository, after Q finishes:

```powershell
python exploratory_codex_2026-10-15/model_free/exploratory_rollout_surrogate.py --batches 4 --paths 64 --bootstrap 500 --max-wall-seconds 180 --out exploratory_codex_2026-10-15/model_free/pilot_outputs
```

This computational pilot checks feasibility and uncertainty for the **whole fixed design**, not which reference to retain. If affordable, the default full analysis is:

```powershell
python exploratory_codex_2026-10-15/model_free/exploratory_rollout_surrogate.py --batches 24 --paths 256 --bootstrap 2000 --max-wall-seconds 900
```

There are 12 legacy zone/cost/discount settings and 3 references (36 sequential groups); partial-liability arms reuse the base gamma-.99 paths. **Execution completes all flat-reference groups first, then all primary threshold-1 groups, then all max-16 groups**, regardless of command-line reference order. Thus a cost cap during a long continuation still leaves the all-condition one-step baseline. This computational ordering does not select a reference based on outcomes; all available results and incompleteness are reported. Full settings use 6144 independent antithetic pairs, i.e. 12288 trajectories per equity probe per group. The conservative worst-case ceiling before early stopping is roughly 36 groups × 12288 trajectories × 7 probes × 2 branches × 6000 steps, about 37 billion branch transitions; actual coupling is expected to reduce this substantially, but no runtime claim is made before the pilot. The explicit wall cap stops between batches, saves progress and returns exit code 2. One long batch can exceed the cap. There is no parallel executor and numerical library thread counts are fixed to one. Default cap is 15 minutes per invocation; do not keep rerunning an unaffordable design without reporting the cost failure.

Checkpoints are saved after each batch. Repeating exactly the same command resumes. Increasing `--batches` is allowed; changing paths/seed/horizon, script or empirical data requires a fresh output directory. Do not extend a pilot based on favorable results alone. Output files are prefixed `exploratory`; checkpoint NPZ files contain batch aggregates, not raw eSett rows. All source data stay in their original ignored locations.

Outputs: `exploratory_indices.csv` (unique conditions, MC intervals, truncation and denominator checks), `exploratory_condition_coverage.csv`, `exploratory_transfer.csv`, `exploratory_summary.json` (all ranks, calibration successes/failures, time, transition counts and scope exclusions), and resumable `exploratory_checkpoint_*.npz`. Costs are counted per shared rollout group; summing duplicated per-alpha paths would overcount compute. No Study R is launched by this script. A promising result would require a newly registered study on new conditions and seeds; weak ranking, uncertain tails, failed calibration or excessive cost should be reported as failures.

## Executed results and stopping decision

The pilot completed all fixed references in **15.72 seconds**; flat/max-16 geometric calibration was unavailable because their small-sample unlearned endpoint estimates were zero. This is a **pilot-only failure**. The full fixed design completed in **116.88 seconds**, with all 36 groups/26 unique conditions and 1,375,345,592 logical branch transitions. All conservative truncation targets were met. No analysis process remains active.

| Fixed reference | Full calibration midpoint (%) | Rank correlation with DP, 26 conditions | Mean absolute discrepancy (percentage points) | O raw-run accuracy |
|---|---:|---:|---:|---:|
| Flat | .040318 | .9286 | .243184 | 43/70 (61.43%) |
| Constant 16 | .409781 | .9395 | .082748 | 58/70 (82.86%) |
| Primary threshold-1 | .366768 | .9499 | .233709 | 58/70 (82.86%) |

All three **full** calibrations separate the historical classes using their marginal MC-plus-tail bands; all references are retained without selection. Primary legacy-only rank is .5524, substantially weaker than its pooled rank, and its absolute discrepancy and near-transition uncertainty remain material. The primary midpoint's joint batch-bootstrap MC interval is **[.214692,.759701]%**, with fixed outcome labels/market sample; all 2,000 bootstrap draws retained positive calibration separation. Excluding revised FI gives midpoint .369428%, interval [.224516,.795734]%, and the same separation count. FI's source mismatch remains explicitly flagged; these checks do not isolate its effect from other approximation changes.

Primary O arm classification is 5/7: alpha.7 and .8 are missed. A joint MC audit including midpoint estimation gives clear index-minus-threshold signs for alpha0/.25 and .9/.95, all four matching their observed labels; alpha.5/.7/.8 intervals cross zero. Threshold1 has median per-condition MC interval width .651421 percentage points. No population reliability or universal learner/budget transfer is established. See `EXPLORATORY_RESULTS.md` for all-reference P/Q failures, controls, noise ratios and limitations.

The separate **design-only** check in `new_fee_design/` evaluated exactly fees .4 and1.5, standard caps/no impact, frozen primary reference/threshold and the fixed alpha grid. It completed in **34.66 seconds**. Its predetermined two-zone interval bracketing rule **failed at fee .4**, which had no eligible low treatment. Fee1.5 alone brackets with alpha.25 high and .9 low. The requested two-fee design therefore fails; no alternative fee, weaker criterion or higher sampling budget was searched after that result. **Direction (a) stops here. No Study R was created, registered or launched.** No new RL agents were trained.

Reproduce the joint MC threshold/control/source audit with:

```powershell
python exploratory_codex_2026-10-15/model_free/exploratory_uncertainty_audit.py
```

The audit reads completed output/checkpoints and writes `outputs/exploratory_uncertainty_audit.json`; its intervals exclude uncertainty in market data/outcome labels. Full evidence for the failed design is in `new_fee_design/EXPLORATORY_RESULTS.md`. Existing registered scripts/results/verdicts/hashes/logs were not modified by this exploratory work.
