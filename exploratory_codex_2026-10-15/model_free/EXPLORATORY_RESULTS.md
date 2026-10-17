# DP-free fixed-reference rollout analysis — exploratory results

Executed 15 October 2026, after Study Q's unchanged aggregation produced its verdict and 100-row run table. No new RL agents were trained. The fixed design and reference policies are in `README.md`. Existing study outcomes were known before the exploratory design; these results are not confirmatory.

## Compute and coverage

The pilot (4 batches × 64 independent antithetic pairs) completed all references in **15.72 seconds**, counting **57,198,408 logical branch transitions**. Its Monte Carlo intervals were too wide to assess calibrated predictions reliably. The full fixed design (24 batches × 256 pairs; 12,288 paired trajectories per equity probe/group) was run because the pilot established affordable cost, with all three references retained. It completed in **116.88 seconds**, counting **1,375,345,592 logical branch transitions**, within its 900-second cap. Both runs were sequential with numerical libraries restricted to one thread. No process remains running.

The full output has 36 shared setting/reference groups and 78 index rows: 26 unique conditions × 3 fixed references. O/P/Q reuse partial-liability condition indices and are not independent replications of the index calculation. All discount truncation targets were met; the largest conservative index allowance was **0.009992 percentage points**. Raw eSett data were neither copied nor redistributed.

## Fixed primary: threshold-1 continuation

The ten-condition calibration midpoint is **0.366768%**. In a separate joint batch bootstrap, all 2,000 resamples retained positive separation of the historical learned/unlearned classes. The Monte Carlo midpoint interval is **[0.214692, 0.759701]%**, conditional on fixed market data and outcome labels. This is not a population confidence interval or a prospective validation interval. Per-condition intervals and the uncertainty in the midpoint remain large.

Excluding FI from calibration gives **0.369428%**, with Monte Carlo interval **[0.224516, 0.795734]%** and separation in all 2,000 bootstrap resamples. The locally rebuilt FI panel differs from the registered J panel; actual and registered hashes/status appear in every affected source manifest. NO2 matches its registered panel. A whole-panel mismatch does not quantify a TRAIN-payoff change, so FI discrepancies cannot be attributed solely to the reference or binning.

The primary surrogate has Spearman correlation **0.9499** with the stored DP index across all 26 unique conditions, **0.9436** after excluding revised FI, but only **0.5524** among the 12 legacy F/H/J/K records. Its correlation on O's 14 treatment conditions is 1.0000; that comparison largely ranks a shared alpha family. Rank success alone does not establish reliable absolute indices or calibrated predictions.

Against the stored optimal DP index, primary mean absolute discrepancy is **0.233709 percentage points**, RMSE **0.355398**, and mean signed discrepancy **−0.011291**. Median Monte Carlo interval width is **0.651421 percentage points**, maximum **1.499251**. These are discrepancies between different estimands/distributions, not pure estimator bias: fixed-policy versus optimal continuation, full empirical versus binned payoffs/noise, no upper equity-grid clipping, and revised FI all contribute.

| Fixed reference | Full calibration midpoint (%) | Rank correlation, 26 conditions | Mean absolute discrepancy (percentage points) | O raw-run accuracy |
|---|---:|---:|---:|---:|
| Flat | .040318 | .9286 | .243184 | 43/70 (61.43%) |
| Constant 16 | .409781 | .9395 | .082748 | 58/70 (82.86%) |
| Primary threshold-1 | .366768 | .9499 | .233709 | 58/70 (82.86%) |

All three **full** calibrations have historical class separation using the marginal MC-plus-tail bands. The flat/max-16 geometric-midpoint failures occurred in the small **pilot**, not the full analysis. The table reports all fixed references and does not select the best.

## Threshold transfer without refitting

Raw run classification below means maximal leverage at the low-equity probe, among treatment runs only. Arm classification uses the causal gap over matched controls, at least 4 leverage units. The calibration threshold is frozen across these cohorts; no reference was selected on the following outcomes.

| Reference | O: raw runs / arms | P 250k: raw runs / arms | P 4M: raw runs / arms | Q A2C: raw runs / arms | Q DQN: raw runs / arms |
|---|---|---|---|---|---|
| Primary threshold-1 | 58/70; 5/7 | 41/50; 5/5 | 6/20; 0/2 | 28/40; 3/4 | 26/40; 3/4 |
| Flat | 43/70; 5/7 | 33/50; 3/5 | 14/20; 2/2 | 20/40; 3/4 | 28/40; 1/4 |
| Constant 16 | 58/70; 6/7 | 37/50; 4/5 | 6/20; 0/2 | 28/40; 3/4 | 26/40; 3/4 |

The primary misses O's learned arms at alpha .7 and .8. Their mean-zone surrogate indices are .286727% and .140163%, below the .366768% threshold, while observed gaps are 8.025 and 6.5. Three of seven O arms have MC-plus-truncation intervals crossing the point threshold. The 1M threshold fails on the longer-budget cohort, consistently with the need for budget calibration rather than a universal threshold.

Including midpoint estimation in a joint MC bootstrap, the mean-zone index-minus-threshold interval excludes zero for alpha0/.25 (above) and .9/.95 (below), all four matching their observed arm labels. Alpha.5/.7/.8 remain ambiguous. Above-threshold bootstrap fractions for alpha0/.25/.5/.7/.8/.9/.95 are 1.0000/.9995/.9415/.4045/.0745/.0120/.0060. This audit conditions on the market and outcome labels and excludes the deterministic tail allowance; it is Monte Carlo sensitivity rather than a population reliability guarantee.

For alpha-1 controls, primary point indices are **0.0000167%** in DK1 and **0.004845%** in DK2, both below the threshold and compared with DP index zero. The audit's MC-only 95% upper bounds are .290402% and .283480%; corresponding bootstrap above-threshold fractions are 1.85% and 1.05%. Positive fixed-policy control contrasts can reflect avoiding fees under a poor continuation policy; they are not evidence of an optimal reward exploit in a fully booked control.

Full empirical/binned one-step noise ratios range from **0.996675 to 1.019243**. Small denominator changes do not account for the whole surrogate discrepancy. The flat and constant-16 calibration midpoints are .040318% and .409781%; their overall rank correlations are .9286 and .9395, respectively. All references and their failures remain reported.

## Interpretation and next work

This analysis demonstrates computational feasibility of a DP-free fixed-policy contrast and broad historical class separation, with useful but imperfect O transfer. It does **not** establish an interchangeable estimate of the optimal action gap, precise transition prediction, learner-independent transfer, or scaling to arbitrary simulators. Uncertainty and the weak legacy rank prohibit using the high pooled rank as evidence for those claims.

Study N remains a scope failure rather than a numerical extension: ledger/time or zombie-age states and terminal reward timing require a new diagnostic and calibration. E/G likewise do not share the action/state/sizing scale. No forced numerical mapping was made and no Study R was created or launched.

A subsequent, separately authorized **design-only exploratory check** evaluated the fixed .4 and 1.5 EUR/MWh candidate fees with standard 25% caps/no impact. It read no learner outcomes or DP outputs and fixed c=.366768%. The two-fee bracketing design **failed** its fixed MC-plus-tail interval criterion: fee .4 has no eligible low treatment in both zones. Fee1.5 alone brackets with alpha.25 high and alpha.9 low. See `new_fee_design/EXPLORATORY_RESULTS.md`; no other fees/criteria were searched and no R study was created or launched. Choosing reference policies or threshold adjustments from O/P/Q outcomes would consume those cohorts as calibration and require a separate validation cohort. Extreme alpha arms alone may provide a weak novelty test.

## Reproduction

Pilot and full commands are in `README.md`; outputs and logs are retained separately as `pilot_outputs/`, `outputs/`, `exploratory_pilot.log` and `exploratory_full.log`. The main summaries are `outputs/exploratory_indices.csv`, `outputs/exploratory_transfer.csv`, `outputs/exploratory_summary.json`, and `outputs/exploratory_uncertainty_audit.json`.

Reproduce the joint uncertainty/source-exclusion/control audit with:

```powershell
python exploratory_codex_2026-10-15/model_free/exploratory_uncertainty_audit.py
```

The audit uses an independent, fixed bootstrap seed and preserves within-group alpha/probe dependence. Its code and output distinguish Monte Carlo sensitivity from market/outcome uncertainty. No registered source, result, verdict, hash or log was modified.
