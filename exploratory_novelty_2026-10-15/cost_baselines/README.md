# Exploratory comparison with simple cost-aware baselines

The completed studies do **not** demonstrate better acquisition prediction by
the DP index or the exact one-step diagnostic than by simple cost-aware scores.
Volatility-adjusted cost scores perfectly separate the original calibration
conditions, and dose alone ties the integrated diagnostics' arm-level ranking
within each mixed-label O/P/Q cohort. Transfer decisions depend materially on
which equally calibration-optimal threshold is used.

This is a retrospective analysis, dated 15 October 2026. The outcomes were
already known. `EXPLORATORY_PLAN.md` fixed the thirteen variants, units,
calibration rule and sensitivities before this script calculated their selected
endpoints; it is not a preregistration. No Study R artifact or interim outcome
was read. Nothing outside this directory was written. There was no training,
new DP solve or change to registered results.

## Evaluation units and calibration

The original ten calibration records retain their stored operational labels,
`G >= 4`, where G is the floor-minus-matched-full-liability mean leverage at the
2% equity probe. All ten labels agree with an independent recomputation from
the run CSVs. These records include discount variants of the same markets and
agents. Six distinct zone-by-cost settings remain after retaining gamma .99
and Study H; the four Danish settings still share two source market panels.
Neither ten nor six is a count of independent market draws.

Each transfer observation is one treatment arm, pooling the two Danish zones
and five seeds. Controls match study, training budget and learner. P's reused O
rows are excluded. Alpha-1 controls establish G but are not additional
discrimination observations. Cohorts are kept separate; there is no random-row
cross-validation or run-level predictive sample-size expansion. Predictors use
TRAIN data only. AUC credits score ties by one half.

Two cutoffs keep their prior provenance: DP 0.5133360550525173% and exact
one-step 0.08966882375371477%. The DP cutoff preceded O/P/Q; the exact cutoff
was developed retrospectively after their outcomes were available. They should
not be described as equally prospective tests.

For a fair supplementary comparison, **every** score is also fitted using the
same maximum-balanced-accuracy rule on the original ten labels. Every optimal
interval is retained, with `learned iff score >= cutoff`. A displayed canonical
cutoff is the arithmetic midpoint of the first finite optimal interval, chosen
using calibration only. Constant calibration scores have no identified
canonical cutoff. Transfer ranges enumerate every prediction pattern permitted
by the complete optimal-cutoff set; their endpoints are descriptive possibilities,
not confidence limits and not a rule for selecting the best test result.

## Transfer results

Correct arm classifications are shown below. The normalized-cost dose score is
`(1-alpha)*sigma/meanExecutedCost`; its calibration-only canonical cutoff is
1417.915863336889. The last column varies that cutoff across its entire optimal
interval, (293.53997353932044, 2542.291753134458].

| Cohort | Arms / learned | Prior DP cutoff | Prior exact cutoff | Cost-dose canonical | Cost-dose allowed correct counts |
|---|---:|---:|---:|---:|---:|
| O, PPO 1M | 7 / 5 | 6/7 | 6/7 | 4/7 | 3–7/7 |
| P, PPO 250k | 5 / 3 | 4/5 | 3/5 | 4/5 | 3–5/5 |
| P, PPO 4M | 2 / 2 | 0/2 | 2/2 | 0/2 | 0–1/2 |
| Q, A2C 1M | 4 / 3 | 3/4 | 3/4 | 2/4 | 2–4/4 |
| Q, DQN 1M | 4 / 1 | 3/4 | 1/4 | 4/4 | 2–4/4 |

The prior exact cutoff misclassifies O alpha .9; the prior DP cutoff
misclassifies O alpha .8. The exact diagnostic's P-4M success does not transfer
to DQN, where it incorrectly labels alpha .5, .8 and .9 as learned. The DP
cutoff misses both learned P-4M arms. Full misclassified-arm lists and all
thirteen comparator results are retained in the JSON and transfer CSV.

Dose alone, cost-dose, cost-adjusted gross dose, both signed local gaps, the
integrated exact diagnostic, DP index, DP region and Gaussian approximation all
have **arm-level AUC 1** in each of O, P-250k, Q-A2C and Q-DQN. This is a
within-cohort dose ordering; these families hold the cost configuration fixed.
Nominal fee, mean cost, cost/sigma and gross-magnitude/sigma are constant across
arms and have AUC .5 there. P-4M has only learned arms, so AUC and balanced
accuracy are undefined. These arm-level AUCs differ from the original raw-run
AUCs and must not replace them silently.

Refitting DP, exact and Gaussian scores by the common canonical rule gives the
same correct-arm counts: 5/7, 5/5, 0/2, 3/4 and 3/4 in the cohort order above.
Their perfect calibration does not identify a unique decision threshold. For
example, the full optimal interval is (0.17117292838364762,
1.5394601699298553]% for DP and (0.007323513245505552,
1.0979017424880366]% for exact. Both the old and newly displayed cutoffs lie
within their respective intervals. Exact permits 0–2 correct P-4M decisions
across this set; DP permits 0–1. Reporting only a favorable midpoint would hide
this nonidentification.

## All prespecified baseline variants

All scores are oriented so that larger means more likely to be learned. Cost
is the executed transaction cost including Study H's impact; sigma is the
standard deviation of the fair-sign net one-period payoff. Magnitude denotes
absolute gross payoff before transaction costs, under the executable cap.

| Score | Legacy-10 AUC | Six-setting AUC | Six-setting family-held-out correct |
|---|---:|---:|---:|
| Dose, 1-alpha | .5000 | .5000 | cutoff unidentified |
| Negative nominal fee | .8333 | .8333 | 5/6 |
| Negative mean executed cost | .9524 | .8889 | 4/6 |
| Negative cost/sigma | 1 | 1 | 5/6 |
| Dose * sigma/cost | 1 | 1 | 6/6 |
| Mean magnitude/sigma | .5714 | .4444 | 3/6 |
| (Dose * mean magnitude/2 - cost)/sigma | .6667 | .6667 | 5/6 |
| Signed g(.02)/sigma | 1 | 1 | 5/6 |
| Signed g(.1)/sigma | 1 | 1 | 5/6 |
| Integrated exact one-step diagnostic | 1 | 1 | 6/6 |
| Stored DP index | 1 | 1 | 6/6 |
| Stored DP maximal-leverage region | 1 | 1 | 6/6 |
| Gaussian mean/variance approximation | 1 | 1 | 6/6 |

Family-held-out analysis fits the same canonical rule after withholding all
records with one zone-by-cost setting. It is a small retrospective sensitivity,
not leave-market-out validation: withholding DK1 base still leaves DK1 Study H,
and markets and seed identifiers recur across settings. The ten-record
weighted counterpart is also saved. No model or score was selected from these
results.

Cost alone's apparently strong ten-record AUC relies partly on repeated Danish
discount records. On the six settings it has two disjoint equally optimal
cutoff intervals, both retained in the JSON. FI has mean cost .3649248892 EUR
and is labeled learned; NO2 has lower mean cost .3337490860 EUR and is labeled
unlearned. DK1's cost .3335531826 EUR is very close to NO2's despite their
opposite labels. Cost/sigma accounts for differing payoff scale and is therefore
more informative than nominal fee or executed cost alone; it uses more input
information than either.

## What the completed cost variation establishes

The only included completed within-zone cost comparison is base F/K versus H
in DK1 and DK2. H jointly changes fee, executable cap and impact. Across those
four conditions every nonconstant score above has AUC 1 and ranks base above H
in both zones; dose is tied, AUC .5. Thus these completed interventions
distinguish dose-only from cost-aware predictors but do not isolate fee,
empirical-tail or DP-specific predictive value. No claim is made that this
two-zone bundled intervention generalizes to new markets or costs.

The local stop-loss quantities are classical one-step reward calculations. A
tail/cost comparator using the same empirical tail function, normalization,
clipping and equity aggregation is an algebraic repackaging of the exact
diagnostic, not an independent competitor or novelty result. The two signed
local scores test aggregation choices; the Gaussian score substitutes moments
for the empirical tail. This analysis cannot establish that tail integration
or dynamic programming is necessary for predicting policy acquisition. An
incentive certificate and acquisition prediction are different estimands.

FI's current panel hash differs from the registered J-panel hash; NO2's
matches. The current FI predictor features therefore do not exactly reproduce
the data used by the old agents. This caveat is saved with hashes, and a
FI-exclusion calibration sensitivity is reported. Excluding FI makes cost-only
legacy AUC 1 as well; it does not create new transfer evidence. Stored DP
features retain their original provenance rather than being silently recomputed.

## Reproduction and artifacts

From the code repository root, run:

```powershell
python exploratory_novelty_2026-10-15\cost_baselines\compare_completed_cost_baselines.py
```

The latest numerical runtime was 1.87 seconds with all numeric thread limits
set to one before NumPy import. Outputs include script, plan and input SHA-256
hashes. The script writes only into this exploratory directory.

- `exploratory_cost_baseline_summary.json`: complete scores, optimal cutoff
  intervals, counts of admissible transfer prediction patterns, ambiguous arms,
  performance ranges, misclassified arms, sensitivities, input provenance and
  hashes.
- `exploratory_primary_comparison.csv`: the compact transfer table above.
- `exploratory_transfer_metrics.csv`: every comparator and transfer cohort.
  Error-arm lists, ambiguous-arm lists and calibration-optimal intervals are
  serialized as JSON fields, rather than dropped when flattening the records;
  `[]` means no errors and `null` means no canonical cutoff is identified.
- `exploratory_condition_scores.csv`: predictor records written before labels
  are attached.
- `exploratory_legacy_labels_and_scores.csv`: stored and recomputed labels plus
  every calibration score.
- `exploratory_transfer_arm_scores.csv`: cohort-specific arm G, labels and
  predictors.
- `exploratory_completed_bundled_cost_intervention.csv`: all variants on the
  four base/H conditions.
- `exploratory_leave_family_out_summary.csv`: ten-record and six-setting
  family-exclusion sensitivities.
- `exploratory_run.log`: console output of the recorded run.
