# Exploratory completed-study cost-baseline comparison

Design fixed before this script's endpoint calculations, 15 October 2026.
Prior study outcomes and earlier estimator results are already known. This is
an analysis plan, **not a pre-registration or an independent validation study**.
Only this new directory may be written. Study R, its design/predictions, and
interim outcomes are not read. No training or DP solve is performed.

## Data, labels and units

Original ten F/H/J/K calibration conditions retain their published operational
label: floor minus matched full-liability mean leverage at 2% equity at least
four. These are ten chosen records, not ten independent markets; discount
variants share features, agents and markets. Report a secondary six-setting
collapse and leave-one-zone/cost-family-out validation, keeping all discount
variants together. Do not randomly split agents or table rows.

Transfers use one observation per preregistered treatment arm, pooling DK1/DK2
and subtracting its same-cohort control mean: O seven arms; P five at 250k and
two at 4M; Q four per learner. All are reported separately by learner and
budget. Predictor values average the two zone-specific values, as in O's
registered arm rule. Alpha-1 controls are not extra discriminative observations.
Raw agents and evaluation draws do not supply an inflated predictive sample.

The only completed cost intervention with otherwise comparable market is
Study H versus base F/K in DK1/DK2. H changes fee, executable cap **and impact**,
so a two-zone paired comparison cannot isolate a pure fee effect. Report that
four-condition discrimination separately. O/P/Q keep that cost configuration
fixed while changing alpha, budget or learner; they do not independently test
cost transfer. FI uses the disclosed revised panel and receives an exclusion
sensitivity. Inputs remain TRAIN-only for predictors.

## All thirteen fixed scores (higher predicts learned)

1. Forgiven share: 1-alpha.
2. Negative nominal per-MWh fee (base .1, H 1; impact deliberately omitted).
3. Negative mean executed transaction cost, including H's impact.
4. Negative mean executed cost divided by one-period payoff standard deviation.
5. Dose times noise/cost: (1-alpha)*sigma/meanCost.
6. Mean absolute gross payoff magnitude divided by sigma, without cost/dose.
7. Cost-adjusted gross dose: ((1-alpha)*meanMagnitude/2-meanCost)/sigma.
8. Signed exact immediate reward g_alpha(.02)/sigma.
9. Signed exact immediate reward g_alpha(.1)/sigma.
10. The existing seven-probe integrated exact one-step diagnostic (%).
11. The stored optimal-DP index (%).
12. The stored DP optimal maximal-leverage region (allocations).
13. A Gaussian mean/variance approximation to the one-step stop-loss reward,
    with mean -meanCost and the same sigma, using the same equity functional.
    Added on primary-literature review before endpoint calculation, to compare
    moment-only versus empirical-tail information. It is a simple model
    approximation, not a new independent accounting identity.

MeanMagnitude is the mean absolute gross payoff before fees, under full
empirical independent fair signs and the executable cap. The one-half factor
in score 7 is fixed by symmetric gains, not fitted. Local scores remain signed.
No coefficient, feature, reference or subset is selected by new performance.

## Calibration and uncertainty about cutoffs

First report original frozen cutoffs without refitting: DP c*=0.5133360550525173%
and exact one-step geometric midpoint 0.08966882375371477% (the latter developed
retrospectively after O/P/Q). Their provenance is different and stays explicit.

For fair comparator calibration, apply the **same balanced-accuracy threshold
rule to all thirteen scores on the original ten labels**. Enumerate every
calibration-optimal threshold interval, with its endpoints and tie conventions
(`learned iff score >= threshold`). If a canonical cutoff is needed, take the
arithmetic midpoint of the first finite optimal interval in ascending order;
unbounded-only or constant calibration is marked unidentified. Report transfer
performance ranges over **all** optimal threshold intervals and whether a test
arm's decision is identified. Never choose a threshold for its test accuracy.
Using held-out predictor values to enumerate possible decision patterns is
allowed; using outcomes to choose a pattern is not.

Metrics: descriptive arm/condition ROC AUC with tie credit, balanced accuracy,
accuracy, every misclassified arm, score ties/distinct values, and the complete
range of transfer decisions consistent with legacy-optimal cutoffs. One-class
cohorts (potentially P 4M) have undefined AUC/balanced accuracy. No p-values,
independent-market inference, random-row CV or run-level pseudo-replication.
Six-family leave-out results and FI exclusion are sensitivity checks, not
new confirmatory evidence or independent sample-size expansion.

The analysis must report when simple cost-aware baselines match or beat the
integrated methods, and when the completed studies cannot distinguish them.
