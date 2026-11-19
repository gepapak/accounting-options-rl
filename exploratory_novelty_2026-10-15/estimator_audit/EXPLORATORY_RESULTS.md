# Exact one-step audit: results and limitations

Exploratory, 15 October 2026. Existing outcomes were known, but no learner
outcome selected the estimator, continuation policy, fees, alpha grid or
calibration rule. Predictions were saved before the script read outcome labels.
No RL training, DP solve, registered mutation or replacement of previous failed
analyses occurred. Top-level/ancestor AGENTS files were checked; none existed
on the inspected paths. All output is under this new folder.

## What is exact and why previous intervals were wide

For a known empirical TRAIN distribution, X = epsilon*m-c has expected value
−mean(c). The floor reward with partial booking is
X+(1−alpha)(−e−X)^+. With flat continuation this gives

    g_alpha(e) = −mean(c) + (1−alpha)h(e),
    h(e) = E[(−e−X)^+].

Sorting −X over all rows and both fair signs, with suffix sums, evaluates h
without sampled trajectories or signs. The seven-probe functional retains its
original log interpolation and 400 reference weights. Its denominator is the
exact full-empirical one-period standard deviation. Alpha-1 controls have
g = −mean(c) and index zero. This is exact empirical integration, not Q-star
estimation or a generic black-box simulator algorithm.

The old flat estimator already integrated both signs by antithetic pairing.
Its remaining noise came from sampling TRAIN rows, especially rare magnitudes.
For NO2 at 2% equity, only **0.4674%** of rows can cause a floor injection, and
the largest **0.1% of rows supply 80.44%** of expected injection. Its antithetic
row reward has mean **0.6081 EUR**, standard deviation **27.7125 EUR**, giving
MCSE **0.3535 EUR** with the previous 6,144 independent row pairs. Roughly
207,700 independent sampled rows would be needed for 10% relative MCSE at that
one state under this empirical distribution. This explains the instability
of the small unlearned calibration endpoint. It is not market uncertainty.

Every one of the 26 old flat-reference marginal MC intervals contains the
corresponding exact index. Their mean absolute point error is **0.058668
percentage points**. The exact audit takes about **0.4 seconds of numerical
runtime**, excluding Python imports, with one numerical-library thread and
no simulation. Direct seven-probe integration checks the suffix calculation.
The maximum difference between seven-probe interpolation and direct h at all
400 points is **0.086227 percentage points**; both are reported, without
choosing one for better outcome agreement.

## Existing-condition validation: improvement in precision, not universal fit

The ten historical labels separate: largest unlearned index **0.00732351%**,
smallest learned **1.09790174%**, geometric midpoint **0.08966882%**. Excluding
revised FI gives **0.09437240%**. The current FI panel differs from registered
J data; source hashes and the revision flag remain in the output.

The exact index versus stored DP index has Spearman **0.93454** across 26
unique conditions but **0.64294** among the twelve legacy records. Its overall
mean absolute discrepancy is **0.27927 percentage points**. Discount variants
are identical for this one-step diagnostic; it cannot itself model patience.

| Cohort | Raw maximal-probe runs correct | Arm contrasts correct |
|---|---:|---:|
| O, PPO 1M | 46/70 | 6/7 |
| P, PPO 250k | 33/50 | 3/5 |
| P, PPO 4M | 14/20 | 2/2 |
| Q, A2C | 20/40 | 3/4 |
| Q, DQN | 28/40 | 1/4 |

The fixed midpoint incorrectly predicts O's alpha-0.9 treatment as learned.
Removing MC error does not remove fixed-reference, learner, budget or
calibration error. The original exact-DP index still has a different estimand.

Calibration alone permits any separating threshold in
**(0.00732351,1.09790174]%**; this is an identification set, not a confidence
interval. The geometric midpoint is a frozen convention. Most prospective
intermediate conditions are unresolved if robustness over every threshold in
this entire set is demanded. No population reliability claim is made.

## Same-alpha cross-cost candidate

Reanalysis keeps exactly the prior two inspected fees, standard caps/no impact,
both Danish zones and alpha grid. It does not reuse any new learner outcomes.
At the fixed historical midpoint, alpha **0.8** gives opposite confident
predictions at the two fees:

| Fee EUR/MWh | DK1 index % | DK2 index % | g(2% equity), DK1/DK2 EUR | Fixed-midpoint prediction |
|---|---:|---:|---:|---|
| 0.40 | 0.158197 | 0.155498 | 9.863248 / 11.421247 | High in both: at least 1.5c |
| 1.50 | 0.020683 | 0.032693 | 6.258950 / 8.336837 | Low in both: at most c/1.5 |

Thus an alpha-only ranking ties these conditions, while the diagnostic gives
different predictions. Positive one-step reward at the high-fee low-equity
probe certifies that a feasible exploit remains in the empirical model; no
optimal DP is needed for that existence statement. It does not prove maximal
leverage is the optimal multi-period action.

At fee .4 the eligible high alphas are 0,.25,.5,.7,.8 and eligible lows .9,.95;
at fee 1.5, highs are 0,.25,.5 and lows .8,.9,.95. All records and controls are
preserved. These predictions can support a new prospective cross-cost test of
the frozen point heuristic, provided its outcome-informed development history,
threshold convention and precise learner/budget scope are disclosed. They do
not retroactively rescue the previous rollout estimator's failed precision
gate, nor guarantee the new learning outcomes.

## Unexecuted extension

For a fixed continuation policy one can replace sampled immediate monetary
reward by its conditional expectation −mean(c)*action plus expected injection
h(e). This is an unbiased martingale-control-variate estimator of finite-
horizon expected return, while states still follow sampled transitions.
Total trajectory variance need not decrease automatically because immediate
reward and future state value are correlated. A paired all-condition variance
benchmark would be necessary; it was deferred while the exact one-step audit
and prospective design were reviewed. No improved trajectory method is claimed.

Outputs are `exploratory_exact_summary.json`,
`exploratory_exact_condition_predictions.csv`, `exploratory_exact_transfer.csv`,
`exploratory_exact_new_fee_predictions.csv` and
`exploratory_prior_flat_MC_comparison.csv`. The JSON records input/script
hashes, source revisions, exact controls, row-tail diagnostics and runtime.
