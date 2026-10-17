# Cash-funded learned controls: pre-statistical design review

16 October 2026. This design amendment precedes every v2 statistical development
or confirmation draw and source freeze. Two fixed deterministic native software
paths have passed; they are excluded from inference. No v2 statistical financial
outcome was used to choose this amendment. Original run001 remains immutable.

## Reason for the amendment

The new cash-funded RAW baseline has lower endpoint -K, while the existing
paired/residual learned representations use -Cbar-B. A win for the former would
not show that learning cannot help a cash-funded representation. Include both
cash-funded Ridge and MLP controls before drawing statistical paths, with the
same cash endpoint and independent FIT/TUNE/CONFIRM separation. Retain every
original representation. This is a stronger comparator design, not a new
generic control-variate principle or a promised learning improvement.

## Target, fitting and predictable coefficient

Let x_k be the discounted actual new-window financial increment at event k:
all forward costs and cost injections are included; direct inherited maturity
gross and injections are excluded. Let Y=sum_k x_k. All actual inherited state
and all old maturity coins remain in the trajectory. The reviewed cash-inflow
contract gives Y>=-K.

For each active maturity coin epsilon_k, FIT-only actual-branch rows fit the
RAW future target R_k=sum_{j>k} x_j with the same 32 common/branch features used
by the paired-continuation models. Separate Ridge/MLP models use seed40300000;
paired-continuation models retain seed39300000 and their original target. Both
model pairs keep the same architecture, preprocessing and stopping settings.
Neither model pair is trained on TUNE or CONFIRM.

For a new maturity, the known current two-branch difference is

    d_local,k = w_k * [g0_k + (Iplus_k-Iminus_k)/2].

Its cost is common to both branches and cancels. For an inherited maturity,
direct gross/injection are excluded, so d_local,k=0. A retained deterministic
cost has no active coin. Use

    delta_k = d_local,k + (q_raw(H_k,+1)-q_raw(H_k,-1))/2.

This includes the current RAW contribution rather than asking a future-only
predictor to estimate it. Both branches share the same frozen predictor and
all common features. The local calculation and prediction use no current coin.
An arbitrary finite inaccurate q preserves the following inference.

Clip beta*delta to the remaining absolute budget B before observing the current
coin; debit its absolute amplitude. Thus M=sum epsilon_k*a_k has EM=0 and
|M|<=B in real arithmetic. It follows that

    T=min(Y-M,b) lies in [-K-B,b], and ET<=EY.

If the separate gross upper envelope Gbar is used, the upper endpoint can be
min(b,Gbar+B). This optional upper improvement and the retained numerical
allowance require additional engineering source access; the weakest theorem
only needs K,b,B and trusted conditional sign centering. A formal floating-point
proof is not claimed. The predeclared conservative allowance remains in all
confidence calculations. Booked harm still uses its separate no-edge/fee proof.

## Fixed comparison and multiplicity

Add alpha0-only `cash_ridge_control` and `cash_mlp_control`. They select beta,
budget and cap on TUNE32 using the same fixed 11 caps, five beta values and five
relative budgets as the original learned representations. Cash capped RAW and
other simple capped methods retain64 development episodes; disclose this
selection-data difference. Every representation shares all320 physical paths
per policy. Every selected representation uses256 independent CONFIRM paths.
RAW and cost limits use320 unselected paths.

Also allocate `cash_local_control`, with delta=d_local and future predictor
identically zero, using the SAME TUNE32 selection and candidate grids. With
beta1 and an untruncated budget, subtracting the known current contribution
alone transforms RAW into the local paired residual. A win over cash RAW could
therefore come from this analytic component, without learning. The separately
allocated local-only baseline and matched zero-future-q descriptive ablation
are necessary before attributing an improvement to the learned continuation.

There are10 alpha0 representations,6 alpha1 representations and1 cost limit
per policy:20*(10+6+1)=340 limits, each delta=.05/340. All320 charged choices and
both model bundles must freeze before CONFIRM. The cash representations have
no alpha1/partial-charge claim. This new95% family is separate from the original
520-limit95% and auxiliary240-limit95% families; their joint coverage is at
least85%, not a paper-wide95% guarantee.

Report uncapped and same-cap zero-control means/variances, plus the same selected
learned cap/beta/budget with future predictor set to zero, as descriptive
ablations, including undefined/zero ratios. Do not turn unallocated ablation
limits into additional discoveries. Track all four regressors' fitting and
prediction costs separately from shared native work. No additional simulator
path or RL training is needed. A tie, zero control weight, higher variance or
abstention remains a substantive result. Neither a win nor failure establishes
general learning superiority, acquisition or future-market performance.

## Remaining gates

Independent source review, literal current/future masking tests, large-budget
arithmetic checks, public/private reproducer agreement and complete input/cache
bindings must precede statistical release. This memo authorizes no execution.
It must be included by hash in the eventual v2 source freeze. Any scientific
change after that freeze requires an explicitly separate study, not an edit.
