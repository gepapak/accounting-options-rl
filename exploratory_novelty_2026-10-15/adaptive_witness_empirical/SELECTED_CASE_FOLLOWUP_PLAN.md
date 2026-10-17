# Selected-case independent certificate follow-up

Written before any new follow-up draw. The final full-clock comparison found
zero certified cases beyond the exact all-allowed-leverage one-action reference,
but some untruncated residual point estimates were positive. Those old outputs
are now development data for a new selected-case test. Their original counts,
failures, confidence families and logs are preserved; this does not change the
38/70 final benchmark count or establish a new comparative discovery rate.

## Freeze one case using old data only

Include **every** full-clock selected controller whose state's exact all-level
one-action reference is nonpositive (flat included). This gives 25 eligible
state/method records. Preserve their complete ranking and parameters. Select
the first maximum of this fixed projected-score heuristic:

`old residual_mean - sqrt(2 * old residual_variance * log(2/.01) / 524288)`.

The score uses an untruncated old mean/variance to predict mean-estimation noise
at the planned future sample size. It is only a selection heuristic, not a
confidence bound: it omits a range penalty, which is fixed by the new cap choice
below. Freeze the old record's exact policy, local duration, initial ledger,
zone, fee, gamma and fair-sign empirical law. No further policy/state search is
permitted after this selection.

## Fresh cap selection and independent confirmation

Use a fresh SEARCH stream with seed3401000 and 8192 whole paths to select a
residual upper cap from {2048,8192,32768,131072} EUR. Score each cap using its
search mean/variance and the fixed future confirmation n524288 in the ordinary
one-sided empirical-Bernstein formula, delta=.01. Select the first maximum;
retain all four scores, then freeze this cap. Prior seeds2401000 were already
used in source validation, so they are deliberately not reused here.

Use independent CONFIRM seed3501000 for **exactly 524288** IID whole paths,
processed in 64 fixed batches of8192 to avoid allocating N-by-duration arrays.
Compute ONE allocated lower bound for min(Z,U), with known range
[-4*fee*sum gamma^t,U], sample variance denominator n-1 and delta=.01. This is a
conditional 99% statement about this one independently selected controller and
fixed empirical law, not 99% coverage over old families or real markets.
Mandatory initial16 supplies exact strict booked harm, so no second noisy
statistical test is required. Same-action full charging remains an analytic
negative control. Report raw and untruncated residual means descriptively only.

The local certificate may be compiled to later deadlines H>=T by deterministic
gamma^(H-T) scaling. No new alpha interval or alpha/horizon statistical family
is fitted in this follow-up. Failure to certify is abstention, not safety.

Record every candidate eligible for selection, all cap scores, frozen policy,
full-N bound or incomplete status, source hashes, phase seeds, actual transition
count, CPU/wall time and accounting identity errors. A separate source-check
stream seed3601000 can replay the selected feedback policy against the unchanged
frozen terminal environment; its paths are never search or confirmation data.

Use one numeric thread and a per-step process-CPU guard at97 seconds to leave
time to save before100 seconds. If all524288 paths do not complete, save
INCOMPLETE with **no inferential verdict**, and do not continue. No new seed,
cap, sample-size, controller or case adaptation is authorized within this plan
after viewing confirmation. This is a selected-case capability test, not a
new performance-grid extension or independent market validation.
