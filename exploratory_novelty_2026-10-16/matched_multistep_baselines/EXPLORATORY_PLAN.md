# Matched multi-step baselines for the selected empirical witness

Written before any new draws, 16 October 2026. The prior full-clock comparison
and selected-case positive certificate are already known. This is a disclosed
exploratory follow-up on that same selected state, not new market validation,
learner training or a comparative policy-search benchmark. No old artifact,
registered source, output, plan or Study R file is changed or used for tuning.

## Frozen law and policies

Use the same DK1 TRAIN empirical rows after the first 200, independent IID row
sampling and independent fair signs, terminal forgiveness, initial booked cash
10,000 EUR, fee 1 EUR/MWh, allocation 20,000 EUR, cap 25% and gamma .99. The global
episode has 96 periods. The old frozen adaptive controller is read verbatim
from `exploratory_novelty_2026-10-15/adaptive_witness_empirical/selected_case_followup.json`:
mandatory leverage 16 initially, then 16 below/equal initial cash during the first 48
periods and below/equal twice initial cash during the last 48, otherwise flat.
Observed ledger clipping and action sizing match the existing source-checked
kernel. This controller is not changed after these results.

Evaluate nine unique predetermined policies:

1. Frozen adaptive controller, active duration 96.
2. Constant leverage 16 throughout 96 periods.
3. Constant leverage 4 throughout 96 periods.
4. Constant leverage 1 throughout 96 periods.
5. Constant leverage .25 throughout 96 periods.
6–9. Open-loop leverage 16 for the final 1, 4, 16 or 48 periods, flat beforehand.

The requested open-loop 16 for 96 periods is exactly the constant 16 policy. Keep
its alias in output metadata without counting it as an additional test or
simulating it twice. This equivalence is established before observing data.
The exact empirical isolated-action reference at the deadline is integrated
for all allowed leverage magnitudes {0,.25,1,4,16}, without Monte Carlo error.

## Matched independent search and confirmation

Every unique policy receives 8192 SEARCH whole paths and 524288 independent
CONFIRMATION whole paths. SEARCH seed 2026101601, CONFIRMATION seed 2026101602,
separate source-check seed 2026101603 are fixed here. Share each phase's global
96-period disturbance paths across policies for a paired comparison. Individual
whole paths remain IID, and no confirmation draw selects a policy or cap.
Process confirmation in 64 fixed batches of 8192 rather than allocating an
N-by-horizon array. Flat prefixes need no active transition evaluation.

For each fixed policy separately, SEARCH chooses a residual upper cap from
{2048,8192,32768,131072} EUR. Maximize the empirical-Bernstein lower score
projected to the fixed confirmation n=524288 using SEARCH mean/variance. Select
the first maximum, retain all scores and freeze the cap. There is no policy
search, new parameter optimization, outcome-based exclusion or confirmation
retuning. The larger cap grid and confirmation budget are matched across all
policies, addressing the mismatch in the previous selected-case follow-up.

## Familywise certificate and interpretation

For each policy, let B be discounted booked payoff, C discounted executed cost,
F=gamma^95*(-L_96)+ and Z=F-C. Predictability and fair signs imply
E[charged]=E[Z]. Nonnegative bounded fees give Z>=-Cmax for a deterministic
policy-specific Cmax. Bound E[min(Z,U)] below by the fixed-n one-sided
Maurer-Pontil inequality, using sample variance denominator n-1. Allocate
delta=.01/9 per bound for 99% simultaneous coverage over all nine tested
policies in THIS new family; dependence from shared paths is allowed. This
does not combine with previous families into a paper-wide 99% guarantee.

For constant/open-loop policies, E[B] is exactly minus the single-step expected
cost times its active discounted-duration sum. For the adaptive policy retain
the exact negative upper bound minus the mandatory initial action's expected
fee. No additional noisy booked-harm test is needed. Report raw, untruncated
residual, booked, fee and forgiveness sample means descriptively; they are not
additional allocated confidence statements or hypothesis tests.

A positive constant-policy lower bound establishes that simple multi-step
exposure suffices at this state. Nonpositive lower bounds are ABSTENTIONS,
not proof that the baseline is nonprofitable or adaptive feedback necessary.
Even if the adaptive controller confirms and all constants abstain, do not
claim optimizer superiority, necessity, learner acquisition or market transfer.
Do not compare policy means inferentially without a separately planned test.
Preserve all results including a possible negative conclusion for novelty.

## Integrity, calls and runtime

Record all policy definitions/aliases, all cap scores, fixed-n certificates,
exact isolated-action values, booked harm, hashes of panels/old source/old
selected-controller output/this script/plan, and source-validation errors.
Replay every policy against the unchanged frozen terminal environment on
independent source-check paths. Numeric threads are one. Count active simulated
transitions separately from global outer time slots and validation steps.

Use a per-step process-CPU deadline at 116 seconds, leaving time to save under
the 120-second limit. If any policy has fewer than 524288 confirmation paths,
save partial statistics and an INCOMPLETE family status with no positive
inferential verdicts. No continuation, larger sample, new cap grid or policy
adjustment is permitted after seeing confirmation. Do not launch any training,
DP solve or other benchmark from this script.
