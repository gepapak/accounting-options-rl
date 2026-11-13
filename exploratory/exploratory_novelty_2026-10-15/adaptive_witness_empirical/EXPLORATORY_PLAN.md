# Exploratory adaptive accounting-witness benchmark

Written before the new benchmark endpoints are computed, 15 October 2026.
Earlier manuscript results are known. This is retrospective method development,
not a preregistration, learner experiment, policy-acquisition predictor or claim
of general safety. No Study R file, DP output or learned outcome is an input.
Only this new directory is written; prior failures and registered files remain.

## Fixed empirical task and policies

Use DK1 and DK2 TRAIN empirical magnitude/cap rows, excluding the first 200,
with IID row sampling and independent fair signs. This is the explicit reduced
independent-sign model, not chronological market execution or an empirical
population guarantee. Transaction fee is the frozen N fee, 1 EUR/MWh; cap,
allocation 20,000 EUR and gamma .99 match its terminal-account dynamics.
Starting ledger fractions {.1,.5,1} and horizons {4,16,48} are fixed subsets of
registered state/clock probes, included in full without selecting on DP/agents.
The target is discounted terminal-forgiven return relative to always-flat,
paired with booked harm for the very same policy and disturbances.

Five methods are retained on every one of the 18 cases:

1. Best standalone one-action policy: trade leverage16 at the last period. For
   initially solvent ledger and nonnegative fee this weakly dominates any
   earlier isolated action; its exact empirical expectation is also reported.
2. Myopic feedback: mandatory first leverage16, then choose {0,4,16} using
   exact one-action/flat-continuation terminal value and remaining clock.
3. Fixed barrier: mandatory first16, then16 while observed ledger is at most
   {1,1.5,2,infinity} times initial ledger (four candidate policies).
4. Structured feedback: mandatory first16, then early/late leverage in {4,16},
   early upper ledger threshold in {1,2} times initial ledger, late threshold
   in {.5,2}, and clock switch in {.5,.75} of horizon (32 candidates).
5. Random parameter search: 32 independently drawn two-stage policies, leverage
   in {0,4,16}, upper thresholds uniform [.5,4] times initial ledger, switch
   fraction uniform [.1,.9], and the same mandatory first16.

Ledger feedback uses the environment's observation clipping [-3,5] allocations.
Policies see no future draws. The mandatory first trade is permitted even when
its isolated one-step reward is negative. Search and confirmation randomness are
separate. No policy, hyperparameter, state or failure is removed after scoring.

Each method receives search budget at most 16,384*horizon logical simulator
transitions per case, dividing paths equally across its candidate policies.
One-action and myopic have one candidate, barriers four, structured/random 32.
Each selected method policy receives exactly 32,768 fresh confirmation paths.
All paths include flat transition slots in the budget. CRN paths are shared
within phases for comparisons; replication units are independent whole paths.
Full empirical-kernel preprocessing is reported separately from simulator calls.

## Accounting reduction, selection and certification

Let X_t=M_t-C_t, with predictable size and E[M_t|past]=0. The booked discounted
return B has E[B]=-E[sum gamma^t C_t]. Terminal forgiveness is
F=gamma^(H-1)*(-L_H)+ for initially solvent ledger. The charged return R=B+F
has the same expectation as Z=F-sum gamma^t C_t. This identity uses independent
fair signs and no anticipative policy, and is not asserted for edge-bearing
markets. Martingale subtraction does not universally reduce variance.

All policies have one predetermined mandatory16 action (at the last time for
the one-action baseline, at time0 for others), so E[B] is at most the negative
of that action's exact discounted expected cost. This supplies strict booked
harm without a noisy booked-return significance test. Fully booked controls
therefore have no positive witness under this same no-edge law.

Search jointly chooses a policy and a cap U in {32,128,512,2048} EUR by maximizing
a conservative empirical-Bernstein score for min(Z,U). It uses no confirmation
draw. Since Z >= -4*fee*sum gamma^t and min(Z,U)<=Z, a positive independent lower
confidence bound on the clipped residual certifies positive E[R]. This remains
valid despite clipping's downward bias; failure is abstention, not safety.

On the SAME selected policy and confirmation paths, compare lower confidence
bounds for raw charged return, untruncated accounting residual, clipped raw
return and clipped residual. A raw-return cap is separately chosen using SEARCH
only from the same fixed cap grid, conditional on the same selected policy;
this separates clipping from the accounting reduction. Raw/residual finite-support bounds use the whole TRAIN support; they
may be too conservative. The clipped width uses only U and maximum cost.
There is no claim that the raw or residual bound must be useful.

Use the one-sided empirical-Bernstein inequality of Maurer and Pontil (COLT
2009, Theorem4), with sample variance denominator n-1. For variables in [a,b],
LCB=mean-sqrt(2*s2*log(2/delta)/n)-7*(b-a)*log(2/delta)/(3*(n-1)); UCB reverses
the signs. Apply it to the reflected normalized variable for the lower bound.
Use delta=.05/(18 cases * 5 methods * 6 bounds), covering all four return
lower bounds plus forgiveness lower and cost upper bounds. The union bound
does not need comparisons to be independent. Conditional on SEARCH selection,
all selected policies/caps are fixed for independent confirmation.

Secondary: bound E[min(F,U)] below and E[discounted cost] above. A fixed policy
then obeys J_alpha >= (1-alpha)*L_F-U_C simultaneously for all alpha in [0,1],
certifying an interval alpha < 1-U_C/L_F when nonempty. This is the same policy
in counterfactual partial-terminal-forgiveness accounts; it supplies no safety
claim outside its interval and no new policy-selection or alpha-specific fit.

Record all opportunities, selected parameters, errors/abstentions, interval
endpoints, exact one-action values, raw/controlled variances, budgets and CPU
time. Stop by 110 seconds process CPU and save a partial status if necessary;
never exceed the requested two-minute CPU budget. Numeric threads are one.

Source for the classical confidence inequality:
https://www.cs.mcgill.ca/~colt2009/papers/012.pdf

The clipped accounting-residual bound itself does not need bounded prices or
their second moment: finite first-moment predictable gross payoffs and bounded
costs suffice, because Z>=-Cmax and min(Z,U)<=Z. The empirical raw comparator
uses its finite-panel support and the tight terminal-account telescoping bound;
the experiment remains conditional on its empirical IID model. This structural
left-bound advantage is distinct from a variance-reduction claim. For gamma1
or an undiscounted terminal owner payoff, raw return can itself have a cash-only
lower bound; no universal superiority is claimed.

## Post-run scope correction (no policy or outcome selection)

Independent code review noted that the implemented baseline/reference scanned
only leverage16, so the initial phrase 'best standalone one-action' was too
broad. All initial scripts, plans and outputs are preserved under
`initial_max16_reference/`. The benchmark policy remains leverage16-at-deadline,
but its exact reference is strengthened to scan all original allowed leverage
magnitudes {0,.25,1,4,16}. Both the max16 and the stronger all-level reference are
reported. 'Myopic' is renamed 'forced-prefix myopic' because the mandatory
initial16 may itself have negative isolated value. Policy actions, cap selection,
search/confirmation seeds, sample sizes, certification family and all original
failures are unchanged. The corrected execution reuses exactly the same random
paths, rather than adding evidence or stopping when a desired result appears.
