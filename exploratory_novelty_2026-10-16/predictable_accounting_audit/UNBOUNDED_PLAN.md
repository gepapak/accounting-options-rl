# Fixed generative validation of predictable accounting compensation

Written before any draws for this new validation, 16 October 2026. Earlier
accounting proofs and empirical results are known. These are constructed-law
capability cases, not independent markets, a learner study, a registration or a
comparison of policy-search algorithms. No old artifact or Study R input is read.

## Fixed sixteen cases and policy

Use every combination of Pareto magnitude shape p in {1.5,2.5}, cost law
`fixed .1` or `proportional .05*Z`, accounting `floor` or `terminal`, and charge
alpha in {.8,.95}: sixteen cases. The magnitude has minimum1 and survival
P(Z>z)=z^(-p) for z>=1. Signs are independent fair +/-1, independent across time
and whole paths. Gross trade payoff is sign*Z. Costs are nonnegative and their
mean is known: .1, or .05*p/(p-1). The proportional law has unbounded costs;
shape1.5 has infinite payoff and cost variance but finite first moments.

Initial equity/booked ledger is2, horizon32 and gamma .99. The frozen controller
trades one unit at time0; afterwards it trades one unit iff the currently
observed ledger is at most initial equity, and is flat otherwise. It sees no next
payoff, uses no charge input and is not tuned by these outcomes. A floor resets
cash at zero with injections. The terminal ledger keeps deficits and forgives
them only on the final transition. Starts are solvent, so the flat baseline is0.

## Analytic oracle and deterministic checks before random draws

Let m=p/(p-1) and call(k)=E[(Z-k)+]. Exactly,

    call(k)=m-k                         if k<=1,
    call(k)=k^(1-p)/(p-1)               if k>1.

An arbitrary positive-part affine payoff follows by rescaling this function
and put(k)=E[(k-Z)+]=call(k)+k-m. The actual evaluation avoids subtracting large
ledger terms. For fixed fee c=.1, nonnegative ledger n has
`h(n)=.5*call(n-c)`; negative n has stable debt drift
`Delta(n)=c+.5*call(c-n)`. Its h0=(m+c)/2. For proportional rate r=.05,
`h(n)=.5*(1+r)*call(n/(1+r))` at n>=0 and
`Delta(n)=r*m+.5*(1-r)*call(-n/(1-r))` at n<0. Its h0=(1+r)*m/2.

Before draws, check h0, c and repair thresholds against exact Fraction values
at both shapes; check the two branch formulas meet at zero, verify a wide
deterministic ledger grid against their moment envelopes, and verify the
one-step initial-equity formula against an independently written closed-form
Pareto-tail expression. No large Monte Carlo oracle-validation run is allowed.

## Two representations on the same paths

1. **Uncapped predictable compensator.** Sum gamma^t times
   `((1-alpha)*h(E_t)-c)*trade` for a floor. For terminal liability sum
   `((1-alpha)*gamma^31*Delta(N_t)-gamma^t*c)*trade`. These observations are
   unbiased and deterministically bounded by known first moments, without
   clipping, finite price support or finite second moments. Use the exact fixed
   initial-action term and action/state-uniform extrema for remaining periods
   to determine a safe range. Do not estimate support from observed extrema.
2. **Upper-capped mean-cost residual.** Replace realized fees in the old residual
   by the predictable known mean c for each executed trade. This preserves
   expectation even when actual costs are unbounded. Let B=sum gamma^t*c*trade
   and F be actual discounted injections/default. The residual
   `(1-alpha)*F-B` is at least `-c*sum gamma^t`. Independently freeze an upper
   cap, then bound the mean of its minimum with that cap. The capped mean is a
   lower estimator of the original expected charged reward.

Direct bounded empirical-Bernstein confirmation on raw charged returns is
unsupported for both positive charge values because their negative tail is
unbounded. No sampled maximum, sampled minimum or sample variance is substituted
for the missing support guarantee. This does not exclude other robust methods.

## Discovery, confirmation and allocated confidence events

Every case receives4096 SEARCH whole paths and exactly65536 new CONFIRM paths,
processed in16 fixed batches of4096. In lexicographic case order p, cost law,
accounting, alpha, assign SEARCH seed202610161100+case_id and CONFIRM seed
202610161200+case_id (case_id0 through15). Distinct phase/case streams are used.
Within one case the two representations use the identical trajectories.

SEARCH chooses only the residual cap from {1,4,16,64,256}; it changes no policy,
case, moment oracle, range or sample size. For each cap use SEARCH mean/variance
in the usual empirical-Bernstein lower formula projected to fixed n65536.
Choose the first maximizer and retain every cap score. The predictable estimator
has no cap or SEARCH-tuned parameter; its SEARCH statistics are descriptive.

Allocate delta=.05/32 to EACH of the two one-sided lower bounds in each of the
sixteen cases. The fixed-sample empirical-Bernstein calculation uses unbiased
sample variance (n-1). A union bound gives95% simultaneous lower coverage for
these32 events only, separately from all earlier experiment families. No
opposite endpoint, post-hoc mean comparison, new charge interval or adaptive
stopping result is allocated. A positive lower bound with mandatory first-trade
cost c>0 certifies a harmful policy; failure is abstention, not safety.

## Analytic negative controls and limitations

For this action set the uniform no-positive-proxy threshold is alpha*=1-c/h0:
29/31 (fixed fee,p1.5),47/53 (fixed fee,p2.5), and19/21 (proportional cost,
either shape). Hence alpha .95 is analytically nonpositive for ALL admissible
adaptive policies in these laws. This is an exact model property, not an extra
statistical event. Report every threshold and actual negative-control bound.
The mandatory first trade gives expected booked return at most -c in each case.

Do not assume variance reduction, positivity at .8, a search advantage, material
real-world economic gain or market generalization. The exact conditional-moment
oracle is an information assumption and must be reported. The two methods have
different conditional-integration work; count oracle scalar evaluations,
vector calls, expected-cost lookups, global payoff draws, active and logical
transitions, CPU/wall time, full-N completion and accounting-check errors.

## Integrity and stopping rule

Write only `run_unbounded_validation.py` and `unbounded_outputs/` in this new
directory, with this plan fixed before the draws. Record all case definitions,
seeds, cap candidates, means, variances, deterministic supports, confidence
bounds, oracle checks, errors, counts and script/plan hashes. Use one numerical
thread. A fixed process-CPU guard at115 seconds permits saving before120 seconds;
if any confirmation budget is incomplete, save INCOMPLETE with no positive
family verdicts and do not continue or alter the design. There is no learner,
DP solve, old artifact mutation, R input or outcome-based extension.
