# Known paired-sum cash inversion: exact scope

This is an exploratory extension of the existing cash diagnostic to the original
settlement engine's ordered writes. It is elementary inverse accounting, not a
new learning algorithm. The V1 hand protocol and outcomes are unchanged.

Let the reported cash immediately before the unknown aggregate settlement be
`a >= 0`. Write the known subsequent signed cash receipts as `w_1,...,w_n`.
Every write uses `cash <- max(0,cash+write)`. Exact real arithmetic gives

`f(z) = max(z+A,T)`,

where `A=a+sum(w_i)` and `T` is obtained from `T_0=0`,
`T_i=max(0,T_(i-1)+w_i)`. This identity follows by induction using
`max(0,max(z+A,T)+w)=max(z+A+w,max(0,T+w))`.

The shadow must share the cash prefix, queue and known suffix. Suppose its
aggregate gross is `K-z`, where K is known from accepted own contracts and a
predeclared public intervention. Put `h=T-A`. For observations `y=f(z)` and
`y'=f(K-z)`, the following is the **sharp real-valued identified set**, before
intersection with other common public support information:

* If `y>T`, then `z=y-A` is unique, subject to consistency with `y'`.
* If `y'>T`, then `z=K+A-y'` is unique, subject to consistency with `y`.
* If `y=y'=T`, then `z in [K-h,h]`. The pair is possible iff `K<=2h`.
  It is unique when `K=2h`, and has unavoidable ambiguity when `K<2h`.

Each statement is a direct inverse of the two piecewise affine maps. Every
point in the last interval produces the same two observations, establishing
sharpness and impossibility of exact recovery from this interface alone. A
common public feasible set I is intersected with these sets; it may remove
ambiguity. Known costs only translate the gross set into the booked set.

One branch alone identifies z whenever its cash exceeds T. At its floor it
only establishes `z<=h` (plus the same public support). Therefore gains over a
one-branch cost-aware baseline must be measured on actual censored events.
The fee-before-floor V1 counterexample is already solved by that baseline;
its correction cannot be credited to the twin query.

At an isolated settlement boundary with no suffix, `A=a>=0,T=0`. If K=0,
the exact pair is always unique: both branches can be floored only at a=0,z=0.
This is precisely a local zero-cost TIR baseline. The broader ordered-write
map helps only when that additional interface or antisymmetry is unavailable.

For a linear MWh contract with observed executed quantities q_i and observed
entry prices P_i, reflect each private delivery settlement around a public,
predeclared center R_d: `S'_d=2R_d-S_d`. Then

`K=2 sum_i q_i (R_(delivery_i)-P_i)`.

Different entry prices at the same delivery and net-zero quantity are allowed.
The evaluator may read the paid settlement to make the shadow. The restricted
estimator receives neither settlement prices nor G; it receives cash, own
contracts and centers. This is an explicitly restricted audit interface: the
original engine itself exposes settlement P&L and prices, so direct-ledger and
settlement-price reconstruction are mandatory information/latency baselines.

Binary64 is a separate statement. `known_sum_v2.py` propagates a conservative
public bound on the discrepancy `G_real+G_shadow-K`, then intersects relaxed
cash inverses. Cash additions must be independently certified to use operands
and exact pre-floor sums within +/-2^50. The resulting interval contains the
represented source gross under those premises. It is **not** a sharp discrete
IEEE set, a finite-sample confidence interval, or proof that a floating-point
singleton was recovered. Actual widths and abstentions must be reported.

The 96 own-code arithmetic cases in KNOWN_SUM_PURE_CHECKS_V2.json verify
containment and include repeated deliveries with different entries, signed
quantities, decimals and net-zero positions. Twenty-six narrow the interval
relative to the real-only inverse with the same own-fill price support. These
are pure arithmetic checks, not additional native trajectories or an empirical
frequency estimate. The positive-cash/negative-booked suffix example is also
only arithmetic; it does not establish native-policy reachability.
