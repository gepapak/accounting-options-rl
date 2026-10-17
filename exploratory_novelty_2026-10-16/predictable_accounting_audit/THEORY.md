# Predictable accounting compensators: exact theory and scope

Exploratory mathematical development, 16 October 2026. This note and its companion
exact rational checks do not alter earlier artifacts, train a learner, simulate
Monte Carlo paths or read Study R outcomes. The directory also contains separately
planned generative and empirical confirmation experiments; their plans and
recorded outputs state their own sampling and coverage assumptions. The
construction specializes classical predictable-compensator/martingale-control
theory; it is not a new general Doob-decomposition or concentration theorem.

## 1. Model and the information actually required

The horizon H is finite, 0<gamma<=1, and alpha is in [0,1]. At time t the action
a_t is predictable: it is chosen before the next gross payoff G_t and executed
cost C_t. Let X_t=G_t-C_t. Conditional on the available history and action,

    E G_t=0,     E|G_t|<infinity,     C_t>=0,     E C_t=c_t<infinity.

Costs themselves need not be almost-surely bounded for the present reduction.
Known uniform bounds on the conditional first moments below suffice; bounded
costs are one simple sufficient case. To evaluate the estimator, its conditional
stop-loss curve and conditional mean cost must be known or integrated exactly.
The unconditional absolute-payoff and cost expectations used by the tower
argument must also be finite; the uniform envelopes below ensure this. Almost
sure finiteness of a conditional moment alone is not an unconditional bound.
A declared finite empirical row kernel and independent fair signs provide this
access. An analytic distribution with evaluable conditional stop-loss also works.
Merely assuming a finite first moment does not give its numerical upper bound.
Plugging a noisy, unbounded Monte Carlo estimate into the curve does not inherit
the stated deterministic bounds or exact unbiasedness without another argument.

For a stationary cash-independent action law, define

    c_a=E C_a,   d(n)=(-n)+,
    h_a(n)=E d(n+X_a),   h_a0=h_a(0),
    Delta_a(n)=h_a(n)-d(n).

The law must remain the same as n varies for statements comparing this curve to
its value at zero. More generally, for each history use its CURRENT conditional
law to evaluate the counterfactual zero-ledger moment h_t,0. The results remain
valid if known constants uniformly dominate these conditional moments and costs.
That counterfactual value need not equal the actual simulator's zero-state law.
Predictable, history-dependent policies are otherwise allowed.

## 2. Sharp one-step debt-drift and floor bounds

Since E X_a=-c_a<=0, Jensen and the monotonicity of d give

    E d(n+X_a)>=d(n-c_a)>=d(n),

so Delta_a(n)>=0. More precisely,

    n>=0:  Delta_a(n)=h_a(n),
    n<0:   Delta_a(n)=c_a+E(n+X_a)+.

On n>=0 the drift decreases from h_a0 to zero. On n<0 it increases from c_a
to h_a0. Dominated convergence supplies the limits. Consequently

    0<=Delta_a(n)<=h_a0                 for all real n,
    c_a<=Delta_a(n)<=h_a0              for n<0,
    0<=h_a(e)<=h_a0                    for e>=0.

These are sharp infimum/supremum bounds. The maximum is attained at zero. For
unbounded support, a limiting lower endpoint need not be attained at a finite
state. Also h_a0=c_a+E(X_a)+ >= c_a. For independent fair signs G=epsilon Z,
Z>=0, with a joint magnitude/cost row independent of the sign,

    h_a0=c_a+(1/2)E(Z-C_a)+.

For an interval of allowed terminal ledgers [l,u], the maximum of Delta is at
the point of that interval closest to zero, and its minimum is the smaller
endpoint value. For allowed solvent floor equities [l,u], h's range is exactly
[h(u),h(l)]. These can tighten bounds when a policy's reachable states are known.

The cash-independent assumption matters. Let cost=.1 everywhere, with fair
gross +/-1 at ledger 0 and fair gross +/-4 at ledger 1. The actual zero-state
h(0)=.55, but the debt drift at ledger 1 is 1.55. The bound by the actual zero-state
curve fails. The same-current-law zero moment at ledger 1 is 2.05 and remains a
valid bound. Cash-scaled exposure or state-dependent risk therefore needs a
uniform conditional envelope, rather than automatic reuse of the zero-state law.

## 3. Unbiased bounded whole-path estimators without upper clipping

### Terminal liability

Let N_{t+1}=N_t+X_t and D_t=d(N_t). Values below are advantages over always flat
under the SAME account. This distinction subtracts existing debt forgiveness
when N_0<0; when N_0>=0 the flat value is zero. Terminal forgiveness is paid on
the final transition, hence its weight is gamma^(H-1), not gamma^H. Write

    Y_alpha=(1-alpha)gamma^(H-1)(D_H-D_0)-sum_t gamma^t C_t.

Tower conditioning and telescoping give

    J_alpha=E Y_alpha=E S_alpha,
    S_alpha=sum_t [(1-alpha)gamma^(H-1)Delta_{a_t}(N_t)-gamma^t c_{a_t}],
    J_book=-E B,             B=sum_t gamma^t c_{a_t}.

S_alpha is a predictable accumulated drift, not the realized ledger reward.
The omitted martingale has expectation zero by finite horizon and integrability.
No clipping bias or tail correction occurs. The same statement holds for
history-conditioned curves/moments with their declared uniform envelopes.

For a fixed action at time t, its sharp state-uniform drift interval is

    [-gamma^t c_a, (1-alpha)gamma^(H-1)h_a0-gamma^t c_a].

If n<0 is known, improve its lower endpoint to
[(1-alpha)gamma^(H-1)-gamma^t]c_a. For a finite action family including flat,
put c_max=max_a c_a and

    u_t(alpha)=max_a{(1-alpha)gamma^(H-1)h_a0-gamma^t c_a}.

Then S_alpha lies in [-c_max sum gamma^t, sum u_t(alpha)]. These additive
whole-path bounds are safe but need not be simultaneously attainable by one
trajectory. Per-action/state bounds are sharp; a tighter full-path support claim
would need reachability analysis. If the initial state/action are fixed, the
first drift term is a known constant and should be removed from the random
range; a one-action policy is integrated exactly and needs no rollout estimate.

### Step-floor accounting

Let E_{t+1}=E_t+X_t+I_t with I_t=(-E_t-X_t)+ and E_t>=0. The charged reward is
X_t+(1-alpha)I_t. Its unbiased predictable estimator is

    S_alpha=sum_t gamma^t[(1-alpha)h_{a_t}(E_t)-c_{a_t}].

Its sharp action/state-uniform one-step range is

    [-c_a, (1-alpha)h_a0-c_a].

Thus S_alpha lies in [-c_max W_H,u(alpha)W_H], where W_H=sum gamma^t and
u(alpha)=max_a{(1-alpha)h_a0-c_a}. Terminal and floor weights differ and must
not be exchanged. Settlement lag, repayments, resets or early liquidation need
their own ledger/potential identity; neither formula transfers automatically.

### Independent confirmation and booked harm

Freeze a candidate policy on discovery data. If the conditional curves and
uniform moment bounds are exact/known, S_alpha is a bounded iid whole-path
observation under independent confirmation trajectories. Standard bounded
one-sided confidence inequalities now apply to its UNCLIPPED expectation,
even when G, C or the original terminal debt have unbounded support and infinite
variance. Existing family-wise selection rules still apply. This removes the
old upper-truncation bias but does not promise a smaller numerical range or
better variance/power in every case.

A mandatory trade at deterministic t0 with positive expected cost gives
J_book<=-gamma^t0 E c_{a_t0}<0. Alternatively a separate positive lower bound
on E B establishes strict booked harm. Nonnegative costs alone give only
nonpositive booked expectation; a zero-cost strategy is not a strict-loss witness.

## 4. A simultaneous fixed-policy charge family

For a policy and dynamics that do not change with alpha, define the predictable
forgiveness A by sum gamma^t h(E_t) for a floor or gamma^(H-1)sum Delta(N_t) for
terminal liability. Both A and B=sum gamma^t c_a are nonnegative and bounded
by known first-moment constants. Moreover S_alpha=(1-alpha)A-B.

Simultaneous bounds E A>=L_A and E B<=U_B therefore imply

    J_alpha >= (1-alpha)L_A-U_B       for EVERY alpha in [0,1].

With L_A>U_B>=0 and E B>0, alpha<1-U_B/L_A certifies a harmful interval for
this fixed controller. No union over alpha is needed. Exact conditional c_a
can also make part or all of B deterministic, eliminating that bound's noise.
The interval is not a global repair threshold and gives no safety above it.
Retraining or changing transitions with alpha invalidates shared-policy reuse.

## 5. A kernel-specific uniform partial-charge no-exploit certificate

Define rho=sup_a h_a0/c_a over positive-cost actions. An action with c_a=0<h_a0
makes rho infinite; actions with c_a=h_a0=0 are harmless and can be ignored.
If no nonflat positive-cost action remains, treat that degenerate case directly.
Otherwise rho>=1. A finite rho, or a valid uniform conditional ratio bound,
gives the threshold

    alpha_* = 1-1/rho.

For alpha>=alpha_*, every floor drift is nonpositive. Every terminal drift is
also nonpositive because gamma^(H-1)<=gamma^t for t<H. Consequently J_alpha<=0
for ALL admissible adaptive policies and every finite horizon, relative to the
same-account flat baseline. With kappa=1-(1-alpha)rho>0,

    J_alpha <= -kappa E B,

so any positive-cost policy has strictly negative expected charged reward.
At the boundary only a nonpositive guarantee is asserted.

This all-state criterion is sharp. If one action violates
(1-alpha)h_a0<=c_a, one trade from equity0 (or a sufficiently small positive
equity) has positive charged expectation. For a fixed terminal H, wait flat
until the last step and take that action. A zero-cost action with h_a0>0
prevents any alpha<1 guarantee. These statements do not establish necessity for
one fixed initial distribution away from zero or a restricted controller class.

The threshold is market/kernel/action-set specific. It does not contradict the
absence of one fixed partial charge that deters exploitation uniformly over
arbitrarily changing markets; h_a0/c_a can be arbitrarily large. It is a
model-conditional no-positive-expected-proxy certificate, not a guarantee about
real markets with conditional edge, estimated tails, nonlinear transformed
rewards, changed dynamics or omitted costs. Full charging always gives the
separate no-edge booked-objective guarantee.

## 6. Limits and exact counterchecks

1. **No universal variance reduction.** Set H=2, gamma=1, E0=2, G identically 0. The
   first action pays fee 0 or 1 equiprobably. If it paid 0, the second action pays
   deterministic fee 1; otherwise stay flat. Actual cumulative cost is exactly 1,
   so actual charged/booked/residual return is -1 with variance 0 and no floor.
   Predictable cumulative cost is .5+1 or .5+0, so S has variance1/4. The
   operation is not the conditional expectation of the whole return given one
   common sigma-field. Removing local innovations can destroy cancellation
   with future feedback. Per-step conditional integration alone is not a
   universal whole-path Rao--Blackwell improvement.
2. **No-edge is essential.** At ledger -1, deterministic G=1 and cost .1 give
   Delta=-.9, contradicting nonnegative drift. With positive conditional edge,
   booked expectation need not be negative and neither repair theorem follows.
3. **Finite moments alone are not uniform numeric envelopes.** A family of
   fair gross +/-k with cost 1 has finite moments at every k but unbounded h_k0
   as k grows. An unbounded action/state family needs an explicit envelope.
4. **Known-kernel scope.** Exact empirical integration is conditional on that
   finite law, not future-market confidence. Independently learned conditional
   curves need valid error propagation. Merely fitting a critic supplies none.

The companion standard-library script enumerates finite-law trajectories of
fixed controllers with Fraction arithmetic, verifies drift/expectation/range
identities at floor and terminal accounts (including an insolvent flat baseline),
checks the alpha family and sharp repair boundary, and records all counterchecks
in JSON. It performs no Monte Carlo, policy optimization, training or DP.

## 7. Prior-art and novelty assessment

The general mechanism is classical predictable compensation: replace an
unbounded potential increment by its conditional drift and add/subtract the
associated zero-mean martingale. Henderson--Glynn (2002), Proposition 1, already
uses this construction with potentially unbounded potentials and bounded drift.
The one-step inequalities here follow Jensen, positive-part identities and
monotonicity. Standard bounded confirmation then supplies the inference.

The useful accounting-specific advance over the earlier clipped audit is an
explicit sharp debt-drift envelope, an unbiased bounded whole-path estimator
without clipping under conditional first-moment access, and a uniform
kernel-specific partial-charge repair criterion. Those are actionable structural
reductions, not new martingale theory or a generic black-box AI method. Empirical
benefit and comparison with established martingale controls must be measured;
neither unbiasedness nor a smaller worst-case support proves superior variance,
search efficiency, target-learner acquisition or real-market validity.

The repair threshold itself is also an established financial ratio:

    alpha_* = 1-c/h0 = E[(X)+] / E[(-X)+].

The gain/loss ratio and related piecewise-linear utility comparisons precede
this work: Bernardo and Ledoit, "Gain, Loss, and Asset Pricing," Journal of
Political Economy 108(1):144--172 (2000), DOI10.1086/262114,
https://www.journals.uchicago.edu/doi/10.1086/262114 ; see also Biagini and Pinar,
SIAM Journal on Financial Mathematics 4(1):228--242 (2013), DOI10.1137/120866774,
https://arxiv.org/pdf/1209.6439 . Their unrestricted zero-cost trading/duality
setting is different, so no literal transfer of that theorem is asserted. The
claim here is only its explicit uniform application to all adaptive floor and
discounted-terminal policies under the stated kernels and costs, not a new
gain/loss ratio or general expected-utility principle.

## 8. Charged-value evidence can itself certify booked harm

For finite rho under the uniform action/conditional-ratio assumptions above,
put kappa=(1-alpha)rho-1. The same pointwise proof gives, for EVERY admissible
fixed or history-dependent policy,

    J_alpha <= kappa E B,       J_book = -E B,
    B = sum_t gamma^t c_t.

For a floor, (1-alpha)h_t(E_t)-c_t <= kappa*c_t. For terminal liability,
(1-alpha)gamma^(H-1)Delta_t(N_t)-gamma^t*c_t <= kappa*gamma^t*c_t,
because Delta_t<=rho*c_t and gamma^(H-1)<=gamma^t. Sum and condition as before.

Consequently, if kappa>0 and a valid charged-value lower bound ell>0 is obtained
for a frozen policy, then E B>=ell/kappa and

    J_book <= -ell/kappa < 0.

No mandatory initial trade, lower bound on trading probability, or second noisy
cost test is required. The policy may remain flat on arbitrarily many histories.
This permits checking arbitrary frozen learned controllers within the declared
predictable-information/kernel model, rather than only a forced-prefix grammar.
An always-flat policy cannot produce a genuinely positive charged lower bound.
The oracle must condition on ALL information the controller observes: exposing
current magnitude/capacity before action can invalidate unconditional row means
and can make the required uniform ratio infinite.

The positive ell event carries its existing error probability; the harm bound
is deterministic on that same event, so no extra statistical test is introduced.
The result is not obtained by dividing by kappa when kappa<=0. In that region
the model already gives J_alpha<=0. If a zero-cost action has h0>0, rho is infinite
and positive charged value need not entail strict booked loss: fair gross+/-1,
cost0, alpha.8 and an optional trade with probability1/4 have charged value1/40
and booked value0. The finite-rho condition is essential.

For example, fair gross+/-1 with cost2/5 gives h0=7/10 and rho=7/4. A controller
that remains flat with probability3/4 and otherwise trades once at equity0 has
J0=3/40, E B=1/10 and kappa=3/4. The exact conservative lower bound ell=3/80
therefore implies J_book<=-1/20; its actual booked expectation is-1/10. This
example has no mandatory trade. `exact_harm_corollary.py/json` separately check
120 optional-trade account/state/time/charge cases, with a zero-cost failure
and a conservative-envelope example; the earlier exact and generative JSON
records remain unchanged.

### Sharpness is a property of the exact admissible kernel

The threshold in Section5 is an if-and-only-if ALL-STATE criterion only when
cash-independent action laws are used exactly and each offending action is
admissible at zero (or arbitrarily small positive equity). Its global necessity
uses that available action from zero at the final period. A loose conditional
moment/ratio envelope yields only sufficiency: doubling rho from7/4 to7/2 changes
the envelope threshold from3/7 to5/7, although the true kernel is already uniformly
repaired at alpha1/2. Likewise, if a trade is prohibited below ledger2 in the
bounded +/-1,cost.4 example, it cannot create a floor or terminal debt from a
solvent start, even at charges below the unconstrained zero-state threshold.
Neither a loose envelope nor unreachable/inadmissible zero-state actions justify
an iff assertion for the actual controller class.

A simultaneous alpha curve estimated from one fixed rollout family requires
alpha-independent actions and physical transitions. The global inequality above
has a different quantifier: for each alpha it covers every policy satisfying the
same kernel/ratio assumptions, including policies whose selection depends on
alpha. The new economic-harm corollary is an accounting-specific application of
classical gain/loss-ratio and martingale reasoning, not a new financial ratio,
general probability result, or guarantee for arbitrary real-market policies.
