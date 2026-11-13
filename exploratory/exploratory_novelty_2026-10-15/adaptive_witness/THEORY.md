# Exploratory adaptive accounting-witness synthesis

Written on 15 October 2026 after the manuscript's completed experiments. This is
an exploratory method proposal and exact synthetic capability check, not a
registered confirmation or a new probability theorem. No Study R outcome is used.

## The task and the actual capability extension

Find an executable, nonanticipative policy with positive expected charged reward
and negative expected booked reward in a specified no-edge simulator. Return
**WITNESS** only with a valid independent lower certificate; otherwise return
**ABSTAIN**. An abstention does not establish safety. Neither result predicts
whether PPO, another learner, or a fixed training budget will acquire the policy.

A one-action positive gate cannot deliberately pay a negative expected prefix to
reach a state with a profitable continuation. The proposed finite policy grammar
adds such prefixes and ledger/clock feedback. It uses whole-trajectory simulation
and a small policy library; it does not compute Bellman values, a state-grid DP,
an optimality gap, or an exhaustive best response. Classical rollout improvement,
simulation-based policy search, control variates, upper truncation and bounded
concentration supply the tools. The accounting-specific elements are the event
grammar, the exact charged/booked identity, the cost-only lower bound, and the
scope of the resulting certificate.

## Assumptions and accounting identity

Let the horizon H be finite and 0 < gamma <= 1. At time t the policy chooses a
position using only information available before the next gross payoff G_t.
Assume E[G_t | history, chosen position] = 0, E|G_t| < infinity, and executed
cost 0 <= C_t <= c_t,max almost surely. Costs may depend on the next magnitude
or capacity row; the conditional gross-mean condition must still hold. Actions,
ledger transitions and costs must be identical in the paired charged/booked
evaluation. Merely labeling a historical strategy "no edge" is insufficient.

For a step-floor ledger E_t >= 0, define

    I_t = (-E_t - G_t + C_t)+,
    E_{t+1} = E_t + G_t - C_t + I_t,
    r_alpha,t = G_t - C_t + (1-alpha) I_t.

For 0 <= alpha <= 1 put

    F_gamma = sum_t gamma^t I_t,
    C_gamma = sum_t gamma^t C_t,
    M_gamma = sum_t gamma^t G_t,
    Y_alpha = (1-alpha) F_gamma - C_gamma.

Then pathwise R_alpha = M_gamma + Y_alpha. Predictability and the finite-horizon
tower property give E M_gamma = 0, hence

    J_alpha = E Y_alpha,
    J_book = -E C_gamma.

Thus positive expected Y_alpha plus positive expected executed cost is a harmful
reported-reward witness. An obligatory first trade with E C_0 > 0 implies
J_book <= -E C_0 < 0 without a noisy second empirical test. If the obligatory
trade is instead at deterministic time t0, the bound is -gamma^t0 E C_t0.
If policies can remain flat, strict harm needs a separate positive cost bound.

For terminal liability use the booked ledger N_{t+1}=N_t+G_t-C_t and D_t=(-N_t)+.
The reward **advantage over always flat under the same account** has

    F_gamma = gamma^(H-1) (D_H-D_0).

The same identity holds. For initially solvent states D_0=0, F_gamma >= 0.
For an initially insolvent state it is bounded below by -gamma^(H-1)D_0.
Subtracting D_0 is essential: a pre-existing terminal payout is also received by
the flat comparator. All examples below start solvent.

These identities concern expected discounted objectives. A sampled booked return
can be positive. They provide no no-edge guarantee for markets with exploitable
conditional drift, nor for policies that see the next fair sign before acting.

## Event-gated continuation lemma

Let rho be a fixed nonanticipative prefix, tau <= H its stopping time, and pi a
fixed continuation. Let V_pi(s,t) and V_flat(s,t) denote their conditional future
values at the reached state and remaining clock, using the same account. For any
event A measurable at tau, the policy that continues with pi on A and is flat
otherwise satisfies

    J(rho followed by [pi on A, flat otherwise])
      = J(rho followed by flat)
        + E[gamma^tau 1_A (V_pi(S_tau,tau)-V_flat(S_tau,tau))].

Proof: prefixes agree pathwise; condition the remaining return on information at
tau and apply the tower property. For tau=H the continuation difference is zero.
If A is the true positive-continuation-advantage event, the last term is
nonnegative. It can outweigh a strictly negative prefix value. For terminal
liability the **difference from flat**, rather than V_pi alone, prevents double
counting the payout on debt created by the prefix.

The lemma is a sufficient construction, not a guarantee that an estimated gate
is beneficial. Search estimates and gates may overfit. The complete frozen
candidate is therefore confirmed on independent trajectories. The grammar can
include compulsory one- or two-trade prefixes, low-equity or insolvency gates,
and remaining-clock switches, followed by flat or a fixed continuation policy.
A gate need not coincide with an actual floor event: a positive but sufficiently
low ledger can carry the next option incentive.

## A terminal-account clock normal form

Fix a local T-step controller whose actions use only the active episode's local
ledger/history and local clock. Embed it in an outer horizon H>=T, with flat
actions before and after the episode. Assume the active-step payoff/capacity/cost
kernel is stationary IID, waiting supplies no information used by the controller,
flat waiting has no costs or ledger changes, and there is no early termination.
The terminal default claim is paid only at the fixed outer deadline. These
restrictions concern the simulator, not calendar stationarity of the actual market.

Couple two placements using the same active row/sign sequence. Write M_T and C_T
for the discounted active gross payoff and cost, and D_T for final active debt.
The active ledger/actions and D_T are identical for every deterministic start s.
For advantage over the same-account flat baseline,

    R_H,alpha(s) = gamma^s (M_T-C_T)
                   + (1-alpha) gamma^(H-1)(D_T-D_0),
    Y_H,alpha(s) = -gamma^s C_T
                   + (1-alpha) gamma^(H-1)(D_T-D_0).

For s2>=s1, residual Y(s2)-Y(s1)=(gamma^s1-gamma^s2)C_T>=0 **pathwise**.
Gross conditional mean zero gives E[R(s2)-R(s1)] equal to the same nonnegative
expected cost difference. Raw charged reward need not dominate pathwise. If
expected executed cost is positive and gamma<1, a strict later start improves
expected charged reward while its expected booked reward remains negative.

Consequently the latest start s=H-T weakly dominates earlier placements for each
fixed local controller and T. A search can prune calendar placement without
transition simulation or an estimated value function. One active trajectory set
can compile all placements algebraically. This does not compare different T,
establish an optimal controller, or dominate arbitrary policies that adapt while
waiting or use global-calendar information.

There is a stronger reuse identity at the latest start:

    R_H,alpha(H-T) = gamma^(H-T) R_T,alpha,
    Y_H,alpha(H-T) = gamma^(H-T) Y_T,alpha,
    R_book,H(H-T) = gamma^(H-T) R_book,T,

pathwise. A lower certificate for a fixed local controller scales to every
H>=T on the same confidence event. Its fixed-policy alpha interval is unchanged.
No additional horizon multiplicity penalty is needed for deterministic algebraic
scaling of that already-certified random variable. This preserves positive sign,
not a fixed economic effect size: values decay as gamma^(H-T). If episode lengths
or controllers are selected on confirmation outcomes, selection still needs a
valid error allocation.

The binary synthetic solvent-prefix controller has T=2, J_0=.02375 and booked
-.1495. Embedded early in H=48 it loses in expectation; moved to the latest two
steps it retains positive value gamma^46*.02375. Every one-action policy remains
nonpositive. This demonstrates a clock placement miss in addition to the
multi-step miss. The exact outputs retain H=4,16,48 and use zero additional
active transition evaluations to compare placements.

The assumptions cannot be dropped. With a predictable deterministic gross payoff
+1 and cost .1, no default occurs and an early trade returns .9 versus .891 one
period later at gamma=.99. Positive edge reverses the dominance. Calendar-varying
fees/capacity, information acquired during waiting, idle costs, early account
closure and a policy using a different global clock can also break the coupling.
For a step-floor account, shifting an episode instead simply scales all rewards
by gamma^s; a positive episode is best placed early. The terminal normal form
must therefore not be transferred between accounting regimes.

## Cost-only bounded lower certificate, including unbounded gross payoffs

Let B_C=sum_t gamma^t c_t,max. For a step-floor or initially solvent terminal
account, Y_alpha >= -B_C pathwise. No upper bound on gross payoffs, price tails,
equity, or forgiveness is required. Integrability and the finite-horizon
assumptions above ensure the expectation exists.

Freeze the policy and an upper clip b > -B_C using only discovery data. On n
independent confirmation trajectories define Z_i=min(Y_alpha,i,b). Then

    -B_C <= Z_i <= b,       E Z_i <= J_alpha.

For a lower confidence procedure valid for bounded iid observations (for example
Hoeffding, or an appropriately stated empirical-Bernstein bound), L_Z <= E Z
implies L_Z <= J_alpha. Report WITNESS if L_Z>0 and strict expected cost has been
established; otherwise ABSTAIN. For Hoeffding at one-sided error delta,

    L_Z = mean(Z) - (b+B_C) sqrt(log(1/delta)/(2n)).

For an initially insolvent terminal state replace the lower bound by
-B_C-(1-alpha)gamma^(H-1)D_0. Discovery and confirmation must be independent;
choosing b, a policy, or a sample size by inspecting the same confirmation data
invalidates this fixed-sample argument unless a valid simultaneous/sequential
method is used. With M separately reported policy/condition certificates, assign
errors summing to at most the desired family error. Shared confirmation paths
between fixed policies do not invalidate a union bound; iid whole paths remain
the sampling units. A positive L_Z is sufficient, not necessary, for a witness.

This is a one-sided truncation bound, not an unbiased approximation to the
unclipped mean. Reducing b may increase clipping bias and destroy power. Upper
clipping of raw reward is a fair baseline when it has a known lower bound.
In particular, alpha=0 step-floor reward satisfies R_0 >= -E_0 by summation by
parts of the nonnegative equity process; it too can be upper-clipped without a
payoff bound. Cost-only certification is most distinctive when the raw reward
lacks a usable lower bound, or when B_C is much smaller than E_0. For alpha>0
step-floor reward and discounted terminal reward, raw reward need not have a
finite lower bound under unbounded gross tails. A finite empirical market sample
does have finite support; the empirical comparison must report whether the new
representation improves certification there rather than assume that it does.

## One fixed policy, an interval of liability charges

Assume the initial state is solvent and a policy is fixed and independent of
alpha. Its transition dynamics and executed actions must remain unchanged as
alpha varies. Freeze a floor clip b_F >= 0 and obtain simultaneous bounds

    E F_gamma >= E min(F_gamma,b_F) >= L_F >= 0,
    E C_gamma <= U_C.

The known cost bound permits an ordinary bounded upper confidence bound on C.
For every alpha in [0,1] on the same confidence event,

    J_alpha >= (1-alpha)L_F-U_C.

If L_F>0 and U_C<L_F, this certifies the **strict** harmful interval
alpha < 1-U_C/L_F, together with a strict expected cost guarantee. No union over
alpha is needed because the affine inequality holds on one simultaneous event.
This is an interval for one executable policy, not a global minimum repair
charge, an optimal-policy boundary, or evidence of safety above its endpoint.
If actions or dynamics change with alpha, this shared-policy argument fails.

At alpha=1, the no-edge identity gives J_1=-E C_gamma<=0 for every admissible
policy in the model. This analytically prunes the whole policy class. It does
not establish a universal guarantee in environments with positive conditional
gross drift or a different definition of booked costs.

## Exact synthetic tests and retained failures

The companion script enumerates all sign paths of each **fixed** policy in a
small binary-action library. There is no state-grid recursion or optimization
over all policies. Gross payoff per executed trade is independent fair +/-1,
gamma=.99, and gates are defined at alpha=0 then frozen. All one-action policies
at the initial equity have strictly negative value in the following cases;
therefore a true myopic policy stays flat.

1. **Floor prefix, e0=.7, c=.4.** One action is -.05. A forced first trade followed
   by another trade only after reaching the floor has, at H=2,
   F_gamma=.6965, C_gamma=.598 and J_0=.0985. Constant trading gives -.07475.
   The fixed policy has J_alpha=(197-1393 alpha)/2000, so its exact strict
   harmful interval is alpha<197/1393. At H=3 its J_0=.1720075.
2. **Solvent prefix, e0=1.5, c=.1.** One action is -.1 and its first loss leaves
   equity .4, still solvent. A floor-only continuation gives -.1, constant two
   trades gives -.02575, but forced-prefix plus positive-continuation gating
   gives .02375 and booked -.1495. The same H=2 policy and values hold under
   terminal liability. At H=3 the step-floor adaptive value is .13401125.

The library abstains in the two floor-prefix cases at alpha=.2, at full booking,
and at e0=5,H=3,c=.4. The first two abstentions at alpha=.2 are library failures,
not proofs of safety. Full booking has the analytic no-edge guarantee. The high
capital example also has a class-wide binary-action guarantee because
e0>=H(1+c), so no possible path can create a floor injection.

The floor-prefix H=2 example is a **negative variance check**: raw charged-return
variance is .29560675, while residual variance is .48862675. Subtracting the
known zero-mean gross martingale increases variance here. Any variance or power
improvement must be measured rather than inferred from the identity.

## Unbounded-support capability example

Let P(A=k)=2^-k for k>=1 and G be an independent fair sign times A. Gross payoff
has zero mean, finite absolute mean 2, and unbounded support. Start solvent at
e0=1, charge c=.1, trade once immediately, then flat until terminal liability at
H=2, gamma=.99. Its expected booked reward is -.1 and J_0=.4445. On a negative
sign raw charged reward is -.01k-.991, unbounded below; on a positive sign it is
k-.1, unbounded above.

The accounting residual is nevertheless at least -.1. Upper clipping it at 1
gives exact expected clipped reward 1587/8000=.198375>0, with range [-.1,1].
Consequently a standard bounded lower certificate is available without a known
price/payoff maximum. This is an exact distributional example, not sampled
confirmation. The script reports a conservative theoretical Hoeffding sample
size and explicitly records that no confirmation observations were drawn.

Clipping F_gamma at 1 gives E min(F_gamma,1)=1099/4000=.27475; with exactly known
cost .1 this yields the population lower interval alpha<699/1099, versus the
true fixed-policy interval alpha<889/1089. A finite-sample certificate will
usually be narrower. The deliberate conservatism is part of the method.

## Algorithm sketch and defensible novelty claim

1. Specify the simulator, no-edge assumptions, cost bound, initial-state/clock
   distribution, budget, policy grammar, discovery/confirmation split and error
   allocation. Include one-action, true or forced-prefix myopic, fixed barrier,
   and unstructured parameter-search baselines.
2. Search only the discovery split. Evaluate the full prefix/continuation policy
   using paired ledgers, optionally score fixed upper clips of the residual, and
   select a candidate plus clips deterministically. Cost of all candidate
   trajectories counts toward the search budget. Preserve all failed cases.
3. Freeze the candidate and clips. Confirm on independent whole trajectories.
   Return a positive lower certificate plus the booked-cost guarantee, or
   abstain. Optionally provide the simultaneous fixed-policy alpha interval.
4. Report simulator calls, elapsed compute, candidate count, witness coverage,
   raw and residual certificate baselines, clipping bias, and failures. Separate
   exact synthetic capability from exploratory historical-model confirmation.

Strongest safe claim: an accounting-specific, simulator-based witness procedure
extends a one-action audit to executable adaptive multi-step policies and can
produce rigorous one-sided harmful-policy certificates under integrable
unbounded gross payoffs and bounded executed costs. For a stationary no-edge
terminal model, a proved clock compiler eliminates deterministic placement from
search for each fixed local controller and reuses its certificate across outer
horizons. It is not a new general
policy-search algorithm, a learner acquisition predictor, or a global safety
certifier. The exact counterexamples establish a capability gap; empirical
comparisons must establish any practical benefit over simple barriers and raw
reward clipping before a performance claim is made.
