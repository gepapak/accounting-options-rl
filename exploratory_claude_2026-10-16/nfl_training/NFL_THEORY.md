# No-free-lunch (NFL) training: statement and proof sketch (EXPLORATORY)

## Setting
A trading simulator steps an account with exogenous inputs. At step t the policy observes o_t and chooses a position q_t.
The exogenous price change is written s_t = eps_t * m_t, with magnitude m_t >= 0 and sign eps_t in {-1, +1}.
- Booked P&L: b_t = q_t s_t - c_t, with own trading cost c_t >= 0 (fees, impact), observed by the trainer.
- Reported reward: r_t = b_t + l_t, where l_t is whatever the simulator credits beyond the booked P&L
  (a forgiven deficit under a zero floor, limited liability, a settlement lag or terminal limited liability; any accounting bug).
  The trainer does **not** observe l_t or the ledger.
- Real market P: arbitrary (it may contain a legitimate edge, e.g. a signal z_t correlated with eps_t).
- Placebo market Q: the same magnitudes, caps and observation process, but eps_t i.i.d. fair and independent of everything
  observed before it (including any signal, which is redrawn independently).

## Proposition 1 (no free lunch)
For every policy pi and discount 0 < gamma <= 1 on a finite horizon, E_Q^pi[ sum_t gamma^t (b_t + c_t) ] = 0.

*Proof.* b_t + c_t = q_t eps_t m_t. Under Q, q_t and m_t are fixed before eps_t is drawn, and eps_t is independent of them with mean 0.
So E_Q[q_t eps_t m_t | F_t] = q_t m_t E[eps_t] = 0. The tower property and linearity give the result. (With an unbounded horizon,
add an integrability condition and use optional stopping.)

**Corollary (leak meter).** E_Q^pi[sum gamma^t (r_t + c_t)] = E_Q^pi[sum gamma^t l_t] =: L_Q(pi) >= 0 whenever the leak is a credit
(l_t >= 0). The left side needs only the reward and the agent's own costs, so a policy's placebo leak is measurable without the ledger.

## NFL objective
    J_NFL(pi) = E_P^pi[sum gamma^t r_t] - E_Q^pi[sum gamma^t (r_t + c_t)]
              = J_booked(pi) + L_P(pi) - L_Q(pi),          J_booked(pi) = E_P^pi[sum gamma^t b_t].

Implementation: mix real environments (reward r_t) and placebo environments (reward -(r_t + c_t)) in one vectorized
policy-gradient batch. With equal numbers, the batch gradient is proportional to the gradient of J_NFL.

## Proposition 2 (exact ledger training without the ledger, in a fair market)
If the real market is itself fair (P = Q, as in the placebo studies B, F, I, J, L, N, O of the paper), then
J_NFL(pi) = J_booked(pi) for every policy. NFL training then has the same optimizer as training on the booked ledger,
although the trainer never sees the ledger or the accounting rule.

*Proof.* L_P = L_Q when P = Q.

## Proposition 3 (general market: when NFL is conservative)
Delta(pi) = L_P(pi) - L_Q(pi) is the residual leak. Suppose the leak is a nonnegative function of the pre-step state and
the realized loss, l_t = g(x_t, (q_t s_t - c_t)^-), increasing in the loss. If, at every state, (i) the policy's
discounted occupancy of states with g > 0 under P is at most its occupancy under Q, and (ii) the conditional loss
distribution under P is stochastically no larger than under Q (positions do not trade against the edge), then Delta(pi) <= 0
and J_NFL(pi) <= J_booked(pi). NFL never rewards such a policy for a leak. A policy that trades with a legitimate edge
satisfies (ii), and (i) holds when the edge moves equity away from insolvency.

**What NFL does not guarantee.** (a) If the observation reveals which market the agent is in (for example a feature
whose relation to past outcomes differs between P and Q), the agent can learn to exploit only in real episodes; the placebo
must be indistinguishable from the observation alone. (b) Leaks that are not credits (l_t < 0, e.g. an overcharged fee)
are penalized less than credited, but they never create an exploit incentive. (c) Under strong occupancy mismatch the
residual Delta can be positive for policies that seek insolvency more often in P than in Q; the pilot measures this.

## Proposition 4 (sign-paired reward: the ledger at every state, without the ledger)
Suppose the simulator can be cloned before a step and the clone stepped with the opposite sign of the same exogenous change
(the same magnitude, -eps_t). Let r_t^- be the clone's reward. Define the sign-paired reward

    r~_t = (r_t - r_t^-) / 2 - c_t.

If the leak at step t is a function of the pre-step state and the step's own outcome, l_t = g(x_t, b_t), then
r~_t = b_t + (g(x_t, b_t^{eps}) - g(x_t, b_t^{-eps})) / 2, so, conditional on the pre-step state and action,
E[r~_t | x_t, q_t] = E[b_t | x_t, q_t] + (P(lose) - 1/2) * (g_lose - g_win).
- In a fair market (P(lose) = 1/2) the conditional mean equals the booked P&L at **every** state, so there is no occupancy mismatch and
  any policy-gradient method trained on r~ optimizes the booked ledger.
- With a legitimate edge and positions that follow it (P(lose) < 1/2), and a credit-type leak (g_lose >= g_win), the bias is <= 0:
  conservative, and small away from insolvency.
Every rule of the ladder used here (ZF, LL, LAG at detection, TERM at the last step) is of this form.

**Scope.** SPR (and NFL) assume the legitimate P&L is linear in the sign-symmetric exogenous change (b + c = q * eps * m). A genuinely
convex legitimate payoff (holding options, volatility harvesting) has a sign-symmetric value that SPR would remove as if it were a leak.
Costs must be the agent's own, observed. The access needed is clone + sign override + own cost; the ledger and the rule are not needed.

Unit check (scratch, 400 episodes per cell, fixed policies, DK1): SPR minus booked per episode, % of K0:
placebo -0.22 / -0.63 (ZF), 0.03 / 0.11 (TERM), -0.05 / 0.05 (LAG) for follow-4 / gamble-low, all within noise; edge market -0.33 / -1.04
(ZF, conservative), about 0 for TERM and LAG. The reported-minus-booked leak these remove is 1-46% of K0 per episode.

## Proposition 5 (uniqueness: why (r - r^-)/2 - c and nothing else)
Fix a state and action. Let the position gain g >= 0 on the favourable sign and lose g on the other, with P(favourable) = p, own cost c,
and an arbitrary nonnegative credit l(+), l(-) on the two sides (the leak). Consider any estimator built linearly from the realized
reward r, the sign-reversed twin's reward r^- and the own cost: Y = a r + b r^- + k c, with constants a, b, k. Require
  (i) no leak (l = 0): E[Y] = E[booked] = (2p - 1) g - c for every edge p and every g, c;
  (ii) fair market (p = 1/2): E[Y] = E[booked] = -c for every leak l.
Then a = 1/2, b = -1/2, k = -1: Y is the sign-paired reward, uniquely.
*Proof.* E[Y] = (a - b)(2p - 1) g - (a + b) c + k c + a E l(eps) + b E l(-eps). (i) for all p, g, c gives a - b = 1 and k - (a + b) = -1.
In (ii), E l(eps) = E l(-eps) = (l(+) + l(-))/2, so (a + b)(l(+) + l(-))/2 = 0 for all l, hence a + b = 0.
Consequences: the plain reward (a = 1, b = 0) violates (ii); the cost-only reward Y = -c satisfies (ii) but violates (i) (it is blind
to any legitimate edge); the booked reward needs the ledger. With both an edge and a leak, E[Y] - E[booked] = (1/2 - p)(l(-) - l(+)),
which is <= 0 for an edge-following position (p >= 1/2) and a credit on the losing side: conservative (Proposition 4).

## Proposition 6 (no variance cost)
If the credit falls only on the losing side and is at most the loss plus cost (a forgiven deficit), then conditionally on the state and
action, Y + c = eps (g - l(-)/2), so |Y + c| <= g = |booked + c|: the sign-paired reward never has a larger conditional variance than the
booked reward, and equals it exactly at every step without forgiveness (which is why SPR and ledger training often produce identical
policies in the experiments).

## Positioning (prior art checked 17 Oct 2026)
- Counterfactual reward subtraction is established for credit assignment: difference rewards (Wolpert and Tumer) subtract the reward with
  the agent's action replaced by a default; COMA (Foerster et al. 2018) marginalizes the agent's action under a centralized critic.
  SPR subtracts a counterfactual of the *exogenous* input (the market's sign), for a different purpose (removing accounting credits).
- Antithetic coupling and common random numbers are classical variance-reduction tools; exogenous-MDP and counterfactual-data-augmentation
  work (e.g. Lu et al. 2020, CoDA) reuse inferred exogenous noise. None of these removes reward credits or gives an accounting guarantee.
- So the mechanism's ingredients are known; the contribution is the identification result (Propositions 4-5: the booked objective is
  recoverable from the reward channel with one sign-reversed query per step, uniquely by SPR), its limits (linear legitimate P&L; known
  own cost; the market is the only sign-symmetric input), and the evidence against the established mitigation (KL / occupancy regularization).

**Main objection to anticipate.** In the simple sleeve the trainer could compute the booked P&L itself (position x price change - cost),
which is exactly the ORACLE arm. SPR matters when the booked P&L is produced inside a black-box simulator (multi-step settlement, cash
transfers, multiple instruments, proprietary accounting) that can be cloned and fed a sign-reversed input but whose ledger is not exposed.
The simple sleeve tests the mechanism under controlled conditions; a demonstration inside a complex simulator (e.g. the paper's engine)
is what would answer this objection.

## Relation to the paper
The placebo certificate of the paper (reported return of a frozen agent in the sign-randomized market) is L_Q(pi) minus costs
for one policy. NFL turns that certificate into a training penalty, so the same market that exposes the exploit removes it.
Existing reward-hacking mitigations regularize toward a safe reference policy (KL or occupancy-measure divergence; Laidlaw
et al.); they penalize a deviation by its size, not by whether it is paid for by the accounting, so a high-value exploit and a
low-value legitimate improvement cannot be separated by one coefficient.
