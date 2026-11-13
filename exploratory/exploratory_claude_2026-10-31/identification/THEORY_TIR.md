> **Erratum (2 November 2026).** The statement below that SPR is "the sharp lower end of the identified set for g" is false as written: with g = 10, C = 1,
> tau = 0 the pair (9, 0) identifies g = 10 while SPR's magnitude |r - r'|/2 is 4.5. SPR's magnitude is a valid, generally non-sharp lower bound; TIR's
> magnitude is the sharp lower end of the identified set (singleton for r != r', interval [0, r + C] for r = r'). Corrected statements and proofs: manuscript
> Proposition 4 and Supplementary Section S20; independent exact checks in `exploratory_codex_2026-11-01/review_theory/` and
> `exploratory_claude_2026-11-02/check_identification_statements.py`. The registered rewards and results are unchanged.

# Twin identification of booked reward from a censored reward channel (exploratory theory note, 31 October 2026)

Status: development note. All statements below are checked in exact rational arithmetic by `verify_identification.py` (200,000 random cases,
the 4/9 linear program) and numerically on random-action rollouts of every accounting rule by `ladder_identification_study.py`. The
manuscript carries the final statements; this note keeps the derivations and the scope limits. It supersedes the sign-paired-reward
claims of `../exploratory_claude_2026-10-16/nfl_training/NFL_THEORY.md` where they differ (see its errata).

## 1. Model

At one step the trainer knows the action (hence the position q and the own cost c = c(q) >= 0) and the realized exogenous sign. It cannot
read the ledger or the accounting rule. Write the gross P&L as G = sigma * g with g = |q| m >= 0 hidden (m the exogenous magnitude) and
sigma = sign(q) * epsilon in {-1, +1} the realized gross sign. The booked P&L is b = sigma g - c. The simulator can be cloned and the clone
stepped with the exogenous sign reversed (a *twin*), which has booked P&L b' = -sigma g - c.

**Assumption C (monotone censoring).** The simulator reports the step reward r = max(b, tau), where tau in [-infinity, infinity) is a
threshold determined by the pre-step state and not by the action or the exogenous draw. The twin reports r' = max(b', tau).

Every rule of the ladder satisfies C at every step (the credit is max(tau - b, 0) >= 0):
- zero floor, reported equity e: r = max(b, -e), tau = -e <= 0;
- limited liability at its closing step: the same;
- terminal limited liability at the horizon, ledger L: r = b + max(0, -(L + b)) = max(b, -L), tau = -L, a carried deficit D = -L > 0 when insolvent;
- settlement lag at detection: the same with the ledger at detection;
- every step without a credit: tau = -infinity, r = b.
(Study N's zombie window reports the *claim increment* max(L + b, 0) - max(L, 0) while insolvent, which is a censored observation of b - D,
not of b; Section 6 treats that convention.)

## 2. Identification

Since g >= 0 the favourable branch is b_+ = g - c and the other is b_- = -g - c, with rewards r_+ = max(g - c, tau) >= r_- = max(-g - c, tau).
The pair {r, r'} = {r_+, r_-} does not depend on sigma.

**Proposition 1.** (a) r != r' if and only if g > 0 and tau < g - c (the favourable branch is uncensored). Then g = max(r, r') + c and
b = sigma (max(r, r') + c) - c exactly. (b) If r = r' then either g = 0 or tau >= g - c (both branches censored, r = r' = tau), and g lies in
[0, r + c]; both endpoints are attained by compatible (g, tau), so the interval is sharp. (c) The conservative estimate (|r - r'|/2) of g
never exceeds g, with equality if and only if the unfavourable branch is uncensored (tau <= -g - c).

*Proof.* If r_+ > r_- then r_+ > tau (because r_- >= tau), so r_+ = g - c, giving (a). If r_+ = r_- with g > 0 then max(g - c, tau) =
max(-g - c, tau) forces both to equal tau, so tau >= g - c; then g - c <= tau = r gives g <= r + c. For (c): |r - r'|/2 = (r_+ - r_-)/2 =
(g - c - r_-)/2 where r_- >= -g - c, so the estimate is at most g, with equality iff r_- = -g - c, i.e. the unfavourable branch is
uncensored. For sharpness take (g, tau) = (r + c, r) and (0, r). QED

## 3. Estimators

- **SPR** (sign-paired reward): b_SPR = sigma |r - r'|/2 - c, the sharp lower end of the identified set for g.
- **TMR**: b_TMR = sigma (max(r, r') + c) - c, the upper end. Exact when identified; when both branches are censored it equals
  sigma (tau + c) - c and its error can be as large as the carried deficit (catastrophic for terminal and lag steps).
- **TIR** (twin-identified reward): b_TIR = b_TMR if r != r', and -c if r = r'.

**Proposition 2.** Under Assumption C: (i) b_TIR = b whenever r != r' (and whenever g = 0). (ii) If r = r', |b_TIR - b| = g <= r + c and
b_TIR = -c equals the booked expectation under sign fairness. (iii) For every realization |b_TIR - b| <= |b_SPR - b|. (iv) If the sign is
conditionally fair given the state, action and magnitude, E[b_TIR | state, action, magnitude] = E[b | .] = -c, for any threshold; if the sign
is favourable with probability p >= 1/2 at a position with hidden g, E[b_TIR] - E[b] = -(2p - 1) g * 1{both branches censored} <= 0:
the bias is zero on every identified step and one-sided, bounded by (2p - 1)(r + c), on censored steps. (v) Where r = r' the reward is
-c <= 0 for every action, so no gain is ever manufactured.

*Proof.* (i), (ii) from Proposition 1. (iii) When r != r', TIR is exact; when r = r', both estimators equal -c. (iv) g-hat is a function of
the unordered pair, hence of (g, c, tau), not of sigma; E[sigma] = 2p - 1 gives E[b_TIR] = (2p - 1) g-hat - c. (v) immediate. QED

**Proposition 2' (uniqueness of the fallback).** Among estimators f(r, r', c, sigma) that equal b on every identified step (r != r') and
whose conditional expectation under sign fairness equals -c on every non-identified step (r = r'), TIR is the unique one with zero
conditional variance on the non-identified steps.
*Proof.* On r = r' the estimator takes one value f_+ when sigma = +1 and one value f_- when sigma = -1. Fairness requires
(f_+ + f_-)/2 = -c; the conditional variance is ((f_+ - f_-)/2)^2, which vanishes iff f_+ = f_- = -c, the TIR value. TMR (f_+ - f_- =
2(r + c)) and the midpoint rule (f_+ - f_- = r + c) satisfy the same unbiasedness and add the noise ((r + c)/2)^2 or more on every censored
step, which is the deficit scale on carried-deficit steps. QED
(The midpoint of the identified interval is the minimax-error choice for g; TIR deliberately trades that worst-case accuracy for zero added
reward variance and for never manufacturing a gain.)

Corollary (training). The twin query does not change the transition. In a fair market the expected return of every policy under TIR equals
its booked return, for the SPR and TIR estimators alike; with an edge, TIR's return differs from the booked return only through
censored steps, and by the one-sided amount in (iv). The linear-uniqueness result of the earlier note (SPR is the only constant-coefficient
linear estimator unbiased for every credit law under fairness) is not contradicted: TIR is not linear, and uses Assumption C.

## 4. Limits of what any twin-based reward can do

**Proposition 3 (identification failure and regret).** Fix a terminal step with a carried deficit D = 10 (tau = 10) and positions q in [-1, 1]
with own cost |q|. Let the hidden gross magnitude be 1 in world W1 and 10 in W2, and let a signal z in {-1, +1} predict the sign with
E[epsilon | z] = 0.2 z. Every query of every action and sign returns r = max(b, 10) = 10 in both worlds, so no function of the queries
distinguishes W1 from W2. The best legitimate policy is flat in W1 (value 0) and q = z in W2 (value 1). Any policy has worst-case regret
at least 4/9 per decision; q = (5/9) z attains it.

*Proof.* With x = E[q z] and y = E|q| >= |x|, the regrets are y - 0.2 x in W1 and 1 - 2 x + y in W2. Their maximum is minimised at
x = y = 5/9, value 4/9 (linear program in `verify_identification.py`). QED

So where the carried deficit exceeds the best attainable booked P&L, the reward channel is uninformative about g under any estimator, and
the sharp-interval fallback is all that can be done. TIR's fallback never invents a gain (Proposition 2(v)); it is not minimax regret.
A further query that leaves the censored region (for example a probe at a larger position) would identify g again; that is an access
extension, not a property of the sign twin.

## 5. What this adds to the sign-paired reward

SPR is correct in expectation under fairness for any credit law (it is linear and generic) but attenuates a legitimate edge when credits and
an edge coexist, and its pathwise error is the half-credit of the unfavourable branch. TIR uses the censoring structure shared by all
ladder rules to be exact wherever the twin pair identifies g, and falls back to SPR's conservative value elsewhere. On random-action
rollouts (`ladder_identification_results.csv.gz`), the mean absolute error on credited steps, in percent of the allocation per quarter:

| rule | reported | SPR | TMR | TIR |
|---|---|---|---|---|
| zero floor | 0.96 | 0.48 | 0.017 | 0.008 |
| limited liability | 6.7 | 3.3 | 0.002 | 0.000 |
| terminal / lag with carried deficits (insolvent starts) | 17-73 | 0.2-0.6 | 15-64 | 0.2-0.6 |

## 6. Scope

- Needs a clonable simulator, a sign override of the exogenous input, known own costs and the realized gross sign, and Assumption C.
- Study N's zombie window reports claim increments while insolvent (r = max(b - D, 0) = max(b, D) - D): the pair identifies g - D, not g.
  TIR is then unbiased under fairness (its g-hat does not depend on sigma) but not pathwise exact; the check on Study N's environment gives
  errors close to the gross size, with signed errors near zero.
- Fairness is a conditional statement given the state, action and non-sign inputs; a signal that predicts the sign is an edge, handled by
  Proposition 2(iv).
- A convex legitimate payoff (an option held by the agent) has sign-symmetric value that every estimator here would remove.
- The "just compute the booked P&L" objection stands for simulators that expose their ledger.
