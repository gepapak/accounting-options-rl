# Exploratory accounting witnesses in ledger/clock states

This protocol is written on 15 October 2026, after the registered Study N outcomes
were available. It is **retrospective and exploratory**, not a new confirmation.
It does not change or rerun any registered study.

The question is narrower than acquisition prediction: can a cheap policy supply
a positive expected reward advantage over doing nothing under the same account,
while its expected booked payoff is negative? A positive witness establishes an
incentive for that policy in the empirical independent fair-sign model. Failure
to find a witness does not establish safety or optimality.

The fixed policies take leverage 16 once and are flat otherwise. In the zombie
window the action is immediate. In terminal liability both an immediate action
and an action at the final period are retained; neither is selected on learning
outcomes. Every registered insolvent zombie probe and terminal probe is reported,
with the matching booked-account controls. The terminal delayed policy depends
on the clock and illustrates a case that an equity-only reduction misses.

The expectation is integrated over both signs of every TRAIN magnitude/cap row,
after the same first-200-row exclusion used by the reduced-model benchmark. Fees
are 1 EUR/MWh, the registered N value. There is no DP, learned policy, critic,
Monte Carlo, fitted threshold or optimal-gap claim. Numerical integration is
exact conditional on the finite historical empirical sample; it supplies no
population confidence interval or learnability guarantee. The registered DP and
agent probes are read only for descriptive comparison after witness computation.

For ledger n and one-step payoff X, write h(n)=E[(-n-X)+] and d(n)=(-n)+.
The immediate zombie-claim reward advantage is E[X]+h(n)-d(n). Under terminal
liability with tau periods left and discount gamma, the immediate one-action
advantage is E[X]+gamma^(tau-1)*(h(n)-d(n)); the delayed one-action advantage is
gamma^(tau-1)*(E[X]+h(n)-d(n)). Booked counterparts are E[X] and its discounted
version. These formulas are checked pathwise against the unchanged registered
environment, including tau=1. The terminal exponent follows the environment's
actual terminal reward timing, rather than assuming its DP's boundary convention.

Outputs contain aggregate expectations and saved policy-probe summaries only.
No eSett raw data are redistributed. All source hashes and exact assumptions
are recorded. Existing numerical results and previous exploratory failures remain
untouched.

Run from the code repository:

```powershell
python exploratory_novelty_2026-10-15/state_audit/exploratory_state_witness.py
```
