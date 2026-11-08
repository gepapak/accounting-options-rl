# Study Z, Part B theory: per-position (isolated) margin (written 3 November 2026, before any Part B agent was trained)

Study Z gives the agent and the trainer both accounts' balances, so a ruin penalty can be applied per account, and it worked there (pilot).
The objection, raised in an AI-assisted review of Study Z's design, is that the per-account remedy is then always available. Part B removes
that option: one account, one balance, and a margin mode in the action. On derivatives exchanges a position opened in isolated margin can lose at
most its margin; a liquidated position's deficit is covered by a pooled fund or vault. A trainer of a black-box simulator sees the balance, the
action index and the reward, not the margin rule.

## Setting
One account with reported equity P >= 0. A position q with own cost c in cross margin is written with the zero floor, P' = max(P + x, 0). In
isolated margin it can lose at most m = c + M |q| (and never more than P): P' = P + max(x, -min(m, P)). Booked x = q s - c in both modes;
s = eps |s| with eps the exogenous sign.

## Proposition (isolated margin)
1. **A single censor.** The reward is r = max(x, tau) with tau = -min(m, P) in isolated margin and tau = -P in cross margin (Assumption C).
   The favourable branch g - c is never below tau when P >= c (because min(m, P) >= c), so the twin pair identifies the booked P&L on every
   step of an account that can pay its own cost, and TIR is exact there.
2. **The ruin penalty is not incurred.** If P > m, then P' >= P - m > 0: an isolated position cannot take the account to zero, so a penalty
   on the account's ruin is never incurred on such a step, whatever its strength.
3. **Forgiveness at every funded equity.** If P > m and the position's gross is favourable with probability p (independent of |s|), the step's
   expected reported reward exceeds its expected booked reward by (1 - p) |q| E[(|s| - M)^+]; it grows linearly in |q| while the own cost grows
   quadratically. On the DK1 training panel (E|s| = 76.7 EUR/MWh, median 43) with M = 100 EUR/MWh, which covers 84% of quarter-hour moves,
   maximal leverage with the signal (p = 0.6) reports 22.6 EUR per quarter-hour more than leverage 4 and books 14.3 EUR less.
4. **The twin rewards.** The accounting is forgiving (credits >= 0 on both branches), so the twin bracket (Proposition 10) holds; under
   conditional sign fairness SPR's expectation is the booked one (general credit law); by part 1 TIR is exact wherever P >= c.
5. **What the structure-aware remedy needs.** A penalty on liquidations (steps on which an isolated position loses more than m) sees every
   forgiven isolated step, but only a trainer who knows the venue's margin rule can compute it.

## A ledger-free exploit finder
The bracket's width `w = r + r' + 2C` is the credit paid on the twin pair for any forgiving accounting (Proposition 10, part 2); it needs the
reward channel, the sign-reversed clone and the own cost, like SPR. An agent trained to maximize the sum of `w` searches the simulator for
forgiveness without the ledger and without knowing the accounting. Under conditional sign fairness the two branches are exchangeable, so the
expected width is twice the expected credit on the realized branch. What the finder reaches is a witness: a lower bound on the credit available
under some policy, not a proof that no other forgiveness exists. Under isolated margin the width of a position q at an account with P > m is
|q| (|s| - M)^+ on every step (one branch always loses), largest at maximal leverage.

## Proof
1. Isolated: P' - P = max(x, -min(m, P)); cross: max(P + x, 0) - P = max(x, -P). The favourable branch is x_f = g - c >= -c >= -min(m, P)
   when min(m, P) >= c, which holds when P >= c because m >= c. Then the pair is unequal whenever g > 0 and Proposition 4 (twin
   identification) gives the booked P&L exactly.
2. P' = P + max(x, -min(m, P)) >= P - m > 0.
3. E[max(x, -m)] - E[x] = E[(-m - x)^+] and -m - x = M|q| - |q| |s| on the adverse branch (probability 1 - p) and negative on the favourable
   branch, so the difference is (1 - p) |q| E[(|s| - M)^+]. The numbers are from `calibration` in RUN_LOG_ZB.md.
4. As in the proof of the split-account proposition, part 4.
5. By definition of a liquidation.

## Exact checks (`test_envs_zb.py`, run before any training)
322,396 checks on random states, actions, modes and signs, no failure: the accounting identity and non-negative credits (80,000); the reward
equals the single censor max(x, tau) of part 1 (80,000); TIR equals the booked P&L on every step whose pair differs (28,822); an isolated
position never takes an account with P > m to zero (21,068); the liquidation flag (32,506); the bracket (40,000); fair-sign unbiasedness of
SPR (40,000).

## Scripted witness (`witness_zb.py`, `witness_zb.json`, no learning)
On the registered test weeks of the edge market, a policy that trades maximal leverage with the signal in isolated margin while equity is at
least 0.25 allocations (leverage 4, cross margin, below) books -9.5% of the allocation per week against +22.0% for the honest optimum,
reports +38.7% against +22.0%, incurs no account ruin and 21.0 liquidations per week; under the ruin penalty at 5 and at 25 its reward is
unchanged (+38.7, above the honest +22.0), under the liquidation penalty at 5 it is -66.3. Always trading isolated maximal leverage drains
low-equity accounts through repeated liquidations (7.6 ruins per week) and is deterred by the ruin penalty (-0.5 at strength 5): the exploit
that survives the penalty is the funded one. In the placebo market the funded policy books -64.5% and reports -29.8% (honest -10.8%, -7.9%),
so its incentive comes from the edge market in which the agents are trained.
