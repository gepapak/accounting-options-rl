# Study Z theory: split accounts (written 3 November 2026, before the pilot results were read)

Simulators and settlement systems often keep several accounts with their own floors: sub-accounts per asset or market, isolated margin,
separate legal entities, or several balance responsible parties in one group, each with its own collateral. A remedy that is measured on the
portfolio, such as Study V's ruin penalty applied to the portfolio's reported equity, then no longer sees where the losses are forgiven.

## Setting
A portfolio holds accounts k = 1..K with reported equities E_k >= 0, each with its own zero floor, E_k' = max(E_k + x_k, 0), trading
continues; x_k = q_k s - c_k is the booked P&L of account k's leg on the move s = eps |s| (eps the exogenous sign), c_k >= 0 its own cost.
The portfolio's reported equity is P = sum_k E_k and its reward r = P' - P; the booked portfolio P&L is x = sum_k x_k. A pooled account writes
the same legs to one equity: E' = max(E + x, 0).

## Proposition (split accounts)
1. **More forgiveness.** In every state, for every action and move, the split accounts forgive at least what one pooled account with equity P
   forgives: sum_k (-E_k - x_k)^+ >= (-P - x)^+.
2. **A hedge reports a gain whatever the sign.** From two empty accounts, positions q and -q on the same move, each with own cost c, report
   (q|s| - c)^+ for either sign of the move and book -2c. The sign twin reports the same, so SPR and TIR return the booked -2c.
3. **The portfolio ruin penalty is blind.** A step after which some account keeps positive equity never takes the portfolio to zero, and a
   step that trades only from empty accounts (the funded ones flat) has a non-negative portfolio reward. A policy that keeps one account
   funded and flat therefore never incurs a penalty on portfolio ruin, and one that trades only from empty accounts never incurs a penalty on
   portfolio losses or drawdowns either, whatever the strength. From an empty account a position q reports E[(q|s| - c)^+] / 2 per step in a
   fair market and books -c.
4. **The twin rewards need no account structure.** Every account's credit is non-negative, so the portfolio's accounting is forgiving and the
   twin bracket of Proposition 10 holds for the portfolio pair. Under conditional sign fairness SPR and TIR have expectation -C (minus the
   portfolio's own cost) for every action, whatever the number of accounts and the assignment of positions to them.

## Proof
1. Account k's credit is E_k' - E_k - x_k = max(E_k + x_k, 0) - (E_k + x_k) = (-E_k - x_k)^+; the pooled credit is (-P - x)^+. With
   a_k = -E_k - x_k, sum_k max(a_k, 0) >= max(sum_k a_k, 0) because each term is at least a_k and at least 0.
2. With E_A = E_B = 0, x_A = eps q|s| - c and x_B = -eps q|s| - c; one of the two gross terms is q|s| and the other -q|s|, so the reward is
   max(q|s| - c, 0) + max(-q|s| - c, 0) = (q|s| - c)^+ for both signs. Booked: x_A + x_B = -2c. Reversing eps swaps the legs' outcomes and leaves
   the reward unchanged, so r = r', SPR = (r - r')/2 - 2c = -2c; TIR's realized portfolio sign is sign(q - q) eps = 0 (and r = r' sends it to its
   fallback in any case), so TIR = -2c.
3. P' = sum_k E_k' >= E_j' > 0 for the account j that keeps positive equity. A flat account's equity does not change, and an empty account's
   change is max(x_k, 0) - 0 >= 0, so a step that trades only from empty accounts has a non-negative portfolio reward. From E_A = 0 the
   account's reward is max(eps q|s| - c, 0), whose mean over a fair sign is (q|s| - c)^+ / 2; the booked mean is -c.
4. Bracket: Proposition 10 needs only non-negative credits on both branches with the gross reversed by the twin and the same known own cost;
   here the credit is a sum of non-negative account credits and the twin reverses every leg. SPR: Supplementary Proposition "Sign-paired
   reward under a general credit law", part 1, with the portfolio credit as the credit law (it depends on the pre-step state and the branch's
   outcome); at p = 1/2 its expectation is the booked one, -C. TIR: reversing eps swaps r and r', so the unordered pair, hence the magnitude
   estimate g_hat = max(r, r') + C (or 0 when r = r'), does not depend on eps given the state, action and move magnitude; the realized portfolio
   sign sigma = sign(sum_k q_k) eps is fair and independent of g_hat, so E[sigma g_hat] = 0 and E[TIR] = -C.

## Consequences tested by Study Z
- Reported-reward PPO should find the exploit in a split sleeve, as in the one-account sleeve (part 1).
- A ruin penalty measured on the portfolio should not remove it at any strength (part 3): the agent can keep one account funded and trade from
  the other, or hedge from two empty accounts (part 2), and neither is ever penalized. The same penalty measured per account sees every ruin.
- SPR and TIR, computed from the portfolio reward channel, a sign-reversed clone and the own cost only, should still remove it (part 4).
- In a pooled control with the same two legs, the portfolio penalty is the account penalty and should work as in Study V.

## Exact checks (`test_envs_z.py`, run before any training)
244,000 checks on random states of both structures, no failure: the accounting identity and non-negative credits (80,000), sub-additivity
(40,000), the hedge from two empty accounts (4,000: the reward is (q|s| - c)^+ for both signs, booked -2c, SPR = TIR = booked), the
blindness of the portfolio when one account is funded and the other trades, at every leverage (40,000: no portfolio ruin, no negative
portfolio reward from an empty account, the account count correct after a funded account's loss), the bracket on every step (40,000), and
fair-sign unbiasedness of SPR (40,000). (A first version with 12,000 park checks, at maximal leverage only and without the loss check, gave
216,000 checks and no failure.)

## Antecedents
Part 1 is the classical fact that a portfolio of options is worth at least an option on the portfolio (Merton 1973, Bell J. Econ. Manag.
Sci. 4:141-183). Parts 2 and 3 are the simulator's form of judgment proofing, in which assets are kept in one entity and the risk is run from
another (LoPucki 1996, The Death of Liability, 106 Yale L.J. 1). A recent market instance: on 26 March 2025 a trader held offsetting long and
short positions in separate accounts on one perpetual contract of a cryptocurrency exchange; the liquidated short was taken over by the
exchange's pooled vault, which faced losses of about 12 million US dollars before the contract was delisted.
