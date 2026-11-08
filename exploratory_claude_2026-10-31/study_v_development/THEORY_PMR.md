> **Erratum (2 November 2026).** Two statements below are corrected in the manuscript (Proposition 6 and its proof, Supplementary Section S20): (i) a finite
> pool is not identically the population mean when uncensored and is not necessarily above it when censored; the valid pathwise statement is mean(U) >= mean(m)
> over the same pooled steps; (ii) the conditional bias identity is E[PMR - x | F_t] = |q_t| d_t (mu_hat_t - mu), so PMR is conditionally unbiased with the exact mean
> and, in a conditionally fair market (d_t = 0), has expectation -C_t for any predictable pool. Assumptions C and M alone do not identify mu (two constant-magnitude
> worlds, m = 1 and m = 10, with a deficit of 10 return identical data), so the claim that M escapes the 4/9 bound needs an additional calibration assumption.
> The registered PMR reward and Study V's results are unchanged.

# Pooled-magnitude reward (PMR): a clone-free route to the booked reward (exploratory, 31 October 2026)

Setting of the paper's Assumption C: the reported step reward is `r = max(x, tau)` with booked P&L `x = sigma * g - C`, `g = |q| m`,
`m >= 0` the exogenous per-unit move, `sigma` the realized sign of the gross P&L, `C >= 0` the agent's own cost, `tau` a hidden threshold
fixed by the pre-step state. The sign twin (Propositions 6-8) queries a clone of the simulator with the sign reversed.

## What a trainer has without a clone
The action (hence `q` and `C`), the realized exogenous sign (hence `sigma`) and the reward `r`. No ledger, no rule, no `tau`.

## Assumption M (exogenous magnitude)
`m_t` is drawn from a law that depends on the exogenous state (time, market regime) but not on the agent's equity or action, and given that
state `m_t` is independent of the sign `sigma_t / sign(q_t)` (the market's sign eps_t may be predictable from a signal; its size is not
tied to the agent's account).

## Proposition (pooled identification; exploratory statement, proof below)
Under C and M:
1. On a favourable step (`sigma = +1`, `q != 0`) whose favourable branch is not censored (`g - C > tau`), `(r + C) / |q| = m` exactly.
2. Let `m_bar = E[m_t]`. If a draw is pooled from favourable steps only, the pool mean is `m_bar` up to the effect of censored favourable
   branches, which can only raise it (a censored favourable step reports `tau >= x`); the bias is at most `P(censored | favourable)` times the
   excess of the censored report.
3. The reward `x_hat = sigma |q| m_bar - C` has `E[x_hat | s, a] = E[x | s, a]` for every state `s` and action `a` when `m_bar` is exact,
   and in a fair market `E[x_hat | s, a] = -C(a)` in *every* state, including those where a credit is certain. No credit can be
   gambled for, and the expected reward of a legitimate edge is the booked one.
4. Pathwise it is not exact: `|x_hat - x| = |q| |m_bar - m_t|`; it is also blind to any signal that predicts the *size* of the move
   (a conditional pool `m_bar(features)` removes that blindness at the price of a regression).

Proof sketch. (1) On the favourable branch `x = |q| m - C`; if `x > tau` then `r = x`. (2) The pooled mean is `E[(r+C)/|q|]` over
favourable steps; with no censoring it is `E[m | sigma = +1] = E[m]` by M; censoring replaces `m` by `(tau + C)/|q| >= m`. (3) `E[sigma | s, a] |q| m_bar - C`
equals `E[x | s, a]` because `E[sigma m | s, a] = E[sigma | s, a] E[m]` by M. In a fair market `E[sigma | s, a] = 0`. QED.

## How it relates to the twin
| | sign twin (SPR, TIR) | pooled magnitude (PMR) |
|---|---|---|
| needs | clone of the simulator and a sign override | nothing beyond the logged action, sign, cost and reward |
| assumption on `m` | none (the pair shares `g`) | Assumption M (invariance of the law of `m` across states) |
| per step | exact where the favourable branch is uncensored | not exact (`|q| |m_bar - m|`) but unbiased given the state |
| carried deficit that hides `g` | cannot do better than regret 4/9 | not affected if M holds, because the pool is built elsewhere |
| legitimate edge in the move's size | kept (the pair holds the real `g`) | lost unless pooled conditionally |

The regret bound of Proposition 8 uses two worlds with different `g` at the same censored state; Assumption M excludes them by tying the law of
`m` across states, which is the price of dispensing with the query.

## Exploratory check (pmr_accuracy.txt)
Random-action rollouts of the edge sleeve, mean reward in percent of K0 per quarter-hour at equity <= 0.1 K0 and leverage 16: booked -0.312,
reported +0.117, SPR -0.288, TIR -0.312, PMR -0.271 (DK1); booked -0.296, reported +0.155, SPR -0.301, TIR -0.297, PMR -0.299 (DK2). Pathwise error of
PMR is large (mean absolute 0.33 on uncredited steps, because the booked magnitude varies), its bias given the state and action is small.
