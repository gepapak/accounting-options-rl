# twin_reward: recover the booked reward from a censored reward channel

A small, dependency-light implementation (NumPy, Gymnasium) of the rewards and the reader of the manuscript's Section 6. Under the zero floor, and under the
other rules of the accounting ladder at their crediting steps, the reported step reward is `r = max(x, tau)`: the booked P&L `x` raised to a
threshold `tau` that the trainer cannot see (minus the reported equity under a zero floor, the carried deficit at a terminal or detection step,
minus infinity when no credit applies). Two cases are outside this form and not covered: the reported increment of an owner's claim while
insolvent (the zombie window of a settlement lag) and simulators that floor several cash writes in sequence; check the form first with
`cost_shift_test` (below). Repeat the step from the same state with the same action and the exogenous sign reversed (the *sign twin*) to get `r'`:

| pair | what it identifies | reward |
|---|---|---|
| `r != r'` | the favourable branch is uncensored: gross magnitude `g = max(r, r') + c` exactly | TIR returns the booked P&L exactly |
| `r == r'` | both branches censored (or `g = 0`): only `g in [0, r + c]` | TIR returns `-c`; SPR returns `-c` as well; TMR returns the whole threshold as noise |

`c` is the agent's own cost, which the agent knows. `tir` uses the identified value where the pair identifies `g` and the sharp lower end of the
identified set (`g = 0`) elsewhere; `spr` uses `|r - r'|/2`, a valid but generally non-sharp lower bound (`g = 10, c = 1, tau = 0` gives the pair
`(9, 0)`: `g = 10` is identified, SPR uses 4.5); `tmr` uses the sharp upper end. Properties (proofs in the manuscript's supplement, checks in `test_twin_reward.py` and in
`exploratory_claude_2026-10-31/identification/verify_identification.py`): exact on identified steps, pathwise error never above SPR's, unbiased
when the sign is fair, no gain manufactured where the pair coincides.

```python
from twin_reward import TwinRewardWrapper
env = TwinRewardWrapper(MySimulator(), estimator="tir")   # needs MySimulator.reverse_next_sign() and info["own_cost"]
```

Requirements on the simulator: it can be deep-copied before a step; it can reverse the exogenous sign of that copy's next step (pass `reverse=` if
the hook has another name); the agent's own cost of the step is known (pass `own_cost=` if it is not in `info["own_cost"]`). The wrapper never
reads a ledger or the accounting rule. Not covered: a legitimately convex payoff held by the agent (its sign-symmetric value would be removed), a
simulator in which the sign cannot be reversed, and carried deficits larger than every attainable booked P&L (then no query of this kind
identifies `g`; see the manuscript's Proposition on identification failure).

**A bracket that needs no single floor.** `twin_bracket(r, r', c)` returns `(lower, upper) = (2*SPR - r, r) = (-r' - 2c, r)`. If the accounting
only forgives (each branch reports its booked P&L plus a credit `>= 0`), the twin reverses the gross P&L and the own cost is known and the same on
both branches, the booked P&L lies in this interval, whatever the accounting rule: one floor, several floored cash writes, terminal liability. Its
width `r + r' + 2c` is the credit paid on the pair, so it is the booked P&L itself where nothing is forgiven; a negative width exposes an
undeclared charge. Over all such accountings it is the sharp identified set (with the realized sign known, intersect it with
`sign * (x + c) >= 0`), and SPR is its midpoint. Summed over an episode it brackets the booked return of any policy (check 11 of
`test_twin_reward.py`; `../../exploratory_claude_2026-11-03/twin_bracket.py` checks it on 60,800 stored test episodes and on the public engine's
own cash-writing methods).

**The reward diagram.** `reward_diagram.py` reads the whole line and not only its two points `t = 1, -1`. `scan(query, ts)` evaluates the reward on clones along `input(t) = reference + t * (input - reference)`, `read_diagram` returns the slopes, kinks and levels, `line_estimate` is the adaptive reading (one query where the twin's width vanishes, a few more where it does not), `minimax_estimate` the endpoint estimator of the range law and `LineRewardWrapper` the tail-slope reward (TSR) as a Gymnasium wrapper. It needs a clone that accepts a changed input and the own cost wherever the account cannot pay it; it assumes a payoff linear in the input over a range the simulator admits. A concave kink or a jump falsifies the floored model and says what the shape is instead. Exact checks: `test_reward_diagram.py` (88,702, in rational arithmetic), `test_line_wrapper.py` for the wrapper. Where it is exact, TSR's reward equals the ledger's.

**Without a clone.** `PooledMagnitudeWrapper` (manuscript proposition "Pooled magnitude", Study V) needs no copy of the simulator. On a
favourable step the reported reward gives the pseudo-magnitude `(r + c) / |q|`, which is the per-unit move whenever the favourable branch is
uncensored and exceeds it otherwise; the wrapper pools these draws over all steps and equities and rewards `gross_sign * |q| * m_bar - c`. It needs
`info["gross_sign"]` (the realized sign of the gross P&L), `info["position"]` and `info["own_cost"]`, and the assumption that the per-unit move is
independent of the agent's equity, action, history and the sign (Assumption M). The bias identity is `E[PMR - x | history] = |q| d (m_bar - mu)`,
with `d` the conditional mean of the gross sign and `mu` the true mean move: in a conditionally fair market (`d = 0`) the expectation is `-c` for any
pool fixed before the step; with an edge the reward is unbiased only if the pooled mean equals `mu`. Censored favourable draws enter the pool above
their move, so a pool fed by censored draws overstates `mu`: with `q = 1`, `m = 1`, `c = 1`, a carried deficit that sets `tau = 10` and a
favourable-sign probability of 0.75, the pool is 11 and the expected PMR reward is +4.5 while the booked expectation is -0.5, although Assumptions
C and M hold (check 10 of `test_twin_reward.py`). A finite pool can also lie below `mu`, and Assumptions C and M alone do not identify the mean (two
worlds with constant moves 1 and 10 behind a deficit of 10 return identical data). Use PMR where favourable draws are rarely censored, or
calibrate the mean on uncensored exposures.

```python
from twin_reward import PooledMagnitudeWrapper
env = PooledMagnitudeWrapper(MySimulator())
```

**Testing the single-floor assumption.** TIR and SPR assume that the reward is ONE monotone censor of the booked P&L (`r = max(x, tau)` with `tau`
independent of the own cost). That cannot be seen in one twin pair (manuscript proposition "Testing Assumption C", part 1), but it can be falsified
by shifting the own cost. `cost_shift_pairs` evaluates the twin pair of one step at several values of a cost knob, and `cost_shift_test` returns
`(violated, reason)` using `single_floor_fit`, which decides exactly whether any single-floor world (a sign, a magnitude `g >= 0` and one threshold)
reproduces all the pairs; it returns that world as a witness when one exists. A violation falsifies the assumption at that state. "No violation" is
not a certificate: the test has power only where a tested cost changes which branch is censored, and a violation that produces pairs a single floor
could have produced (a hidden fixed fee on a censored loss branch, an inflated gross credit, a constant output) is invisible to every reward-only
test. A hidden fixed fee is seen where both branches are uncensored, because there the pair must satisfy `r + r' = -2c`. A simulator that floors
several cash writes in sequence (for example a fee written first with its own floor) fails the test where a tested fee exceeds the account.
`cost_shift_test(pairs, exact=False)` is the earlier check of two implied conditions only (constant inferred magnitude, constant censored value),
which missed a threshold that moves with the cost and some incompatible grids.

```python
from twin_reward import cost_shift_pairs, cost_shift_test
pairs = cost_shift_pairs(env, action, [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6], set_knob=lambda e, v: setattr(e, "fee", v))
violated, reason = cost_shift_test(pairs)
```

In `test_twin_reward.py` the test flags none of 408 states of a single-floor toy and all 402 informative states with a fee above the account of a
two-write toy (and none of the 110 states in which the account can pay every tested fee); the exact test agrees with an independent brute force on
every generated and perturbed pair sequence of a lattice, and flags the violations of the fourth AI-assisted review that the implied-conditions
check passed. `../../exploratory_claude_2026-11-03/exact_cost_shift/` validates it further (no false alarm in 200,000 random single-floor worlds;
power by violation family) and re-runs the benchmark of `../../exploratory_claude_2026-11-02/contract_test.py` (the Study T sleeve, a two-write
variant of it and the public engine's own cash-writing methods) with both versions of the test.

Run `python test_twin_reward.py` (toy environments; under a minute).
