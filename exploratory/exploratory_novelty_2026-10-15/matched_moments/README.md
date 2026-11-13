# Bounded matched-moment accounting counterexample

Exploratory mathematical example, 15 October 2026. No study registration, real
market data, learner outcomes or Study R results are used. The script is a new
standalone artifact and is not part of a registered source/hash set.

This example asks what information is needed to determine the existence of a
positive **one-action accounting incentive**, rather than predict acquisition.
It applies the classical stop-loss transform in the paper's account model.

Both cases start from cash equity 1, execute position 1, pay deterministic cost
1/1000 = 0.001 and charge alpha = 4/5 = 0.8 of every injection. Gross payoff is an
independent fair sign times the following bounded magnitude:

| Magnitude | Probability in A | Probability in B |
|---:|---:|---:|
| 1/4 | 1/2 | 1/5 |
| 1/2 | 0 | 3/4 |
| 3/4 | 1/2 | 0 |
| 3/2 | 0 | 1/20 |

Both therefore have gross mean 0, gross variance 5/16 and mean absolute gross
payoff 1/2. Net means are -0.001, net variances remain 5/16 and mean absolute
net payoff is also 1/2, because every positive magnitude exceeds the cost.
Nominal fee, executed cost, dose, initial equity and the common valid
gross-payoff bound 3/2 also coincide. Thus dose, cost/volatility,
dose*volatility/cost and gross/net mean-absolute-payoff heuristics all tie.

With net payoff X and equity u, the true one-action charged reward is

    m_alpha(u) = -E[cost] + (1-alpha) E[(-u-X)^+].

The exact results at u = 1 are:

| Quantity | A | B |
|---|---:|---:|
| Expected injection | 0 | 501/40000 = 0.012525 |
| Expected charged reward | -1/1000 = -0.001 | 301/200000 = 0.001505 |
| Expected booked payoff of the action | -1/1000 | -1/1000 |

The tail beyond the actual cash floor changes the sign despite all matched
features. B supplies a feasible positive charged-reward, negative booked-return
witness: trade once and remain flat. A does not supply that witness at this
equity. This is **not** a claim that A is safe against dynamic policies, that
agents learn differently, or that the diagnostic has empirically beaten a
prediction baseline. The already documented two-step counterexample explains
why absence of a one-action witness cannot establish safety. Under full booking
(alpha = 1), the fixed action has expected reward -0.001 in both cases; the
positive-reward gate stays flat and has value zero in both. The example thus
also illustrates the existing accounting repair without claiming agents learn
that repair.

Run from the repository root:

```powershell
python exploratory_novelty_2026-10-15/matched_moments/exact_tail_counterexample.py
```

When invoked from the repository root, the command writes `matched_moments.json`
next to the script. It uses only Python's standard library and exact rational
arithmetic, checks every matched feature and account identity, and records the
script hash. It does not import simulator, DP or training modules.
