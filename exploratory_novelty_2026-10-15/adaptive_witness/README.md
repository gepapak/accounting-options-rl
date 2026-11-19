# Exploratory adaptive accounting witnesses

This directory was created on 15 October 2026 after the completed studies were
known. It contains a method note and exact constructed examples, not a new
registered experiment. No existing study, manifest, raw data, verdict or Study R
artifact is modified or read. No learner is trained and no DP is solved.

- `THEORY.md`: assumptions, event-gated continuation identity, independent clipped
  lower certificate, fixed-policy liability-charge interval and limitations.
- `exact_adaptive_benchmark.py`: standard-library rational path enumeration of a
  fixed small policy library, plus an exact unbounded-support certificate example.
- `exact_adaptive_benchmark.json`: reproducible full outputs including path
  values, failures, accounting identities and script hash.

Run from the code repository:

```powershell
python exploratory_novelty_2026-10-15/adaptive_witness/exact_adaptive_benchmark.py
```

With independent fair gross +/-1 and gamma=.99, the two-step floor-prefix example
has one-action value -.05 and constant-trading value -.07475, but an adaptive
trade-then-floor-continuation policy yields charged +.0985 and booked -.598.
A second example starts at equity1.5 with cost .1: the first loss leaves positive
equity .4, so a floor-only gate misses; the positive-continuation gate yields
+.02375 versus -.1 for one action and -.02575 for constant trading. Horizon3,
partial/full booking, high-capital failures and a terminal-liability version are
also retained. A nonpositive library result means ABSTAIN, not safety.

For a stationary IID no-edge terminal model, the note proves a clock compiler:
each fixed local controller weakly improves when placed at the deadline, and its
charged/booked path values then scale from the local episode by gamma^(H-T).
The exact T=2 controller is negative when embedded early in H=48 but positive at
the deadline. One local certificate can cover every outer horizon by this
algebraic scaling; effect size still decays. A predictable-edge counterexample
shows why the assumptions matter.

The residual identity is not a universal variance improvement: the first
two-step example increases variance from .29560675 to .48862675. Its advantage is
the accounting-derived cost-only lower bound, which permits independent upper
clipping and bounded one-sided certification even in certain unbounded-payoff
models. Raw reward clipping remains a required comparator when its lower bound
is available. The separate empirical branch is in `../adaptive_witness_empirical/`.

The underlying rollout, martingale, truncation and concentration arguments are
classical. The contribution under evaluation is their accounting-specific policy
grammar and certificate, with a demonstrated multi-step incentive capability.
These examples make no claim about RL acquisition, training-budget prediction,
empirical superiority, an optimal policy, or a global repair boundary.
