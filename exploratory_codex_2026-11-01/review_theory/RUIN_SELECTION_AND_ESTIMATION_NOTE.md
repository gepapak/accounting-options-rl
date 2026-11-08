# One retrospective feasibility check and the mean-error regret bound

This is an exploratory review of existing data, not a new study or a new AI
method. No training, environments, or models were run. The script read 48
existing Study V JSON files (RUIN1/5/25, eight seeds each, Parts A/C) and only
wrote new review files.

## Selecting the ruin penalty without booked returns

`retrospective_ruin_selection.py` first extracted only the existing
`audit_edge.episodes.tir` arrays, selected the strength maximizing mean TIR
return, and persisted `RUIN_SELECTIONS_BY_TIR_ONLY.json`. Only afterward did
that script extract the booked arrays to evaluate the already persisted
choices. This is a reproducible separation of the computational criterion;
the study outcomes were already known, so it is neither blinded nor prospective.

All numbers below are percentage points of allocation K0 per episode/week;
signed error is TIR minus booked return. Each full-sample row aggregates eight
agents and 100 episodes per agent.

| Part | Candidate | Mean TIR | Mean booked | Signed error |
|---|---|---:|---:|---:|
| A | RUIN1 | 11.669636 | 11.744432 | -0.074797 |
| A | RUIN5 | 18.967916 | 18.967913 | +0.000003234 |
| A | RUIN25 | 17.627111 | 17.627110 | +0.0000004375 |
| C | RUIN1 | -2.522533 | -2.346076 | -0.176457 |
| C | RUIN5 | 17.186744 | 17.186744 | -0.0000000000000174 |
| C | RUIN25 | 4.175732 | 4.175732 | -0.0000000000000363 |

TIR alone selects **RUIN5 in both parts**. After the selection, the observed
booked ordering is RUIN5 > RUIN25 > RUIN1 in both parts, identical to TIR's.

One fixed episode-index split gives the same answer. Select on ordinal
episodes 1,3,...,99 (50 per agent); evaluate on 2,4,...,100. RUIN5 is selected
in both parts, and is also highest by booked return on the even episodes:

| Part | Selected from odd episodes | Even-episode TIR | Even-episode booked | Signed error |
|---|---|---:|---:|---:|
| A | RUIN5 | 18.443912 | 18.443909 | +0.000003500 |
| C | RUIN5 | 16.373915 | 16.373915 | -0.0000000000000105 |

This directly refutes the categorical necessity claim in
`sections/discussion.tex:10` that the working strength 'can be identified only
from booked returns of trained agents'. A defensible replacement is: tuning
requires an evaluation criterion aligned with booked performance; the
reported return alone is unreliable, while a sufficiently accurate twin audit
can provide such a criterion. This check shows feasibility on these stored
candidates, not that every reward-only criterion or new regime will succeed.

Limits: the full-sample comparison reuses selection episodes; odd/even episode
ordinals share the same market draw and can have overlapping windows. Neither
is independent market confirmation, and no p-value or confidence guarantee is
claimed. Return selection does not itself certify exploit removal. A trainer
with this clone/sign-twin access could train TIR directly, so this audit-based
tuning exercise is a practical consequence of the existing estimator rather
than an additional AI contribution. All registered verdicts remain unchanged.

The result JSON contains all candidate/full/odd/even metrics and 48 input
SHA-256 hashes; the selection receipt is hashed before the booked evaluation
phase of this script.

## Finite mean-estimation error: valid conditional regret bound

For a frozen magnitude predictor mu_hat(s), suppose the following hold on one
joint calibration event of probability at least 1-delta:

1. For every reachable state/action, magnitude-sign factorization holds with
   true conditional magnitude mean mu(s), and the original transition kernel
   is unchanged by replacing the reward.
2. `|mu_hat(s)-mu(s)| <= eta` uniformly over those states/actions.
3. `|q(s,a)| <= Q` and `|E[sigma|s,a]| <= d_max <= 1`.

Then conditional expected one-step reward error is at most
`epsilon = Q*d_max*eta`. For every policy in a horizon H,
`|V_hat(pi)-V(pi)| <= H*epsilon`. If pi_hat maximizes the estimated objective
and pi_star maximizes the booked objective,

```
V(pi_star)-V(pi_hat) <= 2*H*Q*d_max*eta.
```

Proof: add and subtract estimated values of both policies; the middle term
`V_hat(pi_star)-V_hat(pi_hat)` is nonpositive. Sum the per-step error along
each original trajectory. For discount gamma<1 replace H by `1/(1-gamma)`;
if optimization is xi-suboptimal in the estimated objective, add xi. This is
the standard plug-in reward-error argument, not a new theorem of substance.

Important boundary: ordinary pointwise/marginal interval coverage is not a
uniform bound over the states a policy selects. An on-policy error bound on
only the realized policy also cannot compare against an unseen optimal
policy. For an online adaptive pool, include the pool/history in the state
and require the bound uniformly over the relevant augmented trajectories;
otherwise the displayed regret guarantee does not follow.

C alone gives `U_i=m_i+b_i` with unknown b_i>=0; it gives no upper bound on
pool contamination, no representative-sampling guarantee, and no finite eta.
A fixed number of representative iid bounded latent draws would admit a
Hoeffding error bound. Censored pseudo-draws additionally need a justified
contamination bound, and adaptive sample counts need a suitable sequential
bound or an independently fixed calibration size. Consequently, the regret
bound is useful if its calibration premises can be proved or audited; it
cannot currently be claimed for Study V's running pool just from C and M.
