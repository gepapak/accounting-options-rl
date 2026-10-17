# Independent predictable-accounting evidence audit

All checks pass without new Monte Carlo draws, DP solves or learner training.
`verify_predictable.py` independently checks 720 held-out, 45 fixed-policy,
32 constructed Pareto and 9 matched-baseline lower bounds. The first three
families require 18,482 assertions and 25 source/plan/data hash checks; the
matched checker separately verifies seven hashes. All recorded empirical-
Bernstein radii and lower-bound calculations recompute exactly. Independently
integrated supports differ by at most 5.83e-11 EUR and isolated-action references
by at most 3.56e-15 EUR. `verification.json` is the full audit output.

The shipped outputs retain means and variances, not individual confirmation
paths. This audit checks those moments for arithmetic, support, paired-output
and accounting consistency; it does not independently reconstruct them from
random draws. Reproduction of those moments requires rerunning the original
fixed scripts in a new location, which was not done here.

## Separate evidence families

| Representation | Held-out selected, /72 | Held-out neutral, /72 | Selected development case, /9 at n=32768 |
|---|---:|---:|---:|
| Raw charged return | 7 | 5 | 0 |
| Upper-capped raw | 17 | 18 | 0 |
| Capped realized-cost residual | 24 | 24 | 0 |
| Capped mean-cost residual | 24 | 24 | 0 |
| Uncapped predictable compensation | 36 | 33 | 3 |

Held-out results use the DK1/DK2 TEST empirical IID laws, two accounting rules,
three equities and two clocks: 24 cases and three independent search repetitions.
The law was held out from development of this audit; its full empirical
conditional moments are provided to the oracle. This is not sample-only
evaluation under an unknown future market law. Whole trajectories are IID units;
72 case/repetitions are not 72 markets. The 720 lower bounds allocate .05/720
each for a separate 95% family guarantee. Confirmation randomness is independent
of search, and shared paths across methods do not invalidate the union bound.

The exact one-action reference is positive in 12/24 cases. Predictable selection
confirms all 12 in all three repetitions; every other certificate is also in
those cases. **No method finds a new state missed by the exact one-action
reference.** Selected-policy counts support conditional confirmation capability,
not additional state discovery or a general optimizer advantage.

The neutral comparison uses the same fixed controller and confirmation paths
for all five representations within each case/repetition. Relative to either
residual comparator, predictable compensation has nine additional positive
bounds and no losses. Across each fixed controller's 24 cases, raw counts are
3/0/2, capped-raw counts 6/6/6, both residual counts 8/8/8, and predictable counts
11/11/11, respectively for constant16, barrier1 and two-stage(1,2). These three
controllers are different; the neutral rows are not three search replications
of one fixed policy. No inferential paired difference or universal variance
claim follows from these descriptive counts.

Search uses all 13 fixed candidates and at most 262144 full-clock transitions
per method/case/repetition: n=1260 at H16 or n=210 at H96, both exactly 262080
logical transitions. Confirmation uses n=131072 or n=21845, giving 2097152 or
2097120 logical transitions. Actual shared computation is 473948288 transitions
and 533725901 conditional-curve scalar queries (including 285 outside recorded
rollouts), with eight preprocessing moment queries. Runtime is 120.703 CPU and
122.962 wall seconds. Equal logical simulator budgets do not imply equal CPU;
exact conditional integration is additional information and arithmetic.

## Frozen development-case estimator comparison

The nine fixed policies from the matched-baseline family receive SEARCH8192 and
fresh CONFIRM32768 each on the same known selected DK1 TRAIN state. Forty-five
bounds allocate .05/45 each, separately from the held-out family. This is a
smaller-budget fixed-policy confirmation comparison, not new market validation
or a new search result. Predictable compensation confirms exactly:

| Policy | Predictable lower bound, EUR | Exact booked mean or upper bound, EUR |
|---|---:|---:|
| Frozen adaptive | 8.0644976717 | -3.3355318258 (upper bound) |
| Leverage16 in final16 periods | .3259685989 | -22.1733132726 |
| Leverage16 in final48 periods | 4.5427946765 | -78.7996121297 |

All four comparator representations abstain on all nine policies. The source
overrides the imported runner's module DELTA for confirmation and restores it
afterward. Python's definition-time default for the separate SEARCH helper
remains .05/720; thus that projected SEARCH heuristic uses a stricter allocation
than the reported .05/45 confirmation bounds. This does not invalidate any
confirmation event or independence. The actual source behavior is retained and
disclosed; there was no retuning or rerun. All 45 confirmation bounds were
independently checked at their actual .05/45 allocation.

This family uses 22487040 active simulated transitions, 35389440 represented
outer-clock slots and 20829763 conditional queries, in 5.328 CPU/5.387 wall seconds.
Every policy receives the same deadline compiler, paths and estimator choices.
Policy-specific supports give low-leverage constant comparators their known
tighter cost/payoff bounds. Numerical oracle allowances expand supports and are
also subtracted for expectation bias.

## Matched larger-budget baseline and Pareto limits

The earlier matched family uses SEARCH8192 and CONFIRM524288 for each of nine
fixed policies with .01/9 lower-tail error, a separate 99% family guarantee.
It confirms both the adaptive controller (LCB8.2022604721 EUR) and simple
leverage16 during the final48 periods (LCB2.4277956047 EUR). All isolated allowed
actions are nonpositive in this state. Thus multi-step exposure supplies a
witness beyond isolated actions, but **adaptive feedback is unnecessary here**.
The constant16 full-clock lower bound -.0786598503 is abstention, not evidence
of negative expected value. Full results remain in `../matched_multistep_baselines`.

The 16 constructed Pareto cases allocate a separate .05/32 across their two
representations. Both confirm the SAME three floor/alpha.8 cases: shape1.5 with
fixed or proportional cost and shape2.5 with proportional cost. All eight .95
controls satisfy the exact all-policy repair frontiers. Uncapped compensation
has greater sample variance than the capped comparator in four cases, IDs
7/9/11/15. There is no extra sign discovery or universal variance improvement.
The analytic oracle and mean-cost comparator remain valid with unbounded actual
fees and infinite variance at shape1.5. Raw bounded empirical-Bernstein bounds
are unavailable at positive charge; this does not exclude other robust methods.

## Retry provenance and reproduction

The held-out original attempt retains only its frozen input manifest. AST
comparison verifies that the second source changes only its output directory
and NumPy-scalar JSON encoding; all numerical kernel, plan, input hashes, seeds,
policies, budgets and confidence allocations are identical. The Pareto failed
source snapshot matches its pre-draw hash, and the sole corrected expression
casts h0 to a native float without changing its double-precision value. Replays
reuse the same streams and supply no extra independent observations. Runtime
counts describe completed attempts and exclude failed serialization work.

Run from the code repository:

```text
python exploratory_novelty_2026-10-16/predictable_verification/verify_predictable.py
python tools/verify_studies.py
```

The first command is read-only unless `--write` explicitly regenerates this
folder's verification JSON. The repository verifier preserves all previous 501
checks and adds 51 checks, passing all **552** with exit0. Its only revision notice
is the pre-existing FI panel rebuilt from revised eSett data. The verification
log is `verify_studies.log`. No registered file or earlier output was edited.
