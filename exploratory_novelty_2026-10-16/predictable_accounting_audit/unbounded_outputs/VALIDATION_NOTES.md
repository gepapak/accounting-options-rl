# Fixed unbounded-law validation: results and limits

All sixteen cases completed their fixed 4,096 search and 65,536 confirmation
paths. The 32 one-sided lower bounds each allocate error .05/32, giving a
separate 95% simultaneous family guarantee under these constructed laws and
exact conditional-moment oracles. Cases do not constitute independent markets.
Within each case both representations use the same paths; caps use search only.

Both representations confirm the SAME three cases, all floor accounts at
alpha .8. The figures below are one-sided lower bounds, not true means or
confidence intervals for the difference between methods.

| Pareto shape | Cost law | Uncapped compensator | Capped mean-cost residual |
|---:|---|---:|---:|
| 1.5 | Fixed .1 | .3519915957 | .2824494683 |
| 1.5 | Proportional .05Z | .2225922649 | .1528586518 |
| 2.5 | Proportional .05Z | .0338808154 | .0265268580 |

There are no additional sign discoveries from predictable compensation in this
suite. All terminal cases and the shape 2.5 fixed-cost floor case at .8 abstain.
The compensator has higher sample variance than the capped comparator in four
of sixteen cases; neither universal variance reduction nor estimator dominance
is established. The difference of point means can reflect the baseline's upper
truncation, so it is not an estimator-accuracy comparison with known truth.

All eight alpha .95 cases respect the analytic uniform no-positive-profit
certificate. The thresholds are 29/31 for fixed cost at p=1.5, 47/53 for fixed cost
at p=2.5, and 19/21 for proportional cost at either shape. The guarantee covers
all adaptive policies in the declared flat/unit-trade action family, with the
same cost laws and no-edge assumptions. It is not a global real-market safety
claim. These thresholds are applications of the classical gain/loss ratio.

The proportional-cost cases have unbounded actual costs. The fair baseline
therefore replaces realized costs with their predictable means before upper
capping. Its expectation is unchanged and its lower bound depends on the known
mean cost, not a nonexistent maximum realized cost. Shape 1.5 additionally has
infinite gross and proportional-cost variance. Direct bounded empirical-
Bernstein evaluation of raw positive-charge rewards is unsupported because
their negative tail is unbounded; this excludes neither other robust methods
nor other information assumptions.

The replay used 7.00 CPU and 7.06 wall seconds on one numerical thread, with
35,651,584 global payoff draws/logical slots, 5,695,585 active transitions and
conditional-curve scalar evaluations, 8,347 vector curve calls, and about 1.47
seconds inside those curve evaluations. These oracle counts concern active
trajectory queries; analytic moment/range initialization and deterministic
oracle checks are additional arithmetic included in total runtime. Both methods
share simulation; standalone method runtimes were not independently benchmarked.

The mandatory first trade gives strict expected booked return at most minus
the known mean unit cost (.1, .15 or 1/12 as appropriate). All supports and oracle
curves use analytic first moments, never sample maxima. The deterministic
positive-part checks match closed-form initial one-step expectations and verify
stable debt drift at extreme ledgers without a separate Monte Carlo oracle test.

`SOFTWARE_REPLAY.md` preserves the serialization failure and identical-stream
replay. The original failed source has a byte-matching snapshot; the plan is
unchanged. Counts and runtime above describe the completed replay, not the failed
attempt; replaying identical streams supplies no extra independent observations.
No learner, old artifact, market data or Study R input is used.
