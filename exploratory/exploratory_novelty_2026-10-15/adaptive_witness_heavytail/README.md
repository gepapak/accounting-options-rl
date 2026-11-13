# Exploratory generative certificate with unbounded prices

This standalone capability check implements the accounting residual certificate
described in `../adaptive_witness/THEORY.md`. It reads no studies, market panels,
trained policies or Study R artifacts. No learner is trained. The fixed synthetic
laws and exact means were chosen as mathematical tests, not independent real
market evidence. The pre-calculation plan is `EXPLORATORY_PLAN.md`.

Both laws have independent fair signs, Pareto magnitude with minimum 1, cash
equity 2 and deterministic executed cost .1. Shape 1.5 has finite first moment
and infinite variance; shape 2.5 has finite variance, still unbounded support.
The audit takes one unit action. Its expected booked profit is exactly -.1.

Search uses 8192 paths to select an upper cap from the fixed grid, then freezes
it. Independent confirmation uses 262144 paths per law. A bounded lower
certificate for capped forgiveness yields a simultaneous affine lower bound
for the SAME fixed action at every charge alpha. The two law-wise bounds use
delta .025 each (95% joint coverage for this family), with no extra alpha-wise
tests. This does not combine with other benchmark families into a paper-wide
95% guarantee. Upper capping and search/confirmation splitting are established
tools; the accounting identity supplies the bounded lower endpoint from costs.

Search and confirmation use separate seeded RNG streams. Within each phase,
the two Pareto laws consume sequential, nonoverlapping portions of that stream;
their disturbances are not paired common random numbers across laws. This
clarifies the implementation without changing the frozen exploratory plan.

| Pareto shape | True proxy mean, alpha .8 | Certified lower bound | Witness at .8 | Certified alpha interval |
|---:|---:|---:|---|---|
| 1.5 | .0450952500 | .0233555169 | confirmed | [0, .8378670002) |
| 2.5 | -.0745446930 | -.0758950197 | inconclusive | [0, .1702959409) |

Both exact capped means lie above their confidence lower bounds. Full booking
has mean -.1 for both laws. The selected residual caps were 64 and 4. A direct
bounded empirical-Bernstein calculation on raw alpha-.8 reward is unsupported:
its negative tail is unbounded, and a sample extremum is not a support bound.
This comparison does not exclude other robust-mean methods or assert universal
variance reduction. The normalized one-step noise score is undefined for shape
1.5 because its gross/net variance is infinite; the accounting certificate
uses no second-moment or finite-price-bound assumption for the untruncated law.

The exact expectations follow by integrating the Pareto tail. The code records
every cap score, the selected cap, exact/capped truth, confirmation statistics,
confidence coverage, script/plan hashes and runtime. Its numerical work took
about .09 seconds excluding imports on this machine. Failure to confirm at a
specified alpha is not a safety result. The charge interval is policy-specific;
it does not identify a global minimum repair charge.

Reproduce from the code repository root:

```powershell
python exploratory_novelty_2026-10-15/adaptive_witness_heavytail/bounded_cost_certificate.py
```
