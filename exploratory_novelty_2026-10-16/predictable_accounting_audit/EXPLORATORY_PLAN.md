# Exploratory predictable-accounting audit: fixed evaluation plan

16 October 2026. Written before new evaluation draws or inspection of held-out
payoff summaries. All previous results, including failed R and the selected-case
certificate, are development information. This is not a registered learner study.
No learners are trained, no existing study output is overwritten, and no claim
of acquisition prediction is tested.

## Proposed extension and information access

Replace realized forgiveness/cost noise by the predictable conditional drift
of the debt/floor potential. The generator/Doob-martingale construction is
established prior art; the proposed specialization supplies a first-moment,
state-uniform range, avoiding clipping, and a kernel-specific partial-charge
repair certificate for all admissible policies. It requires evaluable conditional
one-step expectations and moment envelopes, not merely black-box samples.

For X=G-C, EG=0, C>=0 and d(n)=(-n)+, let h_a(n)=E d(n+X) and
Delta_a(n)=h_a(n)-d(n). The proposed range is 0<=Delta<=h_a(0).
Terminal estimator: sum[(1-alpha)*gamma^(H-1)*Delta_a(N_t)-gamma^t*c_a].
Floor estimator: sum gamma^t[(1-alpha)*h_a(E_t)-c_a].
Both should have the same mean as charged return under the declared kernel.
They need not have smaller variance for every policy.

## Development/validation

Exact rational small-law checks establish identities, ranges, charge threshold
and variance counterexamples independently. Direct empirical row integration
checks numerical stop-loss queries. Ledger paths are checked against the old
immutable simulation kernel, including floor and terminal timing. Any failure
must be fixed and disclosed before evaluation starts; no held-out outcomes may
select another candidate library or budget.
The frozen floor implementation is replayed at its declared .10 fee without
changing its module; the terminal reference and new evaluation use fee1.
All methods receive policy-specific finite-support envelopes (including the
smaller actions of constant policies), rather than unnecessarily loose max16
envelopes. A conservative finite-row arithmetic allowance expands the oracle
estimator's support and is also deducted for possible integration bias.

## Held-out-kernel comparison

Only split metadata was inspected before this plan: each public Danish panel
has TRAIN and TEST rows. Earlier audit development used TRAIN. This evaluation
uses every TEST row after excluding its first 200 rows, matching the existing
warmup convention. The rows were present in older learner research: they are
held out from this audit's development, not newly collected markets or unseen
human information. Each resulting empirical IID fair-sign kernel is the law
being audited; confidence is conditional on that law, not future-market coverage.

Fixed 24 cases: DK1/DK2 x floor/terminal x initial equity fractions
{.1,.5,1} of 20000 EUR x H{16,96}; fee1 EUR/MWh, gamma.99 and the original
25%-of-capacity volume cap. Three independent search repetitions per case.
No payoff-dependent pruning or selective case omission.

All search methods receive the SAME 13 admissible candidate policies:
one action16 (immediate for floor, deadline for terminal); constants .25/1/4/16;
barrier16 at .5/1/2 times initial equity; two-stage16 policies with barrier pairs
(1,2),(1,.5),(2,.5),(2,2), switch at half the horizon; forced-prefix myopic using
the accounting conditional drift. Feedback sees only clipped ledger and clock.
All non-one-action policies have a mandatory first trade. Constants use their
declared leverage from the first step; no adaptive advantage is presumed.

Methods: raw, upper-capped raw, upper-capped accounting residual, upper-capped
mean-cost residual, and uncapped predictable accounting drift. The mean-cost
residual replaces realized fees by their predictable conditional means and
isolates cost-noise removal from integration of the forgiveness potential.
This fifth baseline was added before any evaluation draw, on theoretical review,
not on held-out outcomes. Each uses its own SEARCH criterion with the same
candidate set and paths. Caps {32,128,512,2048,8192,32768,131072} are selected
on search data only; the predictable estimator has no cap. Search criterion is
a projected empirical-Bernstein lower bound at the planned confirmation n,
used only as a heuristic. No DP or learned value function is used.

Each method receives at most262144 full-horizon search transitions and2097152
confirmation transitions per case/repetition. Let n_search=floor(262144/(13H))
and n_confirm=floor(2097152/H). Whole trajectories are the independent units.
Logical budgets count all H state transitions including flat ones. Shared
simulation across methods is disclosed; actual unique transitions, executed
actions, drift queries, preprocessing and wall/CPU time are recorded separately.
Equal simulator budgets do not imply equal arithmetic/runtime.

Also evaluate a selection-neutral representation ablation: the predetermined
constant16, barrier1 and two-stage(1,2) controller respectively in repetitions
0,1,2. Each gets the same confirmation budget, and all five representations.
This avoids evaluating only controllers selected to favor the new estimator.
All caps for this controller still use search data alone.

Family error .05 is split over 72 case/repetitions times ten reported lower
bounds (five selected-method bounds plus five neutral-policy representations).
Methods may share confirmation randomness; the union bound needs no independence
across these bounds. Search and confirmation RNG streams remain independent.
Whole-trajectory n is fixed, with no early-success stopping.
Search seed base4100000, confirmation4200000, deterministic case/repetition
offset100*case+repetition. Numeric threads1. A900CPU-second guard records partial
results with no complete-family discovery claim if not all planned paths finish.

Every case includes exact isolated-action integration over all original levels,
including flat, shared by all methods. Report new certificates only at states
where this reference is nonpositive, as well as overall confirmation counts.
The same-action full-charge control is analytically nonpositive. Also report
the kernel-specific all-policy partial-charge repair boundary and its assumptions.

## Reporting and provenance

Retain every selected policy, cap, heuristic score, confirmation lower bound,
abstention, actual n, declared range, input/script/plan hashes, source/model
assumptions and runtime. Report method totals and all three repetitions, and
separate neutral-policy estimator results. No algorithm superiority conclusion
from a single selected case or from algebraic horizon copies. All failures
remain. No design extension after inspecting this benchmark's outcomes.

An optional separately planned analytic-Pareto validation can check the same
bounded-drift theorem without finite price variance; it is constructed-law
validation, not another real market or held-out optimizer evaluation.
