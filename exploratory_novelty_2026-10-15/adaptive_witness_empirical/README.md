# Exploratory adaptive witnesses in empirical terminal accounts

The final full-clock comparison found **no adaptive-search discovery gain beyond
the exact isolated-action reference at its fixed search/confirmation budget**. Structured, random, barrier and
forced-prefix-myopic search each certified 8 of 14 states, versus 6 for sampled
one-action confirmation and 9 for the exact all-leverage reference. No method
certified a state outside that reference's positive set. The two earlier
benchmarks and all unsuccessful candidates are retained. This is an
incentive-auditing experiment, not an acquisition-prediction or trained-agent
experiment.

A later, explicitly selected-case test with fresh randomness does certify one
multi-step policy beyond the isolated-action reference: its one-sided 99% lower
bound is **11.09292566 EUR**, with exact negative expected booked return. This
larger-budget capability test preserves the benchmark rates and supplies no
comparative optimizer advantage; its selection and scope are detailed below.

Source-of-truth guide: `selected_case_followup.json` is the fresh selected-case
capability certificate and `selected_case_selection.csv` preserves its full
development-data ranking. `full_clock_summary.json` and
`full_clock_certificates.csv` are the final full-duration result;
`exploratory_extension_*` files are the earlier short-window compiler result;
`exploratory_empirical_*` files are the initial early-start benchmark.
`initial_max16_reference/` additionally preserves its original narrower
reference before the disclosed all-leverage correction. Each family has its
own plan, independent search/confirmation phases and explicitly separate .05
error allocation. None is a new registered learner study.

The experiment uses the finite DK1/DK2 TRAIN magnitude/cap distribution, IID
rows and independent fair signs, fee 1 EUR/MWh, gamma .99, solvent initial cash
fractions {.1,.5,1} and horizons {4,16,48}. This law is explicit and narrower than
chronological market execution. The script reads no DP or learned outcomes and
no Study R artifacts. The independent confirmation uncertainty concerns fresh
simulation paths under this fixed empirical law; it does not establish a
population confidence statement for historical markets.

## Reusable computation and certificate

`benchmark_adaptive_witness.py` provides an empirical payoff kernel, a
nonanticipative ledger/clock feedback simulator, a candidate library and
`selected_certificate`. Search and confirmation use independent RNG streams.
Each of the five methods has 16,384*horizon logical search transitions per case;
one, four or 32 candidates divide that budget differently. Each selected policy
has 32,768 fresh confirmation paths. Flat time slots count in this initial
benchmark's transition budget.

Reusable interfaces are ordinary importable Python functions:

```python
kernel = EmpiricalKernel(spread_magnitude, cap_mwh, fee)
policy_actions = action(policy, ledger, step, horizon, initial, kernel, gamma)
paired = rollout(kernel, policy, row_indices, independent_signs, initial, gamma)
bounds = path_bounds(kernel, policy, initial, horizon, gamma)
bound = empirical_bernstein(samples, known_lower, known_upper, delta)
certificate = selected_certificate(
    kernel, search_frozen_policy, search_frozen_residual_cap,
    search_frozen_raw_cap, independent_confirmation_draws, initial, delta
)
calendar_record = compile_calendar(certificate, active_duration, horizon, initial, kernel)
```

`compile_calendar` is defined in `benchmark_latest_placement.py`; the other
functions are in the initial benchmark module. `rollout` returns charged,
booked, gross, cost, forgiveness and residual arrays, plus the accounting
identity error. `EmpiricalKernel.h`, `.one_action` and `.best_one_action` provide
the exact finite-kernel one-step reference. These interfaces implement the
stated terminal-account/IID model; they are not a black-box adapter for arbitrary
simulators, unbounded costs or policies that observe future disturbances.

For predictable gross payoff M_t and nonnegative transaction cost C_t,
X_t=M_t-C_t and E[M_t|past]=0. The same-policy booked return has expectation
minus expected discounted costs. With solvent initial cash and terminal
forgiveness F=gamma^(H-1)*(-L_H)+, subtracting the zero-mean gross-payoff
martingale from charged return yields Z=F-sum gamma^t C_t, without changing its
expectation. A mandatory first action supplies an exact strict negative upper
bound on expected booked return: minus that predetermined action's expected
discounted fee. This avoids a separate noisy booked-harm significance test.
Full charging changes only the reward, keeping ledger dynamics, actions and
disturbances fixed; every such same-action control has negative expected reward.

For bounded per-period costs, Z has the cost-only lower bound
`-4*fee*sum gamma^t`, even when price payoffs are unbounded with finite first
moment. SEARCH selects cap U from {32,128,512,2048} EUR, then independent
confirmation bounds E[min(Z,U)] below. Since min(Z,U)<=Z, a positive lower bound
certifies positive expected charged return. It can also erase a real witness.
Failure is abstention; it is never a safety or optimality certificate.

The confidence calculation uses the established one-sided
[Maurer–Pontil empirical-Bernstein inequality](https://www.cs.mcgill.ca/~colt2009/papers/012.pdf),
with sample variance denominator n-1 and fixed sample size. The six reported
events per selected policy are raw-return, residual, clipped-raw and
clipped-residual lower bounds, forgiveness lower bound and cost upper bound.
Bonferroni delta=.05/540 covers all 90 selected case/method pairs jointly.
Shared paths across methods do not affect the union bound. Opposite endpoints
returned by the generic bound function are marginal one-sided bounds, not
additional jointly guaranteed confidence intervals.

Clipping and split-based cap selection are established techniques, including
[Thomas, Theocharous and Ghavamzadeh (AAAI 2015), Theorem 1 and Remark 5](https://mohammadghavamzadeh.github.io/PUBLICATIONS/aaai15.pdf).
They are not presented as new algorithms here. The accounting identity supplies
the cost-only left bound and exact booked-harm reduction. Coefficient-one
martingale subtraction is **not** a universal variance reduction: a separate
exact two-step example has raw variance .29560675 and residual variance
.48862675 at equal mean .0985. The empirical ablation must be read directly.

## Initial benchmark results

Counts below use the same selected policy and confirmation paths for all four
representations. The raw bound uses a terminal-account telescoping bound,
rather than adding an unnecessarily large forgiveness maximum to its range.
The raw cap is chosen separately on SEARCH from the same cap grid, conditional
on the same selected policy, to distinguish clipping from accounting reduction.

| Method | Cases | Raw | Residual | Clipped raw | Clipped residual | New cases beyond isolated action |
|---|---:|---:|---:|---:|---:|---:|
| Max16 one action at deadline | 18 | 0 | 0 | 0 | 6 | 0 |
| Forced-prefix myopic | 18 | 1 | 2 | 0 | 6 | 0 |
| Fixed equity barrier | 18 | 0 | 2 | 0 | 6 | 0 |
| Structured equity/clock feedback | 18 | 0 | 2 | 0 | 4 | 0 |
| Random parameter search | 18 | 1 | 2 | 0 | 4 | 0 |

The exact full-empirical isolated-action reference, scanning **all** allowed
leverage magnitudes {0,.25,1,4,16}, is positive in 9 of 18 cases. Flat is included,
so a nonpositive best reference equals zero. The executed one-action policy
remains max16 at the deadline; both its exact value and the stronger all-level
reference are saved. The simple exact reference needs no Monte Carlo confidence
test conditional on this finite kernel and therefore finds opportunities that
the fixed-budget confirmation bounds miss.

The residual variance is below raw variance on all 90 selected empirical
policies; median ratios by method are .03746, .04329, .05285, .02853 and .03020
in table order. This observed result does not override the counterexample above.
Clipped raw loses the two certificates found by the corresponding raw methods.
Thus neither clipping nor accounting reduction is uniformly best.

At initial cash .1 and horizon 4, structured-policy residual sample means are
19.21 EUR (DK1) and 15.40 EUR (DK2), but its SEARCH-selected cap32 produces
negative clipped confirmation means and abstentions. The 512 search paths per
structured candidate were insufficient to choose a useful cap there. At states
without a positive all-level isolated-action reference, structured residual
means range from approximately -.017 to -8.81 EUR, with no certificates; this
is a discovery failure as well as an uncertainty limitation. All means and
lower bounds are saved rather than only the positive outcomes.

The same independently confirmed forgiveness lower and cost upper bounds imply
`J_alpha >= (1-alpha)*L_F-U_C` for this fixed policy under counterfactual partial
terminal forgiveness. A nonempty interval `alpha < 1-U_C/L_F` is a simultaneous
policy-specific harmful interval, without an alpha grid or extra alpha tests.
It is not a global boundary between safe and unsafe accounts; outside the
interval the method abstains.

## Latest-placement extension: a compiler, not better empirical discovery

The initial failure motivated a disclosed extension fixed in
`EXPLORATORY_EXTENSION_PLAN.md` before its endpoints were computed. Under the
stationary IID active-step kernel, with no information/cost while flat and no
early closure, a fixed T-step local controller can wait until H-T and then run
unchanged before deadline H. Couple identical active disturbances. Its ledger
and final deficit are unchanged, active costs gain the factor gamma^(H-T), and
the accounting residual weakly improves pathwise versus running early and then
waiting. Its expected charged return also weakly improves by the no-edge
identity; raw return need not improve on every path.

More strongly, the latest placed charged and booked returns equal
`gamma^(H-T)` times the local T-period returns **pathwise**. A local certificate
therefore scales to every H>=T with no new simulation or horizon test; the
compiled clipping cap scales by the same factor and the alpha interval is
unchanged. This is a guarantee for the same frozen local controller, not a
comparison against arbitrary full-horizon policies, information-seeking waiting,
market drift or early termination. Every comparator receives the same compiler.

The extension includes all positive registered N ledger probes
{.02,.05,.1,.25,.5,1,2}, in both Danish kernels: 14 states. All feedback methods
search T in {1,2,4,8,16}; the isolated-action baseline uses T1. Each method gets
at most 262,144 active search transitions per state and 32,768 independent
confirmation paths for its selected controller/duration. SEARCH now predicts
the bound at the prespecified confirmation sample size, rather than penalizing
caps using the much smaller per-candidate search sample. This classical cap
selection correction is applied equally to all methods. Original failures are
not replaced or pooled into the new confirmation sample.

| Method | States | Raw | Residual | Clipped raw | Clipped residual | New states beyond isolated action |
|---|---:|---:|---:|---:|---:|---:|
| Max16 one action | 14 | 3 | 3 | 4 | 6 | 0 |
| Forced-prefix myopic | 14 | 5 | 5 | 4 | 7 | 0 |
| Fixed equity barrier | 14 | 5 | 5 | 4 | 6 | 0 |
| Structured equity/clock feedback | 14 | 4 | 5 | 4 | 7 | 0 |
| Random parameter search | 14 | 4 | 4 | 4 | 7 | 0 |

The exact isolated-action reference is positive in 9/14 states. Structured
feedback again has **no new certified opportunity beyond that reference** and
ties random and forced-prefix-myopic search. Its seven clipped certificates are
DK1 cash fractions .02,.05,.1,.25 and DK2 .02,.05,.1. At cash .25 the structured
clipped lower bound is .962758 EUR in DK1 and -1.600485 EUR in DK2. This small
boundary difference and all abstentions remain in the output. No scalable
search superiority or learned-policy prediction is established.

Residual variance exceeds raw variance for 2/70 selected extension policies;
even within this empirical model subtraction is not universally beneficial.
All same-action full-charge repairs have strictly negative expected reward by
the exact fee identity. This is a reward repair certificate for the fixed
policies and law, not a new observation of how retrained agents would behave.

The extension's six one-sided events per state/method use delta=.05/420.
This supplies simultaneous coverage within the extension family. The initial
and extension families each use .05; no .95 joint-coverage claim is made over
both families or other exploratory experiments.

There are 420 calendar records over H={1,4,16,48,96,192}. They are deterministic
scalings or incompatible-duration abstentions, not 420 independent experiments.
Compatible records total 84/84 for one-action, 68/84 for forced myopic, 68/84
for barrier, 67/84 for structured and 64/84 for random search. The local bound
does not apply when H<T. The compiler represents 813,268,992 idle-inclusive
confirmation time slots using 18,579,456 actual active confirmation transitions;
search used 18,342,688 active transitions. The total 36,922,144 active transitions
excludes separate environment-validation checks. No additional simulation call
is charged for algebraic calendar placement.

The extension completed in 16.08 CPU seconds, 17.34 wall seconds, one numeric
thread. Forty additional delayed-placement charged/booked paths in the frozen
environment agreed within 4.55e-13 EUR; the accounting identity error was at most
3.64e-12 EUR. It reads source environments only for validation and changes no
registered artifacts or Study R files.

Extension artifacts are separate:

- `benchmark_latest_placement.py`: shared-compiler benchmark and source check.
- `exploratory_extension_certificates.csv`: all 70 selected local policies.
- `exploratory_extension_calendar.csv`: all compatible placements and explicit
  abstentions, including paired early/late sample means and exact expected
  improvement lower bounds.
- `exploratory_extension_candidates.csv`: every duration/controller/cap score.
- `exploratory_extension_summary.json` and `exploratory_extension_run.log`:
  selected controllers, fixed-family bounds, source hashes and compute.

## Final full-clock result and stopping point

The last extension was fixed in `EXPLORATORY_FULL_CLOCK_PLAN.md` after the two
earlier negative discovery results. It changes the active duration set to the
entire registered clock grid {1,4,16,48,96,192}, keeping all 14 solvent states,
all five methods, the same policy grammars and cap grid, the shared compiler,
262,144 search-transition ceiling per state/method and 32,768 independent
confirmation paths per selected policy. Fresh phase seeds are recorded in the
plan and script. No subsequent empirical extension was run.

| Method | States | Raw | Residual | Clipped raw | Clipped residual | New states beyond isolated action |
|---|---:|---:|---:|---:|---:|---:|
| Max16 one action | 14 | 3 | 3 | 3 | 6 | 0 |
| Forced-prefix myopic | 14 | 1 | 3 | 1 | 8 | 0 |
| Fixed equity barrier | 14 | 1 | 3 | 0 | 8 | 0 |
| Structured equity/clock feedback | 14 | 0 | 0 | 0 | 8 | 0 |
| Random parameter search | 14 | 0 | 2 | 1 | 8 | 0 |
| Total selected state/method pairs | 70 | 5 | 11 | 5 | 38 | 0 |

The all-level exact isolated-action reference is positive in 9/14 states.
Structured feedback's eight certificates are the four initial cash levels
.02,.05,.1,.25 in each Danish kernel. It ties all three non-one-action
comparators and supplies no empirical optimizer superiority. All four bound
representations use the SAME selected policy and paths, but that policy was
selected using the clipped-residual search criterion. This conditional ablation
is not a general comparison of optimally tuned estimators or search algorithms.

No certificate outside the isolated-action positive set does **not** establish
absence of a multi-step exploit. For example, the selected DK1 cash-.5/T96
structured policy has residual sample mean +12.909514 EUR but clipped lower
bound -39.494179 EUR, so it abstains. At DK2 cash .5/T48 the corresponding
mean is +2.011415 and lower bound -12.224099 EUR. Large-tail clipping and finite
search/confirmation can miss opportunities. Positive point estimates do not
replace the prespecified certificate. Every miss and negative result remains.

At equal search-budget ceilings the longer libraries have few search paths per
candidate: 734 for forced myopic, 183 for barriers and 22 for each structured or
random candidate, versus 262,144 for the single one-action candidate. Integer
allocation leaves some budget unused; actual work is reported below rather than
calling these exactly equal realized budgets. One-action's exact full-kernel
reference is stronger still and is kept visible.

| Method | Active search transitions | Active confirmation transitions |
|---|---:|---:|
| One action | 3,670,016 | 458,752 |
| Forced-prefix myopic | 3,668,532 | 36,339,712 |
| Barrier | 3,658,536 | 23,789,568 |
| Structured | 3,518,592 | 30,015,488 |
| Random | 3,518,592 | 33,161,216 |
| Total | 18,034,268 | 123,764,736 |

All 70 state/method pairs completed: 74.23 CPU seconds and 86.68 wall seconds,
one numeric thread, below the CPU guard. The benchmark used 141,799,004 active
transitions. An additional 2,344 vectorized validation transitions are included
in the instrumented 141,801,348 counter; separate frozen-environment validation
uses 8,448 `step` calls. Forty late-placement arm paths at H192 match the frozen
environment within 3.64e-12 EUR; paired accounting identities agree within
5.46e-12 EUR. No Study R or registered output is read or changed; frozen source
environments are read only for validation and their hashes are recorded.

The 420 calendar rows include compatible counts 84,46,53,46,45 in table method
order; remaining rows explicitly abstain because the selected duration exceeds
the horizon. Their 728,924,160 represented idle-inclusive confirmation slots
are deterministic reuses of local certificates, not measured extra simulations,
independent samples or a general-purpose speedup claim. The final family uses
.05/420 per allocated one-sided event. No paper-wide .95 coverage is claimed
across the three exploratory benchmark families.

Current final artifacts are `full_clock_benchmark.py`, `full_clock_summary.json`,
`full_clock_certificates.csv`, `full_clock_calendar.csv`,
`full_clock_candidates.csv` and `full_clock_run.log`. The executable includes a
per-step CPU guard and preserves partial completed rows if ever triggered.
The separate sibling `adaptive_witness_heavytail/` tests the cost-only
certificate's unbounded-price capability with known synthetic laws; its
positive certificate is not empirical search-success evidence for this task.

## Fresh selected-case certificate beyond isolated actions

After the full-clock result, a single follow-up was authorized and fixed in
`SELECTED_CASE_FOLLOWUP_PLAN.md` before any new draw. All 25 old selected
controllers at states without a positive exact isolated-action reference were
ranked by the declared projected untruncated-mean/noise score. Those old results
were development data for this selection, not the new confirmation sample.
Every eligible record and its frozen policy is saved in
`selected_case_selection.csv`, including unsuccessful alternatives.

The winner is DK1, initial cash 10,000 EUR, active duration 96. It takes mandatory
leverage 16 at time 0. During the first half it takes 16 when observed booked
ledger is at most 10,000 EUR, otherwise flat; in the second half the threshold
is 20,000 EUR. This policy can stop after an early gain and resume at the clock
switch. Its initial action is allowed despite negative isolated value; it is
not restricted to trading after the cash floor has already been hit.

Fresh SEARCH uses 8192 paths, seed 3401000, and freezes a cap chosen from
{2048,8192,32768,131072} EUR using the projected confirmation bound. It selects
32768. Independent CONFIRM uses seed 3501000 and exactly 524288 whole paths in 64
batches of 8192. Only ONE lower-confidence event is allocated delta .01; no
sample increase, new cap, case replacement or policy adjustment follows its
result.

| Quantity | Result |
|---|---:|
| Capped residual confirmation mean | 15.0645825612 EUR |
| One-sided 99% radius | 3.9716569027 EUR |
| One-sided 99% charged-return lower bound | **11.0929256585 EUR** |
| Exact expected booked-return upper bound | **-3.3355318258 EUR** |
| Best nonflat isolated action, all allowed levels, at deadline | -0.0143326816 EUR |
| Best isolated action including flat | 0 EUR |

The negative isolated-action value is exact conditional on the finite empirical
kernel. A fixed isolated action at an earlier time is weakly worse under the
same nonnegative fees, so no state/clock-only isolated-action timing/magnitude
mixture supplies a positive witness here. The independently confirmed 96-step
feedback policy does. This establishes multi-step incentive capability beyond
the isolated-action audit for this selected empirical-model case. It does not
show that adaptive search beats random search, constant multi-step exposure or
another optimizer: those comparisons were not independently retested at this
larger budget.

The cost-only lower endpoint is -247.5811527582 EUR, and the cap is frozen before
confirmation. By the accounting identity, the lower bound on E[min(Z,U)] also
bounds expected charged reward. Booked harm follows exactly from the mandatory
first action's known positive expected fee. Same-action full charging is thus
a negative control without a second noisy test. The raw-return sample mean
2.35060279 EUR, residual mean15.06458256 and cost mean102.33452404 are descriptive;
the only allocated new inference is the stated one-sided lower bound. It is not
a two-sided 99% interval or a paper-wide coverage guarantee.

This conditional 99% statement remains valid despite post-selection because
the final controller and cap are fixed independently of its confirmation paths.
It concerns the stated empirical IID no-edge law, not future-market coverage or
policy acquisition. The original 38/70 final comparison and all prior failures
remain unchanged. The latest-placement compiler scales this local lower bound
by gamma^(H-96) at every H>=96, without another statistical test.

All 524288 planned confirmation paths completed in 6.125 CPU seconds and 7.27 wall
seconds, one thread. Search plus confirmation use 51,118,080 active transitions;
192 vectorized validation transitions bring the recorded count to51,118,272.
Four selected-policy charged/booked frozen-environment path checks agree within
2.27e-13 EUR; the paired accounting identity error is at most 5.46e-12 EUR.
There was no subsequent adaptation or extension.

Reproduce with `selected_case_followup.py`; the plan, complete 25-row selection table,
all four cap scores, fixed policy, independent phase seeds, bound, compute and
source hashes are retained in `selected_case_followup.json` and its log.

## Validation, provenance and files

Deterministic feedback paths were replayed against the unchanged registered
`LagEnv`: 24 paths and 48 charged/booked arm checks, maximum discounted-return
error 3.64e-12 EUR. The paired accounting identity error over all simulations
was at most 5.46e-12 EUR. Each method used 6,684,672 logical search transitions
and 13,369,344 confirmation transitions. Across all methods this is 100,270,080
logical transitions. Full empirical-row preprocessing is additional arithmetic,
not hidden simulation calls. The initial execution took 18.5 CPU seconds and
22.83 wall seconds on one numeric thread; final metadata corrections retain
the same policies, seeds, paths and sample sizes.

`EXPLORATORY_PLAN.md` predates the first endpoint calculation. A post-run
independent-review correction strengthens the exact isolated-action reference
from max16 to all allowed magnitudes and renames forced-prefix myopic. It changes
no actions, random draws, certificate or discovery count. All initial scripts,
plans, outputs and logs remain in `initial_max16_reference/`.

- `exploratory_empirical_certificates.csv`: all 90 cases, selected caps,
  exact references, means, variances, bounds, abstentions and call budgets.
- `exploratory_search_candidates.csv`: every candidate and cap, including
  unsuccessful policies and heuristic search scores.
- `exploratory_empirical_summary.json`: selected policy definitions,
  certificates, environment validation, source hashes and compute accounting.
- `exploratory_run.log`: recorded execution output.

Run from the code repository root:

```powershell
python exploratory_novelty_2026-10-15\adaptive_witness_empirical\benchmark_adaptive_witness.py
python exploratory_novelty_2026-10-15\adaptive_witness_empirical\benchmark_latest_placement.py
python exploratory_novelty_2026-10-15\adaptive_witness_empirical\full_clock_benchmark.py
python exploratory_novelty_2026-10-15\adaptive_witness_empirical\selected_case_followup.py
```
