# Frozen learned-policy audit: authentic harder-simulator adapter

Status, 16 October 2026: the original internally hash-frozen native experiment
is complete and independently verified. All six methods abstain in all 40 cases
at both charges; the auxiliary new-claim component also gives 0/40 throughout.
All 12,800 paths have zero actual and paired financial forgiveness, and all 160
learned choices have beta=budget=0. There is no exercised learned-control benefit.
All 40 cost lower bounds are positive. Primary private verification passes
601,179 checks; auxiliary verification passes 324,195; the standalone public
replays pass 99,690 and 66,887 checks. Full artifacts are in `results/run_001/`
and `results/run_001_new_contract/`. Original source freezes and negative results
remain unchanged. The separate `native_joint_fault_v2/` study launched at
13:15:22UTC from source freeze
`17d3afc43611da9e95c4fdc58c8514abaddef7bdde0b5967f3f0a09251a7b498`.
It has6,400 paths/340 limits, including cash RAW, local-only and learned cash
controls. Final outcomes remain unavailable until complete confirmation and
verification. It follows the original null result and cannot replace it.

## Scientific scope and comparison

The target is the forward financial-event component under fresh hidden fair
settlement signs, not total NAV, physical/battery profit or the native shaped
training reward. All 20 saved plain/feasible MAPPO actors retain their 12D/20D
interfaces and matching battery DQN. Each is crossed with two explicit distressed
profiles, yielding 40 cases rather than 40 newly trained agents. No RL policy is
trained in this extension; Ridge and a two-layer 64-unit MLP are audit controls.

Per case, FIT 32 / TUNE 32 / CONFIRM 256 gives 12,800 audit paths and 1,228,800 audit steps;
repeated eight-context setup adds 245,760 warm steps. Six methods share paths:
RAW, capped RAW, capped realized-injection residual, local sign pairing, Ridge
control and MLP control. All 520 primary limits share a 95% family allocation.
The separate auxiliary plan excludes direct old-claim maturity G/I/A while
retaining inherited state, all forward financial costs and the original controls
and choices. It uses 256 CONFIRM paths for each of 240 limits, with a separate 95%
family; combined coverage is at least 90%, not a joint 95% guarantee.

The realized residual already avoids conditional-magnitude integration and
supplies the cost-only lower endpoint. Pairing can tighten local upper support;
neither pairing nor learning guarantees lower whole-path variance. Generic
branch controls and lower-confidence inference are established prior art. See
`native_validation/ADVERSARIAL_NOVELTY_ASSESSMENT.md` and
`validation/NATIVE_EPISODE_STATE_REVIEW.md` for the novelty and state audits.
The practical question is whether learned controls beat the strongest simple
baseline after their additional fit/prediction cost. Ties and abstentions count.

Chronology caveat: some preserved native risk features use full-scenario revenue
aggregates that enter native risk controls. Financial price
normalization is primed on 2024 history before the 2025 contexts, but the whole
interface is not a leak-free causal backtest. Fresh hidden settlement signs are
independent of these fixed context features; the conditional simulator audit
remains distinct from a future-market/deployment claim. See
`INTERPRETATION_NOTE.md`; frozen engine/experiment bytes are unchanged.

After complete primary/auxiliary verification, separate standard-library public
bundles reproduce all confidence arithmetic without an engine, actor call or
model fit. `POST_RUN_REVIEW_CHECKLIST.md` lists the completion gates. These
exports cannot validate native event instrumentation or make external inputs
publicly available. Earlier exact-kernel/partial-repair conclusions retain their
stronger uniform conditional-moment assumptions.

`PUBLIC_REPRODUCIBILITY.md` and `public_reproducibility_manifest.json` document
the external engine/checkpoint/data dependencies. This public audit code is not
yet a self-contained package for the authentic native experiment; authorized
external inputs and a clean-install reproduction remain necessary.

`REPRODUCTION_ACCESS_PLAN.md` refines the conservative30-file dependency inventory:
the current runtime reads only ten CSVs (53,910,563bytes), including just
history019/020. None is an eSett input. The authorized complete local review
packet is `review_bundle_private/native_review_complete_v1.zip`, kept gitignored
and private. It contains exact native/auditor sources, all20paired investor/battery
checkpoints and configs, the ten runtime inputs, optional episode20ANN bank and
reconstruction/provenance notes. All155archive members were hash-checked;
the archive is53,564,447bytes, SHA-256
`d96cf47f5d6746cf760608293bfbe5e62a5720f2b98c30702a89debf5f6b8299`.
Packet-only replay passed for two fixed software paths (960 native steps), with
all 30 arrays per path exactly equal and original-native access forbidden.
Three executed attempts used 2,880 native steps in total; two earlier attempts
passed numerical replay but failed the final I/O allowlist gate. This uses the
existing dependency environment, not a clean installation or all-case rerun.
See `native_validation/PACKET_PORTABILITY_VALIDATION.md`.
The packet preserves original metadata
paths and records the remaining artifact-license/attribution and anonymity work.

## Available genuine learned policies

`inventory_checkpoints.py` writes `checkpoint_inventory.json` using only zip JSON
metadata, file metadata/hashes and installed-package metadata. It never executes
checkpoint objects or normalization pickles. The inventory covers every available
Study L20/R60 checkpoint and the three existing Prototype5 final investor families.

Study L's twenty policies were trained with one observed feature, clipped reported
equity divided by EUR20,000, and nine signed leverage actions. The runner uses
`norm_obs=False`; its saved VecNormalize statistics concern reward normalization,
not actor input. Study R's sixty policies have the same 1D observation/action
contract, with different registered fees/charges and separately saved normalizers.
Adding context or multiple-asset coordinates to their observations would change
the trained interface. They can audit transfer under changed cashflows, but do
not establish that a learner uses rich observed context.

Actual richer checkpoints exist in the local `Paper1/Prototype5` tree despite not
being redistributed in the code repository. The authentic plain MAPPO investor
has a 12D local actor, a centralized training critic, continuous scalar exposure
action and frozen `[256,128,64]` architecture. The actor observes price momentum,
volatility, cash, current exposure, cumulative P&L, decision cadence, capital/risk
limits, drawdown, recent P&L, medium momentum and time of day. The feasible-action
arm adds eight trained forecast/constraint features, for 20D. The checkpoints use
the native custom Beta policy classes; actor inference must preserve their action
mapping. Existing rolling-past normalization history and evaluation/settlement/
liquidity data are present. This is temporal context with an endogenous financial
state, not an IID scalar-payoff model with cosmetic extra features.

Important scope limit: all three families use the engine's **one-factor investor
sleeve**. Wind/solar/hydro are component book entries allocated by the execution
logic; the actor does not choose three independent exposures. This is not an
independent multiasset learned portfolio experiment.

## Native engine mechanisms to preserve

The default training/evaluation contract has six engine steps per investor decision
and six-step maturity contracts, actual MWh sizing, a 25% participation cap based
on observed activated volume, bid-ask costs, volume-dependent nonlinear impact,
annual market access costs, collateral funding, margin/solvency checks, rule-based
risk/meta controllers, and a separate learned battery controller. The actor's
native observation builder and rolling-past normalizer remain unchanged; the
full-scenario risk-feature caveat above applies.
The inventory retains the exact frozen contracts and source hashes.

Relevant source locations in the authentic engine:

- `environment.py:2547`: fixed-horizon action mapping occurs before maturity cash.
- `environment.py:4241`: pure roll-cost estimator; `:4309` charges roll costs.
- `environment.py:4335`: annual access fee; `:2714` collateral funding.
- `environment.py:4458`: mature financial contract settlement and cash floor.
- `environment.py:4716`: payoff dispatch; `:4732` finance/battery/physical cashflows.
- `environment.py:6465` and `observation_builder.py:90`: native observation interface.
- `environment.py:6930`: forecast calibration also calls the unit-payoff helper.
- `evaluation.py:1616`: frozen configuration hydration; `:1915` rolling-past history.

The original complete campaign reports roughly 10--35 minutes per full evaluation
job. Short independent episodes and inference throughput need a software-only
validation before selecting confirmation budgets. Native actor loading and two
bounded deterministic trace checks are now complete; their timing is recorded
below.

## Proposed placebo instrumentation, before any endpoint choice

Use the native engine and frozen actor, not a reimplemented feature vector. At
each actual maturity event, **after the new action has been mapped**, multiply the
joint mature financial gross payoff by a fresh independent fair sign. Keep the
real chronological context, contract queue, native costs/caps and physical state.
One common fresh sign across the mature components preserves their joint
symmetry. It is hidden from all actors until the settlement is booked. Fresh
independent signs make this audited financial payoff conditionally mean zero
given all actor information, despite arbitrary temporal magnitude/context laws.

Do not globally override `_horizon_unit_payoff`: forecast calibration also queries
that function. The adapter wraps only the actual settlement call and restores
the helper immediately, with a hard assertion that the hidden sign is accessed
inside the post-action maturity scope. No future rollout branching is
needed to form the local positive/negative cashflow pair. Forecast observations
remain native features; they do not reveal the fresh placebo sign. Their
information timing must be interpreted with the full-scenario caveat above.

## Agreed component estimand and broader alternative

Native cash receives battery arbitrage as well as financial trading; generation
revenue contributes to fund NAV separately. Native RL rewards include transformed
return and risk/drawdown penalties, and are **not** the monetary charged objective
in the certificate theorem. Applying `E booked = -E costs` to total fund NAV or the
native shaped reward would be invalid.

The adapter implements route 1 below. Route 2 remains an alternative requiring
different inference; it has not been implemented as an endpoint experiment.

1. **Financial-sleeve component audit:** retain the entire native physical/battery
   evolution and actor observations, but record the separate financial gross,
   execution/access/collateral costs and cash-floor injections caused at those
   exact financial events. The financial booked contribution has conditional
   expectation minus its recorded costs. This is explicitly a component audit,
   with other cashflows affecting the observed cash before the event. Its
   accounting identity must use event-level ordering, not silently treat total
   engine cash as a separately reflected financial ledger.
2. **Total native cash/fund audit:** retain every cashflow and estimate both
   charged monetary and booked total expectations with independently allocated
   bounds from known finite empirical/physical support. No analytic negative-book
   shortcut applies, because physical production and battery arbitrage can earn
   positive expected profit. An inconclusive or non-harmful result must remain.

The first route isolates the proposed method under genuine endogenous context and
real frozen actors. The second answers a broader question but may need much more
sampling and does not inherit the no-edge booked-harm implication. A new independent
multiasset policy would require explicit new training with the correct observed
context/action contract, rather than appending features ignored by old weights.

## Sample-only method boundary

With a fair hidden settlement sign, the realized local sign pair can be evaluated
from logged gross magnitude, sign-invariant cost and pre-event cash/ledger without
integrating over an empirical row distribution. Any clipping is fixed on discovery
data and confirmation uses independent whole episodes. A learned approximation to
conditional drift alone is not an unbiased compensator. Its error requires a
valid bound or exact martingale correction, not an unsupported plug-in claim.
Theory and numerical endpoint plans are being coordinated separately; this file
does not freeze or select a positive audit condition.

## Implemented API and software checks

`native_engine_adapter.py` supplies `NativeRuntime`, immutable `EpisodeSpec`,
`NativeScenario` and fresh `NativeEpisode` objects. The runtime loads the authentic
native custom actor and battery policies. The endpoint API supports a prespecified
384-step native original-sign burn-in in each policy/profile/context, followed by
a 96-step audit window. The unhooked native state, including forecast calibration,
risk, physical, contract-queue, normalization and RNG buffers, is cached. Each
audit episode deep-copies the entire warmed state rather than reconstructing only
cash or calling native `reset()`, which can preserve continuing financial gains.
The non-economic memory manager alone is recreated with a fresh thread lock.
All other native buffers have matching recursive hashes before the initial cash
intervention, and the cached parent remains unchanged after the audit.

Public API: `EpisodeSpec(checkpoint_dir, zone, start_row, horizon=96,
native_seed=7, profile='strict'/'vulnerable', cash_fraction=..., burn_in=384)`;
`NativeScenario(runtime, spec).new_episode(sign_sequence96)`. The convenience
function `prepare_scenario(..., H=96, burn_in=384, ...)` has the same behavior.
`fixed_context_grid(runtime)` returns both original/v2 contexts at starts
0, 12,944, 25,888 and 38,832 for the existing 39,312-row archives. The financial
sign stream begins only after burn-in. Trace `step` is relative audit time0--95;
`native_step` retains calendar positions 384--479 inside each 480-row window.

Strict restores the saved guards. Vulnerable disables sleeve margin and sets the
MTM loss-exit threshold to positive infinity, serialized as `disabled_inf`;
zero would make exits more aggressive. Both retain all other mechanisms and
historical halt flags. Initial cash is a specified fraction of the ORIGINAL
initial trading allocation, while notional sizing retains that original base.
The two warmed software paths used fraction .1. The later source-based endpoint
proposal changes this to `1e-4`, because the original sleeve is roughly 662 million
DKK while native capped contract volumes are much smaller. No software financial
outcome is used to select that fraction; root owns its final plan/contract check.

Pending prepaid contracts at the audit start remain part of the declared initial
state. Their prior costs are outside the audit window and future financial gross
is fair-sign shocked. Native margin checking occurs after maturity cash, so the
strict profile is a guard-restored comparison, not a proof of zero injections or
zero certificates. Do not interpret a positive accounting contribution from
initial queued claims as proof of newly chosen risk after the cash intervention.

`episode.step()` performs genuine deterministic saved-policy inference for every
native actor and uses the native risk/meta rules. `episode.trace_arrays()` returns
NPZ-ready event and path arrays. Every financial event includes engine step, kind,
pre/post cash, actual gross, nonnegative cost, exact injection, fresh sign and the
original aggregate gross before that sign, plus full investor observation/action.
The name `unsigned_base_gross` means before multiplying by the fresh sign; that
original aggregate can itself be signed. Settlement is one aggregate event,
matching the single native cash floor after mature contract aggregation.
Roll, access and collateral costs are separate native floor events. Battery,
distribution and margin-conversion calls and the residual native cash change are
retained as descriptive exclusions. Native shaped rewards and fund NAV are also
descriptive, never substituted for the component's monetary objective.

`NATIVE_SOFTWARE_VALIDATION_PLAN.md` fixes exactly two deterministic 96-step paths,
plain7 and feasible7, before execution. The first attempt passes all checks;
the trace NPZ files, source/plan snapshots and report are under
`native_validation/attempt_1/`. There are no confidence bounds, charge searches,
learned incentive labels or extra independent observations. Native actor shapes
12/20, battery shape12 and every action space match their saved checkpoints;
event cash/injection reconciliation, post-action maturity timing, helper
restoration, deterministic support checks, config isolation and native source
hashes pass. Both final contract queues are empty.

Measured steady rollout time including actor inference is .004686 seconds per
native step for plain7 and .007542 for feasible7. Fresh environment construction
takes about .272 seconds in each case. One-time imports take 32.435 seconds;
first scenario preparation takes 10.274 seconds and warm preparation .397 seconds.
The two-path run uses 35.203 CPU and 44.925 wall seconds. These measurements are
software budget evidence, not a statistical performance or incentive result.

`NATIVE_WARM_CLONE_VALIDATION_PLAN.md` specifies two additional fixed warmed checks,
plain7/strict and feasible7/vulnerable. The first warm attempt passes the plain
path but its generic buffer fingerprint lacks slot-based-object support for the
native forecast processor. `warm_attempt_1/` preserves that source, plan and error.
The utility is extended to hash every slot buffer; `warm_attempt_2/` replays the
same fixed paths and passes. No dynamics, policies or statistical parameters are
changed in response, and replay adds no independent endpoint observations.

The completed warm check uses 33.953 CPU and 37.793 wall seconds including imports.
384 native burn-in steps take 2.112/2.457 seconds, deep-clone and buffer validation
take .0491/.0541 seconds, and audit steps including actor inference take
.005672/.006212 seconds for plain/feasible, respectively. Worker RSS is 567/575 MB
and peak working set 592/602 MB. Both queues end empty; native/helper/source,
actor-contract and parent-immutability checks pass. Initial input files include
the actual primed history files, aligned settlement/liquidity files, native
forecast source/view and both actor/battery checkpoint hashes in the metadata.

After the two-path check, a conservative support guard was strengthened without
new paths: the fixed-initial sizing bound uses allocation upper bound1, so it does
not confuse the engine class's .75 with the saved config's .8 maximum. It is
tightened by the native current/causal liquidity helpers only on strictly positive
capacity branches; a zero-cap branch retains the fixed sizing bound. The original
validation source snapshot and traces remain unchanged. A final-step empty-queue
assertion now also applies in every future endpoint run.

Support constants use configured hard sizing/capacity and the entire declared
finite context, never the largest value observed in confirmation samples. They
are not exact conditional payoff moments. Preparation and native forecast/risk
arithmetic remain extra computational work, even when simulator paths are shared
across estimator comparisons. A later statistical protocol must freeze its own
policy/profile list, case law, split streams, query budgets and error allocations.

## Calendar-support verification before the statistical freeze

`native_support.py` supplies source-derived per-calendar envelopes, separately
from the passed adapter's broad guards. Possible openings are limited to the
native six-step clock and finite maturity window; the bound includes initial
prepaid claims, current-price entries, configured settlement clamps, native
current/causal capacity helpers and their zero-cap sizing fallback. Roll fees
are bounded only on possible opening steps, access once each step, and funding
on outstanding contracts after current maturity. No conditional payoff moment
or observed trajectory maximum enters these bounds. The analytic unavoidable
access-cost lower bound uses the raw deterministic fee and downward rounding.

`validate_native_support.py` creates only
`native_validation/calendar_support_attempt_1/`. Its preserved report is PASS:
both immutable `warm_attempt_2` traces satisfy each cost-kind/calendar envelope,
the aggregate maturity/gross envelope, whole-path discounted bounds and empty
terminal queues. It reconstructs only the same two deterministic 384-step native
warm prefixes, then checks fresh initial states at cash fraction `1e-4` in the
plain/strict and feasible/vulnerable interfaces. **No new audit steps, financial
endpoints, confidence limits or independent observations are generated.** The
saved traces retain their original `.1` cash fraction; the support is independent
of the intervened cash balance. Total verification time is 112.032 wall and
83.734 CPU seconds on this machine. The passed adapter SHA-256 remains
`da93030da39c1f2a4d0eaef3666177fc6628b7da97161ed892d1b9abfa38c879`.

The phase worker prepares and checks all eight predetermined contexts before
audit sampling and checks the actual path against its context's calendar bounds.
Separate development and confirmation jobs each rebuild their warm snapshots:
40 cases therefore require 245,760 deterministic warm steps in addition to
1,228,800 audit steps under the frozen 32/32/256 episode budgets. This repeated
setup is computational work, not extra independent statistical observations.

The deterministic forecast-cache preparation report under
`native_validation/forecast_cache_prep_attempt_1/` also passes: all eight complete
480-row views match exact native source-derived CSV bytes, with zero actor calls,
native/audit steps or random draws. Missing views are atomically published before
forecast-enabled parallel workers start; an existing mismatched view is never
silently replaced. Frozen numerical sources are unchanged. Read-only runtime
provenance reports record completed-phase support and source/resource consistency
without inspecting financial outcomes. The earlier detached startup created no
LAUNCH record or queries; the managed execution is the actual experiment launch.
