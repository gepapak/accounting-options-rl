# Reproducing the authentic frozen-policy audit

The public audit source alone cannot currently reproduce the complete native
experiment. It loads an external `Prototype5` engine, trained actors, matching
battery controllers, chronological physical/price inputs, settlement/liquidity
panels, normalization histories and forecast features. These dependencies are
locally available for this experiment but are outside the code repository.
Their hashes establish identity; they do not establish access or redistribution
rights. This is a packaging/access gap, not a claim of a self-contained public
benchmark.

`public_reproducibility_manifest.json` records filenames, roles, sizes, package
versions and SHA-256 values without copying external source or inputs. Its
metadata-only builder, `build_public_reproducibility_manifest.py`, loads no
policies, performs no simulation and reads no financial outcomes. It binds the
audit's source freeze to 34 native Python modules, 60 actor/battery/config
artifacts for the 20 studied policies, and 30 context/history/forecast files.
It also provides an advisory static import closure of 21 native modules. Dynamic
imports and custom-policy deserialization mean that this static closure is not
a tested standalone minimum. The full frozen native module set is the source
bundle used in the validated environment.

## Exact dependencies and interface

Preserve this directory layout, currently assumed by the adapter:

```text
<Paper1>/
  github/accounting-options-rl/
    exploratory_novelty_2026-10-16/sample_only_learned_audit/
  Prototype5/
    <native modules and external dependencies listed in the manifest>
```

The 20 policies comprise all ten frozen plain MAPPO and all ten frozen
feasible-action MAPPO seeds. Each needs `investor_0_policy.zip`, the matching
`battery_operator_0_policy.zip`, and `training_config.json` from its listed
`final_models` directory. The plain actor receives its original 12D observation;
the feasible actor receives its original 20D observation. Both replay the
matching 12D battery controller. The custom saved policy classes, native action
mapping, rolling-past normalization and historical/forecast calibration must
remain available. Replacing these actors with the simpler 1D Study L/R policies
would change this experiment.

The conservative manifest above inventories 30 potential context/history/forecast
files. The exact runtime minimum is ten CSVs: original/v2 chronological data,
aligned settlement and liquidity panels, newest normalization histories 019/020,
and two native forecast files. The audit uses eight fixed 480-row windows and
the native history loader. `REPRODUCTION_ACCESS_PLAN.md` records this reduction.
The complete LOCAL PRIVATE v1 packet contains all ten inputs, exact34native
modules,20paired actor/battery checkpoints and20configs, with hashes checked
inside its155-file archive. No eSett inputs are included. This closes local
review input access without claiming public release or a clean installation.
Original metadata paths, licensing/attribution and anonymous-delivery work are
explicit release tasks. Public confidence-arithmetic exports omit raw market
and actor-feature rows and cannot replace full engine replay.

Package versions and Python version are taken from the pre-execution inventory.
They document the tested local environment rather than supplying a validated
clean-install lockfile. A fresh installation must also preserve the native
module namespace required to deserialize the saved custom policies. A separate
clean installation has not been tested. A narrower packet-only replay has passed
in the existing dependency environment: two fixed 384+96-step software paths,
30 arrays per path exactly equal, and no access to the original native tree.
This is not an independent full 40-case rerun. All attempts and their work are
recorded in `native_validation/PACKET_PORTABILITY_VALIDATION.md`.

## Execution and verification

The outcome-blind plan is `EXPLORATORY_EXECUTION_PLAN.md`; the runner performs
source/input freezing, all FIT/TUNE development, pooled FIT-only nuisance fitting,
per-case tuning, selection freezing, independent confirmation and aggregation.
For an independent authorized reproduction, choose a fresh run identifier:

These commands require all eight complete forecast views before concurrent
feasible-action workers start. The historical preparation helper remains bound
to reference `run_001`; preserve it. The new unfrozen
`prepare_packet_forecast_cache.py` accepts explicit packet/source/freeze/cache
and provenance roots, verifies every source/model/input hash, and atomically
prepares or checks exact 480-row views with zero actors or native episodes.
`PORTABLE_CACHE_PREPARATION.md` documents compatible-layout mode: OUTPUT equals
the packet's audit directory, writes are restricted to its new forecast-cache
subdirectory, and all frozen source hashes must remain unchanged. The frozen
runner then finds those views using its normal HERE/ENGINE layout. External
OUTPUT is supported for the separate replay wrapper with an explicit in-memory
HERE override. Preparation completed for all eight views with zero native or
actor queries, and the compatible-layout packet replay passed. These results
do not establish a clean-install/full-run reproduction.

```powershell
python run_native_audit.py freeze --run-id run_reproduction
python run_native_audit.py run --run-id run_reproduction
```

Do not overwrite or resume a partial/completed reference run. A technical repair
requires the original failed artifacts, an explicit repair record and a separate
exact-stream attempt; it does not permit an outcome-dependent change of budgets
or choices. Three single-threaded workers and the fixed 32/32/256 episode split
are specified by the plan. The total is 1,228,800 audit steps plus 245,760 repeated
deterministic warm steps; warm-up work is not independent audit evidence.

Native forecast views are deterministic 480-row cache slices. Before concurrent
forecast-enabled workers start, the separate engineering helper
`prematerialize_native_forecast_views.py` validates existing complete views or
atomically publishes missing ones. Its preserved preparation report proves
zero actor queries, native steps, audit draws or financial endpoints. Its input
and view hashes should agree with phase provenance. This avoids a concurrent
partial-file read without changing the frozen numerical source or experiment.

`validate_runtime_provenance.py` is a read-only guard over completed phase
metadata, source/input hashes, all eight context supports, fixed query budgets
and cross-phase resource/config identities. It does not load financial outcome
rows. The independent numerical verifier is separate. Public aggregate case and
method results can support rechecking confidence-bound arithmetic once complete;
they cannot replace the private-input full-pipeline re-execution or validate the
instrumented simulator's no-edge assumption on their own.

The financial-event contribution excludes physical/battery profits and native
shaped reward, retains prepaid initial claims, and intervenes with fresh hidden
fair settlement signs and thin initial cash. The separate auxiliary target
removes only direct old-maturity returns/injections and keeps inherited state.
Native risk features use full-scenario revenue aggregates, so the preserved
interface is not a wholly causal market backtest. Price histories precede the
audit contexts and forecasts use historical settlement windows; these separate
facts do not make every native feature causal. Fresh hidden signs remain
independent of fixed context inputs. The certificate concerns this declared
simulator stress law, without ordinary-deployment profitability, future-market
safety or arbitrary black-box-access claims. See `validation/NATIVE_CHRONOLOGY_REVIEW.md`.
