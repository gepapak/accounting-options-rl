# Native adapter software validation, fixed before execution

This is a software/interface check, not a statistical experiment or hypothesis
test. Exactly two deterministic 96-step paths are specified: plain MAPPO seed7
and feasible-action MAPPO seed7, both on the first 96 rows of the original native
evaluation context. The plain path uses signs +1 at all engine times; the feasible
path alternates +1/-1 by engine-step parity. All native seeds remain7. No additional
policy is selected from numerical results. No learners, cap fitting, confidence
bounds, charge search, or endpoint calculations are run.

The paths check actor/action space identity, original checkpoint inference,
event-level financial cash reconciliation, financial cost/support inequalities,
fresh config isolation, native contract maturity ordering, final queue clearance,
and restoration of the unit-payoff helper outside actual maturity calls. Full
observations/actions, native shaped rewards and excluded battery/margin/distribution
transfers are retained for software provenance. Native monetary sums are only trace
reconciliation values, not learned-policy incentive conclusions.

The authentic native sources, checkpoints and datasets remain read-only. Bytecode
and caches/logs are disabled or redirected into this new audit directory. Each
attempt snapshots its adapter/runner bytes and writes under `native_validation/`.
Software failures and deterministic same-path replays are retained. A replay is
not another independent observation. A later endpoint experiment needs its own
frozen policy/context list, streams, budgets, estimand, comparators and error
allocation; these two software traces must not be treated as held-out endpoints.

Runtime is measured as process CPU and wall time. Environment construction,
actor inference, simulation and conditional-feature/oracle arithmetic are disclosed
separately where available. The measured timing determines feasible budgets; it
must not select positive cases or alter the native policy interface.
