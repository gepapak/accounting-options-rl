# Read-only interface and statistical review

16 October2026. No inference, simulation, random draws or training were executed.
This review reuses the engine owner's metadata inventory at
`../checkpoint_inventory.json`; it does not create a competing checkpoint list.

Study L has20 real saved PPO policies under
`revision_experiments/2026-10-10/continuation_experiment/jobs/`. Its frozen
`run_continuation.py` observes only clip(cash/20000,0,5) at line33, uses
norm_obs=False and norm_reward=True at line57, and saves actor and normalization
at line60. A probe JSON records ABSOLUTE leverage only. Native signed action
evaluation must use the saved actor, not absolute probe interpolation.

Study R has60 real PPO policies under
`ACCOUNTING_OPTIONS_2026-10-14/study_r/results_r/models/`. `run_r.py`102-105
defines the same1D equity view; lines111-119 fix reward-only normalization and
verify signed saved-actor actions. Lines125-134 save ZIP, normalizer and safe
numeric actor layers; the job JSON binds their hashes. Three fees/charge-related
settings in source must not be inferred from performance: native training fees
are.4/1.5 and alpha training cohorts0/.8/1. All are frozen policies; none is
guaranteed flat or harmful at an arbitrary new state. These eighty actors are
useful portable learned-policy inputs, but do not satisfy a rich-trained20D
controller claim.

Authentic richer actors are present in Prototype5: plain MAPPO10 has12D local
observations; feasible-action MAPPO10 has20D. Both have1D continuous[-1,1] action
spaces, custom `policy.CentralizedCriticBetaActorCriticPolicy`, actor architectures
[256,128,64], recorded gamma.995 and526032 training steps. A48D centralized
critic is separate from the local actor input. Each has a paired battery,
training config, native runtime contract and normalization metadata. The
plain cohort disables forecast utilization; the feasible cohort enables it.
The source/contract differences require within-policy estimator comparisons;
the cohorts are not interchangeable adapters or a controlled causal comparison.

The cached plain seed42 contract illustrates the native complications: six-step
settlement, delta-exposure mapping(scale.2), impact and liquidity constraints,
collateral funding, margin liquidation/disable, rolling-past normalization and
annual market-access fee160175DKK. `engine/environment.py`818-832 mutates carried
price-normalization state on the config; reset1842ff may continue time, and
2365-2368 increments persistent training_global_step. Independent whole episodes
need fresh config/state snapshots and explicit causal history. Seed resets alone
do not establish IID episodes.

The existing `audit/setup_audit_engine.py` provides a copy-and-patch construction
workflow and explicit protocol-mismatch recording, not a statistical guarantee.
It was READ ONLY here. Source compatibility requires preserving the real runtime
contract or naming every deliberate intervention. Full-engine rewards include
physical assets, battery flows, shaped terms and unavoidable fees; the simple
no-edge financial expected-cost identity cannot be asserted for that aggregate.
The primary protocol therefore targets ordered native financial-event cashflows.

Before any endpoint draw the independent validation checklist is:

- Verify all20 actor/battery/config hashes and actual native local observation
  dimensions; deterministic inference stays frozen.
- Prove new hidden fair signs are unavailable at action time, including through
  cache/forecast/settlement features, and preserve native queue/cost timing.
- Check each financial gross and cost event separately against native cash floor
  changes; summing G-C before one final clamp is generally a different account.
- Isolate independent engine/config/normalizer/queue states across episodes and
  ensure all pending positions and liabilities are settled or valued at the end.
- Count every fitted-model sample, environment clone, policy/critic inference and
  failed query; freeze actual sample budgets only after outcome-blind runtime
  validation.
- Establish true deterministic supports and strict booked harm, or allocate
  direct cost/booked bounds and allow explicit unsupported/abstain outcomes.

`PROPOSED_FIXED_PROTOCOL.md` gives the full design, conditional mathematical
identities, proposed budgets and error allocation. It is not an execution
manifest, and this validation task authorizes no new training or simulations.
