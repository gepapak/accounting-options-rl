# Auxiliary forward component excluding direct prepaid maturity returns

16 October 2026. This is a separately defined exploratory auxiliary analysis,
internally source-frozen after primary development generation began and BEFORE
any primary CONFIRM episode. It is not a preregistration or a revision of the
primary 520-limit experiment. No auxiliary statistical endpoint was inspected to
choose this definition. The primary source, selection rule, output, cases and
all failed/abstaining results remain unchanged.

## Source-based question and fixed component

Warm native snapshots may carry claims whose roll fees were paid before audit.
Their direct maturity forgiveness can contribute to the original forward-window
financial target. Test a narrower component excluding the GROSS AND INJECTION
of those old maturity events, while retaining every forward financial cost,
every cost-event injection, every new-claim maturity, and all actual state effects
of old claims. This is a component attribution on the original native paths. It
is not a counterfactual removal of old claims, a comparison with flat deployment,
complete lifecycle profit, or evidence of new learning/acquisition.

The fixed settlement lag is six steps. Initial claims opened before audit must
mature at relative time <6; no audit-created claim can mature before relative
time6. Verify every initial claim in all eight source-derived context supports
has burn_in<=settle_step<burn_in+6. Verify lag/frequency6 and possible new-opening
clocks. With the frozen compact event format, identify OLD maturity events by
`event_sign_epsilon != 0 AND event_step < 6`. They must have zero event cost and
a maturity time present in the initial-claim metadata. Verify all actual later
maturities are compatible with allowed new openings. Any violated boundary or
attribution assertion stops this auxiliary analysis; do not silently redefine it.

For event weights w=gamma^t, gamma=.995, retain all economic financial costs C.
Let O be the old-maturity set and define

    G_new = sum_{j not in O} w_j G_j,
    F_new = sum_{j not in O} w_j I_j,
    A_new = sum_{j not in O} w_j (I_j(+)+I_j(-))/2,
    Y_new = G_new - C + F_new.

Cost-only events have no fair coin, so their native injection remains. Local
pairing uses ACTUAL pre-event cash, not a recomputed no-old-claim history. Retain
the primary whole-path predictor controls and all their innovation coins,
including old maturity coins: arbitrary predictable symmetric controls remain
mean zero for this new target. Do not fit a new predictor or reselect its budget.

## Fixed confirmation and comparison protocol

Use ALL forty original policy/profile cases and ALL six original methods, at
alpha=0 only. Every method uses the same 256 primary CONFIRM trajectories per
case; RAW does not pool development paths in this auxiliary analysis. Reuse the
primary alpha0 cap, beta and path-budget choices exactly, as selected by the
unchanged frozen primary TUNE/FIT rule. These choices optimize the original
target, not this component; disclose that limitation. No new policy selection,
fit, tuning, native query, context, sample-size expansion or stopping rule.

The six representations are new RAW; capped new RAW; capped F_new-C; capped
A_new-C; and the latter minus the original Ridge or MLP centered control, capped
after subtraction. Existing primary Cbar/Gbar supports conservatively cover this
component, because the retained events are a subset of gross/injection events.
Use the same deterministic engineering numerical-pad rule and sample ddof1
empirical-Bernstein formula, with delta=.05/240 for 40*6 one-sided lower limits.
Positive means LCB>0 and the deterministic unavoidable access-fee lower bound>0.
No extra cost confidence interval: E[G_new-C]=-E[C] and native unavoidable access
fees establish booked harm analytically.

This is a SEPARATE 95% family. It does not combine with the primary 95% family
into a paper-wide 95% guarantee; a union bound over the two families supplies at
least90%, without any independence assumption. Shared paths are permitted.
Different RAW n, family allocation and target prevent interpreting changed
certificate counts as an isolated causal effect of removing old forgiveness.

## Freeze, execution and complete reporting

`analyze_new_contract_component.py freeze --run-id run_001` refuses any existing
confirmation phase directory, preserves exclusive plan/review/analyzer snapshots
and binds the original SOURCE_FREEZE and cases hashes. The analysis action must
verify its freeze UTC precedes EVERY confirmation phase start UTC. It also
requires the primary full completion/aggregation markers. Until then it performs
no auxiliary numerical analysis and emits no partial endpoint.

After full completion, verify primary input hashes, rebuild G/C/I/A from compact
events, retain the original coins/deltas and choices, and save all 240 bounds,
all cases/abstentions, scalar supports, analytic booked-harm bounds, per-path
component summaries, old-event exclusion counts and provenance hashes. Report
the component's inherited-state interpretation and the original-objective
choice limitation. Preserve primary outputs and frozen files unchanged.
