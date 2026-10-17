# Proposed fixed protocol: sample-only auditing of authentic rich learned policies

16 October 2026. DESIGN ONLY: no policy inference, environment episode, fitting,
random draw or new training was run by this validation task. Root and the engine
owner must complete compatibility/runtime validation and write a final frozen
execution manifest before endpoint evaluation. The numerical query budget below
is a proposal to be accepted or changed using runtime/compatibility information
only, before fresh fitting or confirmation outcomes. This is exploratory work,
not a new preregistered learner study.

## 1. Fixed policies and authentic interfaces

Primary cohort: ALL ten Prototype5 plain MAPPO investors and ALL ten feasible-
action MAPPO investors listed in `../checkpoint_inventory.json`. Freeze the
inventory digest, twenty actor digests, paired batteries, training configurations,
causal forecast/cache inputs, policy implementation and adapter source. No policy
is selected using a probe, historical return, fit score or confirmation result.
The forecast-anchor checkpoint is a separately named reference, not silently a
twenty-first primary learned policy. Study L20/R60 are a portable secondary
cohort only; their one-dimensional actors must not stand in for rich-trained
controllers.

Plain investors use 12 local features; feasible-action investors use 20. Both
use continuous delta actions and a custom centralized-critic Beta actor. Restore
the correct action distribution, local actor feature order, executed action
mapping/cadence, paired battery and each cohort's recorded runtime contract.
The centralized critic is not needed to turn the local actor into a 48-feature
policy. Native reward shaping is recorded descriptively and is NOT the monetary
audit target below. Freeze inference and any normalization statistics. Do not
update weights or refit actor observation/reward normalization on audit episodes.

Confirm all twenty policies under TWO fixed solvency profiles at initial cash
1e-4 times recorded initial trading-sleeve allocation, preserving the ORIGINAL
initial allocation as the notional sizing base. The first profile restores each
checkpoint's native strict margin and MTM-loss-exit contract. The second explicitly
disables those two guards as an accounting-vulnerability stress, preserving costs,
caps, impact and all other mechanisms. This is a named intervention, not the
checkpoint's original runtime contract. The final thin-cash proposal is based on
public contract metadata: the native original trading sleeve is about 662 million
DKK, so .1 times that sleeve is about 66 million DKK, whereas 1e-4 is about 66,200
DKK. Earlier .02 and .1 proposals are superseded before fitting or confirmation
outcomes. Original notional sizing, liquidity caps and costs are preserved; this
is a substantial solvency stress rather than a new original-market test.

The strict profile is a GUARD-RESTORED comparator, not an analytically inactive
or zero-forgiveness control. Warm snapshots retain prepaid pending claims, and
native maturity can precede margin processing. Retain actual financial events,
activity and injections for every strict policy even if guards suppress new
trades. The same-action alpha=1 ledger, rather than this profile, supplies the
analytically nonpositive no-edge booking comparison.
A normal-cash state may be used for deterministic compatibility checks only,
outside the primary endpoint grid. The final manifest must specify exact config
switches and reset balance before endpoints; if
compatibility fails, retain that case as NOT EXECUTED with its technical reason,
not as a dropped statistical observation.

## 2. Native harder law and independent episodes

Use native chronological cached market/settlement/liquidity contexts, settlement
queues, exposure state, collateral/margin and costs. The new stress law changes
only the sign of each declared financial maturity gross event: a fresh fair coin
is hidden until AFTER the controller's action and all current observations.
Freeze whether a coin acts on a single contract or an aggregate maturity batch.
Magnitudes and context dependence remain native. This is explicitly a controlled
no-edge stress law; it does not imply fair signs or no edge in actual markets.

Use two explicitly fixed physical contexts, sampled through a declared mixture
of eligible chronological native held-out block starts. The exact scenario IDs,
context weights and start distributions must be frozen after compatibility
validation, before FIT/TUNE outcomes. These are components of each case's law,
not an extra pair of outcome-selected cases. FIT, TUNE and CONFIRM must be
independent draws from that SAME mixture; otherwise RAW cannot pool phases.
An episode is the independent unit. Within an episode market contexts, actions,
cash, liquidity, queues and feedback are non-IID. Instantiate each whole episode
from a fresh deep copy of configuration, controller/normalizer state, random
streams and the declared causal history snapshot. Calling reset(seed) on a
continuing native object is insufficient. Independently draw an eligible cached
block/scenario with replacement under the frozen sampling law; reuse of a fixed
archive does not supply new markets. Forecasts/normalizers must be available at
the historical decision time, with correctly aligned timestamps and no future
settlement labels or full-episode normalization.

Proposed active decision horizon is 96 native calendar transitions, following a
384-step native warm history. There are eight fixed context snapshots per case;
each audit episode clones its entire warm state independently. The final
manifest must define settlement/closure of all remaining financial queues and
inventory. Current source refuses new financial contracts whose settlement would
fall outside the complete context window; assert the native queue is empty at
every audit episode's endpoint. Warm snapshots may contain prepaid claims. Audit
costs and returns are the declared forward financial component, excluding sunk
warm-history costs; do not call this complete lifecycle profit. If a changed
window permits residual liabilities, add a fixed sufficient tail or an explicit
conservative terminal valuation before freezing. Native zero delta action is HOLD,
not liquidation.
If flattening is used, invoke the native target-exposure close operation, charge
its costs and include every tail transition. A certificate cannot censor pending
liabilities at the horizon. Define a fixed total calendar count T (including any
tail), gamma=.995 to match these actors, and its discount convention before draws.

## 3. Monetary target and ordered financial events

Preserve native event ORDER. Log separately every financial maturity gross event
and every financial roll/access/collateral/execution cost event, with the actual
pre-event cash and native floor injection. Do not merge gross and cost into one
end-of-step G-C floor: intervening injections generally differ. Exclude battery,
physical-asset and other nonfinancial floor injections from the target. Native
fund return and shaped RL reward remain separate descriptive outputs.

For financial event j at calendar t(j), let G_j be signed gross, C_j>=0 its
economic booked financial cost, and F_j>=0 the native injection attributable to
THAT event. All share w_j=gamma^t(j). Cost-only events have G_j=0. Define

    B = sum_j w_j*(G_j-C_j),
    F = sum_j w_j*F_j,
    Y_alpha = B + (1-alpha)*F.

Primary view alpha=0 is the financial component of native floor accounting;
alpha=1 is its same-action FULL BOOKING ledger. They use the SAME native physical
dynamics, actions and cash path. Full booking is not a separately redeployed or
retrained counterfactual engine. Both views are reported for every checkpoint
and profile; there are 40 policy/profile cases, not 80 independent policies.

Conditional fair signs after action give E[sum w_j G_j]=0 for this financial
component, so E[B]=-E[C], C=sum w_j C_j. Strict booked harm follows from a proven
positive unavoidable financial fee or an independently allocated lower bound
E[C]>0. Otherwise report a charged-only certificate and ABSTAIN on strict harm.
If attribution contains physical edge, negative/rebated costs or previsible
financial signs, do not assert this identity: either correct the target before
freezing or use direct simultaneous charged-lower/booked-upper bounds with
valid deterministic supports.

Idle access fees can make always-flat booked cash negative. Absolute E[Y]>0 and
E[B]<0 remains a clearly named false-profit certificate. A paired always-flat
ADVANTAGE is a different estimand; report it only with its own direct bounds.
After subtracting flat, incremental costs need not remain nonnegative. Never
reuse the old cost-only identity for that difference without proof.

## 4. Sample-only representations and exact validity

No method may read the full held-out magnitude distribution, evaluate an exact
conditional magnitude-law expectation, solve a DP, or substitute an uncontrolled
fitted drift for its true conditional expectation. The fair stress coin is a
known structural intervention; describing the method as sample-only refers to
unknown magnitudes/contexts, not to an assumption-free arbitrary simulator law.

Fix six primary representation families before fitting:

1. RAW: charged whole-episode return, with a model-derived deterministic support.
2. CAPPED RAW: upper-capped charged return with the same valid lower support.
3. RESIDUAL: upper-capped Z=(1-alpha)*F-C, with lower support -Cmax.
4. SIGN-PAIRED: at each actual pre-event cash/context and realized magnitude,
   A_j=.5*(F_j(+magnitude)+F_j(-magnitude)); cost-event injection is integrated
   deterministically in native order. Use upper-capped (1-alpha)*sum w_j A_j-C.
5-6. RIDGE and MLP CONTROLS: two separately fixed bounded predictors give symmetric branch predictions
   u_plus,u_minus using the SAME model, never true/model mixed branches.
   m_j=u_actual-.5*(u_plus+u_minus) has conditional mean zero. Upper-cap
   W=Zpair-sum_j M_j only AFTER subtracting the control. Zpair is the
   sign-paired base; M_j is in globally discounted episode-currency units.

For SIGN-PAIRED, both branch injections must be exactly obtained from the native
local event accounting rule. Average at the ACTUAL current state; do not replace
future native histories by an unverified model history. If both scalar branch
cash floors can be read algebraically from the existing G/C ledger, no extra
environment transition query is needed. Count both branch arithmetic evaluations.
If complete native event branching/cloning is required, count every cloned
transition and verify its state/randomness coupling before the experiment.

For either LEARNED CONTROL, condition on pre-coin history and realized magnitude. Branch
features can be fixed functions of that information and a hypothetical sign;
they must not contain the actual coin in the shared context. Arbitrary model
error is allowed because centering is exact. Fit only on independent FIT episodes
and choose hyperparameters/coefficient/caps on independent TUNE episodes.
If u is bounded in [0,U], |m_j|<=U/2. A fixed beta gives lower support
-Cmax-|beta|*U*sum_j w_j/2. An optional PREDECLARED whole-path control-amplitude
budget Bctrl gives the tighter support -Cmax-Bctrl. Proposed rule:
delta_j=.5*(q_theta(pre,+)-q_theta(pre,-)),
amp_j=sign(delta_j)*min(abs(delta_j),Rremaining),
M_j=epsilon_j*amp_j, Rremaining-=abs(amp_j), starting Rremaining=Bctrl.
Predictor targets are future globally DISCOUNTED paired-base-minus-cost returns,
so amp_j must not be discounted again. Spend budget using the symmetric absolute
branch difference before using the actual coin, ensuring sum_j abs(M_j)<=Bctrl.
Never spend according to
whether the actual signed control looks favorable. Upper clipping is a lower
mean estimator since E[sum_j M_j]=0. Accuracy is not an assumption;
variance or confidence width may be worse, and failed certificates remain.

The native floor always has nonnegative pre-event cash. Consequently its simple
Tanaka correction equals RESIDUAL and is an algebraic equivalence check, not a
fifth independent method. If a separate terminal ledger is later studied,
d(n+X)-d(n)+1{n<0}G is nonnegative: d(n+X) at n>=0 and C+(n+X)+ at n<0.
That sample-only identity has the right conditional mean under no edge, but
it must not be applied to a different native account without matching its events.

An exact conditional-law oracle may be reported as a separately labelled
benchmark only if the complete harder conditional law actually supplies one.
It is not a sample-only competitor, receives no free queries, and cannot be
quietly reconstructed from all confirmation magnitudes.

## 5. Phase 1, query matching and confirmation plan

Phase 1 is compatibility/runtime validation on declared development contexts,
without endpoint tables or policy/condition selection: actor identity, local
feature dimensions, actual actions, event accounting and branch rules, complete
state isolation, timing/closure and deterministic supports. Freeze its outputs
and any technical failures. Set the FINAL numerical budget using these timings
before fresh FIT/TUNE/CONFIRM draws. The initial ceiling placeholder
Btotal=262144 transitions/case is withdrawn as impractical at native runtime.
Root's current outcome-blind choice is (Nfit,Ntune,Nconfirm)=(32,32,256),
giving 1,228,800 audit transitions plus 245,760 warm-history transitions for forty
cases and eight contexts prepared separately for development and confirmation.
The total is 1,474,560 native transitions. MAX3 process workers is the current memory
limit; do not substitute threads for the process-local native runtime. See
`RUNTIME_AND_POWER_NOTE.md`; this draft is not authority to launch work. The final
execution manifest, including resource/partial-family rules, remains root-owned.

Primary comparison matches TOTAL simulator queries, not just final confirmation
paths. Charge every actual/conditional/clone transition, FIT episode, TUNE cap
search and any failed/retried query to its algorithm. Count policy forward calls,
critic forward calls, fitting samples/optimization passes, branch arithmetic,
preprocessing, CPU and wall time separately. Sunk prior actor training is shared
fixed input, reported with its actual historical steps; no new RL training is
hidden in audit cost. Shared fitting can be amortized only under a frozen declared
cohort-level accounting rule, with total and marginal costs both shown.

The final plan fixes Nfit, Ntune, Nconfirm whole episodes per case, all from the
same native law. RAW, with no fitted choices, pools ALL three phases for its CI.
Capped raw/residual/paired methods choose caps on FIT+TUNE and confirm only on
independent CONFIRM. Ridge and two-layer MLP fit on FIT, choose caps/coefficient/
control budgets on TUNE, and confirm on CONFIRM. Freeze every case's nuisance
fits BEFORE any confirmation endpoint is exposed. This shares exactly the same
total native episodes across methods and charges all three phases to all
algorithms; numerical fitting is additional work reported separately. If a method
needs cloned environment transitions, charge those too and resolve the fixed
budget before confirmation, rather than treating them as free. Round DOWN to
complete T-step trajectories and retain unused slack. No stopping on success or
increasing n after abstention. A secondary same-confirmation-n ablation may use
the common CONFIRM phase for RAW as well, but it is separately labelled and cannot
replace RAW's fair pooled primary result. Its mean/variance/width is descriptive
unless another 80 lower-bound events are allocated before confirmation; the 520
primary events do not cover extra RAW confirmation-only certificates.

Proposed TUNE coefficient grid beta={0,.25,.5,1,2}; relative path-control budget
grid={0,1/16,1/4,1,4} times a fixed currency scale. Freeze the scale, cap grid,
predictor clipping and optimizer architecture/epochs before FIT. Two-layer MLP
and Ridge are separate methods; wrong/null predictors are exact-toy controls,
not extra selected empirical methods. Fitting/tuning may optimize its own score
but must not choose policies, physical contexts, profiles or sample sizes.

Suggested independent seed bands: FIT26300000, TUNE27300000, CONFIRM28300000,
plus 1000*policy_id+10*profile_id and an explicit episode index/substream. Use a
separate compatibility band25300000. Record the exact native RNG derivation;
independence must include every engine/NumPy/Torch/cached-context substream.

For six methods, two views and 40 policy/profile cases, allocate 480 charged lower
events plus 40 common-prefix cost lower events: delta=.05/520 for a separate 95%
family guarantee. Use a fixed minimal common confirmation prefix for the cost
test, chosen from the predeclared method budgets, not from observed costs.
If a positive unavoidable cost is proved deterministically, the corresponding
cost event can remain conservatively allocated. A changed method/case grid or
direct-booked fallback must receive its final allocation BEFORE confirmation.
Use fixed-n one-sided empirical-Bernstein bounds with ddof1 and genuine model
supports. The proposed deterministic engineering floating-point allowance is
max(1e-6, 1e-10*(1+Gbar+Cbar+Bctrl)*96), used to expand supports and lower
endpoints. This is an explicit numerical allowance, not a formal rounding proof.
Clipping caps/coefficient grids and any predictor clipping U/Bctrl must be fixed
before TUNE and selected without confirmation. If the model has no finite known
lower support, use a separately justified robust bound or mark UNSUPPORTED;
neither a sampled minimum nor a plug-in cost mean creates a valid support.

## 6. Reporting and audit gates

Retain every checkpoint/profile/view/method row, including inactivity, full-booking
controls, technical failures and abstentions. Report paired detection/width
patterns and query-versus-n separately. No random-row CV, counting time steps as
IID, selecting the best policy from confirmation, or treating correlated seeds
as new markets. Native cohorts differ in information/execution contracts, so
cross-cohort outcome differences are descriptive, not a causal algorithm test.

Store source/data/model hashes, checkpoint metadata, phase seeds, exact budgets,
support derivations, all fitted models/hyperparameters/cap scores, per-episode
path summaries, event accounting identities and cost/query logs. Include the
partial/full state when a CPU guard triggers; if any planned family component is
incomplete, mark the family INCOMPLETE and make no complete-family discovery-rate
claim. A serialization fix may replay identical streams with immutable failed
source and byte/AST disclosure, but cannot change the numerical design.

Final interpretation distinguishes (a) authentic learned-policy compatibility,
(b) valid sample-only confirmation under the declared stressed native law,
(c) conditional power/query tradeoffs, and (d) original-market performance.
This experiment directly tests (a)-(c); it cannot establish (d) or generic
estimator/optimizer superiority from one engine/archive and two solvency profiles.
