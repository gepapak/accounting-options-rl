# Sample-only audits of authentic frozen RL policies

16 October 2026. This is an exploratory, outcome-blind execution plan, internally
hash-frozen before the statistical experiment. It is not a new registered
learner study. Existing registered studies, raw results and checkpoints are
read-only. The software-validation traces are excluded from every estimate.

## Question and contribution boundary

Can local sign pairing and exactly centered, learned continuation controls
certify positive loss-forgiven financial incentives using sample trajectories,
without integrating a conditional payoff or cost law? Does learning improve on
raw returns, capped raw returns, and simple cost-aware residuals on actual
trained policies in the full Prototype5 engine? A null result or a simple
baseline tie remains a result. No training of new RL policies is authorized by
this experiment. Ridge and MLP nuisance regressors are fitted only on FIT.

The target is the discounted financial-event component, not fund NAV, battery
profit or the native shaped RL reward. Fresh hidden fair signs multiply whole
native maturity batches only, after action mapping. Native observation,
normalization, controllers, batteries, physical transfers, delayed settlement,
nonlinear impact, access and collateral costs remain active. Sign symmetry is a
known instrumented simulator primitive; neither arbitrary black-box access nor
an unknown asymmetric real market is covered. Costs must have a deterministic
executed support bound. Price magnitudes need not have a known conditional law.

## Fixed cases and generator

Use every available policy in the two final ten-seed cohorts: plain MAPPO and
feasible-action MAPPO, seeds 7,42,123,2025,3007,5001,8102,9005,10001,11202.
No checkpoint is selected by returns or by this audit. The scalar continuous
financial actor has native 12D or20D observations and a centralized critic;
the matching saved battery DQN is also replayed. There is one controlled
financial exposure factor, not an independently controlled multiasset portfolio.

Each of20 policies has two cases: strict (native guards) and vulnerable
(disable the trading-sleeve margin gate and set the MTM loss-exit threshold to
positive infinity). Both start the audit at cash=1e-4 times the ORIGINAL initial
sleeve F=662068965.5172415 DKK, preserving F as the notional sizing base. Native
halt flags, histories and pending claims are retained. This is an explicit
thin-cash intervention, not evaluation of original deployment profitability.
The strict arm is a guard-restored comparison, not a guaranteed inactive or
zero-forgiveness control. Maturity can precede margin processing; prepaid claims
from warm-up can contribute to the forward-window target. Report those claims.

Earlier design drafts used .1F. Before any statistical endpoint was obtained,
inspection of the original F, participation limits and one-maturity risk scale
motivated 1e-4F as an intentional distressed condition. This does not claim .1F
could never default over a whole path. The two .1F software traces and failed
fingerprinting attempt are retained unchanged and excluded from the experiment.

For each case use a uniform mixture of eight fixed chronological contexts:
native original/v2 evaluation datasets, each with four integer linspace starts
from0 to length-(384+96). Current lengths give starts0,12944,25888,38832.
Run384 causal native warm-up steps at original cash before cloning the full
economic/controller/normalizer/RNG state. Apply the cash intervention after
cloning, then run96 audit steps. Only the non-economic memory-manager lock is
replaced on clone. All maturity queues must end empty; the native opener refuses
contracts settling beyond the finite window. Prepaid initial queues are recorded.

For every episode use NumPy SeedSequence([20261016,phaseband,case_id,episode_id]),
with bands FIT26300000,TUNE27300000,CONFIRM28300000. Its generator draws context
uniformly from8 and96 independent equiprobable signs. Cases are ordered by
policy index (plain0..9,feasible10..19), then profile (strict0,vulnerable1):
case_id=2*policy_index+profile_index. Native seed equals the saved checkpoint seed.
All phase generators have the same episode law and independent streams. The
claim is conditional on these fixed native contexts and warm-up states.

## Budget, fitting and selection

Per case: FIT32, TUNE32, CONFIRM256 independent whole episodes. Exactly12800
audit episodes,1228800 audit steps. No outcome-dependent extension or case
removal. Development and confirmation regenerate eight warm snapshots per case,
adding245760 deterministic native warm-up steps,1474560 total native steps.
Count actual steps and reruns explicitly. At most three process workers, one
numerical thread each. Resource-driven reduction of concurrency does not change
streams or sample size. There is no outcome-based early stop; technical failures
halt and retain partial outputs for exact-stream repair/replay. No automatic
wall-time failure threshold is used. Completion time is measured, not a stopping
rule. Work is launched with a durable log and PID record.

All methods share320 native trajectories/case. RAW makes no fitted choice and
pools all320; the other methods confirm on256. Simple capped baselines select
on all64 development episodes. Learned controls fit pooled FIT from all40 cases,
then select on each case's32 TUNE episodes. All models and all choices freeze
before the first CONFIRM episode. All methods receive the same source-derived
support information and local ledger instrumentation. Supervised fit/prediction
CPU is additional and reported separately. No fitted conditional moment is
treated as exact.

Six methods: RAW; capped RAW; capped realized-injection-minus-cost residual;
capped local SIGN-PAIRED residual; Ridge centered control; MLP centered control.
The terminal Tanaka identity is not counted as another floor method because it
coincides with the realized residual here. Charge views alpha0 andalpha1 reuse
the same actions and physical histories; alpha1 is an accounting view, not a
new policy trained under full charging.

The cap grid in DKK is [1000,4000,16000,64000,256000,1024000,4096000,16384000].
Control beta grid [0,.25,.5,1,2]; path absolute-amplitude budget B/cap grid
[0,1/16,1/4,1,4]. Zero beta/budget is one deduplicated candidate. Select the
highest projected fixed-CONFIRM empirical-Bernstein lower score on development
(simple methods) or TUNE (learned methods); deterministic ties prefer smaller
budget, beta, cap. Search scores are not confidence statements.

Nuisance features are the actual presign native actor observation padded to20,
common cash/base payoff/cost, hypothetical sign/after-cash/injection, actual
action, time, policy/profile/type and local paired injection:32 columns. Money
features divide by original F. Both branch queries use the SAME frozen model
and synthetic feature map. The target is globally discounted FUTURE paired
injection-minus-cost, strictly after the current event, in DKK. No true/model
branch mixing or future-observation feature is allowed. FIT-only target1%/99%
clipping and StandardScaler; Ridgealpha1; MLP(64,64),ReLU,Adam,alpha.001,
learning rate.001,max120iterations,early stopping on internal20%FIT,
patience12,batch256,seed29300000. Predictive accuracy is not a validity assumption.

## Certificates and supports

For solvent pre-event cash e, gross epsilon*m and financial cost c, actual
injection I=(-e-epsilon*m+c)+ and local pair A=[(-e-m+c)++(-e+m+c)+]/2.
The paired path residual (1-alpha)*sum(gamma^t*A)-sum(gamma^t*c) is unbiased for
the expected charged financial component because E(epsilon*m|presign)=0.
For each learned model delta=(qplus-qminus)/2; spend predictable amplitude
sign(delta)*min(beta*abs(delta),remainingB), subtract epsilon*amplitude, and
decrement remainingB by absolute amplitude before seeing that event's sign.
The resulting control is exactly centered for any misspecified predictor and
has absolute path sum at mostB. Upper-cap AFTER subtracting the control.

Use gamma.995. Source-derived per-calendar envelopes include carried warm-up
claims, every allowed roll, funding and access fee, native entry prices and
CONFIGURED settlement price bounds, and the source's zero-cap fallback. The
support of the uniform context mixture is the maximum deterministic bound over
all8 contexts, never a sampled trajectory extremum. With discounted Cbar,Gbar:
RAW has support[-Gbar-Cbar,Gbar]; realized residual[-Cbar,(1-alpha)*Gbar];
pair[-Cbar,(1-alpha)*Gbar/2]; control[-Cbar-B,(1-alpha)*Gbar/2+B]. Capping lowers
the upper endpoint to its minimum with the cap. Excluded transfers prevent using
minus initial cash as a component lower bound.

One-sided empirical-Bernstein lower limits use sample ddof1 variance:
mean-sqrt(2*s2*log(2/delta)/n)-7*range*log(2/delta)/(3*(n-1))-pad.
Expand endpoints by the predeclared engineering numerical allowance
pad=max(1e-6,1e-10*(1+Gbar+Cbar+B)*96)DKK. This is conservative numerical
engineering, not a formal floating-point error proof. Support violations stop
the experiment. Use delta=.05/520 for480 charged limits (40*2*6) and40 cost
limits. Positive means a strictly positive lower limit. Unselected cost limits
pool all320 same-law episodes and have[0,Cbar] support; unavoidable native access fees separately imply strict expected booked
harm. At alpha1 the expected component is nonpositive; any positive certificate
is a control failure to investigate, not evidence of an exploit.

## Outputs, integrity and interpretation

Save all phase path summaries, seeds, contexts, injections/costs/gross, support
metadata, activity, terminal queues and initial pending claims. Save FIT/TUNE
private traces/features and frozen models/choices; CONFIRM cached branch
predictions/deltas permit arithmetic revalidation. Raw market/checkpoint-derived
features and model bundles remain gitignored under private execution storage;
do not redistribute eSett raw/panels or native licensed inputs. Public artifacts
contain aggregate results, anonymized case identifiers and provenance hashes.
Preserve failed attempts and never overwrite a completed artifact. Exact replay
after a software repair uses the same plan/seeds/N, with old source snapshots
and a separately documented new attempt; it cannot retune for stronger results.

Report every case/method/charge result, abstention, support width, cost harm,
native activity, initial prepaid claims, CPU/wall time and query counts. Compare
Ridge and MLP to the best simple baseline, including sign pairing. No advantage
for the neural model means no demonstrated need for neural learning. Stress
success does not establish forecast skill or ordinary profitable operation.
Generic antithetic integration and learned martingale control variates are prior
art. The proposed contribution is the accounting-specific sample-only valid
construction, its explicit access assumptions and test on authentic frozen RL
controllers; its empirical strength depends on the results.
