# Study R: a prospective cost contrast at a fixed forgiveness dose

Internal preregistration, 15 October 2026. The protocol, prediction data, analysis,
runner, environment, software checks and dependency hashes are frozen in
`PREREGISTRATION_HASHES_R.txt` **before any full R agent is trained**. No public
registry is claimed. All 60 full runs will be reported, whatever the result.

## Motivation and prospective boundary

The original O/P/Q findings already existed when this method was developed.
Forgiven share alone ranks O outcomes well; same-dose contrasts are needed to
test information beyond that ordering. A previous fixed-policy Monte Carlo
surrogate failed a two-fee numerical precision gate and no study was registered
then. That failure remains in the manuscript and exploratory records.

After a renewed request for deeper novelty work, exploratory exact integration
was developed in `exploratory_novelty_2026-10-15/estimator_audit/`. It computes a
different quantity: expected reward from taking leverage 16 once, then staying
flat. The fees .40 and 1.50 had already been inspected in the earlier predictor-
only feasibility analysis, but **no RL agent was trained at these R fee/alpha
conditions**. The present arm selection uses those predictor calculations and
known O/P/Q outcomes. Only the forthcoming R learner outcomes are held out.
This is prospective confirmation of a selected frozen heuristic on this new
condition family, not independent confirmation of method discovery or a claim
of exhaustive predictor superiority.

## Exact empirical diagnostic and frozen calibration

Let X be the one-period fully booked payoff, with an independent fair sign for
the gross spread, and let e be reflected equity. Net reward in euros is
X+(1-alpha)*(-e-X)+. Its expected value is

    m_alpha(e) = -E[cost] + (1-alpha)*E[(-e-X)+].

Average both signs over **all TRAIN rows after the first 200**. The score is
100 times the mean positive part of seven-probe log-interpolated m_alpha on the
same 400 log-spaced equities [.05,2] as O, divided by the exact full empirical
standard deviation of X at leverage 16. The seven probes are .02,.1,.25,.5,1,1.5,2
allocations. No DP, learned critic or simulator rollout is used.

The standalone `predict_r.py` uses the original ten labeled F/H/J/K calibration
conditions only. The geometric midpoint of largest unlearned and smallest learned
indices is **0.08966882375371477%**. The separating range is wide:
(.007323513245505552,1.0979017424880366]%; the midpoint is a heuristic rather than
an identified transition. The one-step score also ignores gamma, so calibration
rows with the same market/cost but different gamma repeat its value.

The Finnish calibration panel is the disclosed revised local copy, rather than
the original J source. Excluding FI gives .09437240005651974% and the same R arm
predictions; this sensitivity is exploratory and does not replace the primary
threshold. The new R panels are DK1/DK2 only. Their exact file hashes are frozen.

Predictions at alpha=.8:

| Fee (EUR/MWh) | DK1 score (%) | DK2 score (%) | Frozen prediction |
|---|---:|---:|---|
| .40 | .158197208029 | .155497857440 | G >= 4 |
| 1.50 | .020683398215 | .032692918450 | G < 4 |

Every low-fee score is >=1.5c; every high-fee score is <=c/1.5. These are
point-heuristic confidence bands in the earlier operational convention, **not
statistical confidence intervals for acquisition**. The exact one-action gain
at 2% equity stays positive at both fees: approximately 9.86/11.42 EUR at .40 and
6.26/8.34 EUR at 1.50. Thus the high fee does not remove every feasible exploit.
Both fully booked controls have exactly zero positive-part index. All exact
predictions, signed gaps and calibration inputs appear in `predictions_r.json`.

Known retrospective failures are not hidden: on O this heuristic classified 6/7
arm gaps but only 46/70 raw maximal-leverage endpoints, and it incorrectly called
alpha=.9 learned. Budget and learner transfers also fail. It is not claimed to
replace Q-star or predict arbitrary learners. Exact empirical integration removes
Monte Carlo error, not market, approximation or calibration uncertainty.

## Training design

- Fees: .40 and 1.50 EUR/MWh.
- Reward shares alpha: 0, .8, 1 at each fee; alpha 1 is the matched full-booking
  control, and alpha 0 supplies a high-incentive descriptive anchor.
- Zones: DK1 and DK2. Seeds: 60, 61, 62, 63, 64, distinct from O/P/Q.
- Total: 2 fees × 3 reward arms × 2 zones × 5 seeds = 60 PPO agents.
- The reflected account, continued trading, TRAIN placebo-sign market, cap25%,
  K0=20000EUR, leverage levels, observations and episode starts match O. Only
  the declared fee differs. `envs_r.py` uses an explicit parameter and never
  changes an existing registered environment or global fee.
- PPO configuration matches O: equity only, 8 environments, n_steps 1024,
  batch 256, 64×64 network, gamma .99, VecNormalize rewards, CPU, 1M requested
  timesteps. SB3 rounds to the next 8192-step rollout, 1007616 actual steps.
- Shared seed labels pair the fee/alpha conditions; no model state is shared.
  Job order is fixed by seed, fee, alpha and zone, independent of outcomes.

## Single composite primary decision R1

For each arm, L is greedy absolute leverage at 2% equity. Let

    G(f) = mean_zone,seed L(f,.8) - mean_zone,seed L(f,1).
    D_s = mean_zone{[L(.4,.8)-L(.4,1)] - [L(1.5,.8)-L(1.5,1)]}.

R1 is SUPPORTED only if **all four** conditions hold:

1. G(.4)>=4;
2. G(1.5)<4;
3. mean_s D_s>=4;
4. an exact one-sided sign test of independent seed-cluster D_s gives p<=.05,
   with zero counted nonpositive. With 5 clusters this requires 5/5 positive,
   giving p=1/32=.03125.

The statistical null is P(D_s>0)<=.5 for the fixed two-zone design and independent
training-seed clusters. This test needs no symmetry assumption, does not test
the population mean, and does not generalize to a population of markets. The
effect-size/G criteria are operational cohort thresholds; G<4 does not prove
population equivalence, absence of gambling or removal of an incentive. The
composite is a single primary decision, not four independent successes.

Power is limited: if independent seed interactions are positive with probability
.9 or .8, the sign component passes with probability .9^5=.59049 or .8^5=.32768
before the other gates. A failure may reflect weak acquisition separation,
seed noise or a poor heuristic; all component values will be reported.

Dose-only 1-alpha has identical values in the same-alpha cost contrast and cannot
supply its directional ranking. Success would establish prospective separation
in this condition family, not dominance over every cost-aware simpler predictor.

## Secondary descriptive and exploratory outputs

Every alpha 0 anchor and alpha 1 control is reported, as are the full probe curves,
maximal-leverage counts and per-zone outcomes. The exact empirical 5^5 paired-seed
bootstrap interval for meanD is descriptive; it does not replace the sign test.
Training visitation/action counts in fixed equity bins, aggregate booked P&L,
fees and injections are saved. Final policy/value weights and normalization are
saved for exploratory diagnosis. They are not new registered hypotheses, and
cannot be used to change R1, the training budget, threshold or arm set.

No stochastic TEST-period evaluation is required for the primary fixed-probe
question. This avoids conflating acquisition with realized deployment profit.
The primary is comparable to O's fixed-probe endpoint. Independent deployment
evaluation, if later requested, will be a separate exploratory analysis.

## Software validation, integrity, completion and failures

Before freeze, `validate_r.py` compares the new environment at fee.10 with the
unchanged O environment over 12096 TRAIN transitions, checks observations,
dynamics and reward equality to numerical tolerance, checks counters, and tests
the analysis on synthetic supported/failed and incomplete cohorts. A separate
8192-step TRAIN-only software smoke uses only the fully booked fee .4 control,
DK1, seed 60, writes `development_smoke/`, and is excluded from R. No treatment
pilot or full R agent is used to choose any design element.

The full runner refuses an absent or changed manifest. Resume skips only valid
complete summaries with the exact settings, full budget, probes, counters and
three saved model artifacts; interrupted jobs restart with the same seed and
settings, and failures are logged. No outcome-based exclusions or replacement
seeds. Full results are atomic job files; all60 unique jobs are required before
the unchanged analysis executes. The aggregator refuses to overwrite any
existing runs/verdicts, reports R1 however it turns out, and appends completion
to the run log. A later rerun must not replace frozen completed job files.

`RUN_LOG_R.md` is hash-protected through its `## After hashing` header and
append-only thereafter. Registered scripts/protocols/predictions/dependencies
are byte-exact. New model artifacts stay local; no eSett raw data are copied.
No commits, pushes, branches or history changes are part of this study.
