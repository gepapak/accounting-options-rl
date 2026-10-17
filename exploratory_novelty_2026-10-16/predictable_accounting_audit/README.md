# Conditional-kernel accounting audit (exploratory)

Completed 16 October 2026. This is a specialized policy-verification method,
not another acquisition predictor or a newly registered learner study. Study R
remains failed; none of its records is changed. No learners were trained here.

## Contribution and required access

`THEORY.md`, the exact scripts and the manuscript proofs specialize classical
predictable compensation. For net increment X=G-C with conditional EG=0 and
C>=0, write d(n)=(-n)+, h(n)=E d(n+X), c=EC. The debt drift
Delta(n)=h(n)-d(n) lies between 0 and h(0) for every ledger. Evaluating that
drift along a frozen controller gives a bounded UNCLIPPED unbiased observation
of its charged expected reward, provided the conditional stop-loss curves and
known uniform first-moment bounds are available. Realized prices AND fees can
be unbounded. A finite empirical or analytic oracle supplies the moments;
finite moments alone, a fitted critic or uncontrolled Monte Carlo estimates do
not supply the guarantee. The law must condition on everything the controller
sees. Additional state/history information may invalidate row-averaged curves.

With a FINITE uniform ratio h_t(0)<=rho*c_t, alpha>=1-1/rho makes expected proxy
advantage nonpositive for every admissible adaptive policy and finite horizon
under the floor and terminal clocks. With kappa=(1-alpha)*rho-1>0, a positive
charged lower bound ell also implies expected booked return<=-ell/kappa.
No mandatory trade or additional noisy cost test is required. The sharp
all-state threshold for cash-independent laws equals the classical gain-loss
ratio. Harmless0/0 actions are ignored; zero cost with positive h(0) prevents a
finite ratio. No claim of a new Doob theorem, ratio, optimizer or concentration
inequality is made. Full charging remains the robust no-edge remedy across
unrestricted kernels; a fixed partial charge is kernel-specific.

## Completed evidence (all failures retained)

| Representation | Selected /72 | Neutral /72 | Fixed development policies /9 |
|---|---:|---:|---:|
| Raw reward |7|5|0|
| Capped raw |17|18|0|
| Capped realized-cost residual |24|24|0|
| Capped mean-cost residual |24|24|0|
| Uncapped predictable drift |36|33|3|

The 24 empirical cases are DK1/DK2 x floor/terminal x three initial equities x
H16/H96, with three independent search repetitions. TEST rows are excluded from
audit development, although earlier learner research already used TEST data.
Every method gets the same 13 candidates and logical transition budget. Neutral
controllers are fixed independently of estimator scores: constant16, barrier1,
and two-stage(1,2), one per repetition. They are DIFFERENT controllers, not
three repetitions of one fixed policy. All720 lower bounds allocate .05/720;
confidence is conditional on the finite empirical IID law, not future markets.

Exact isolated-action integration is positive in 12/24 cases. Predictable search
confirms all 12 in all 3 repetitions; no estimator adds a case beyond that
reference. These are confirmation gains, not broader exploit discovery.

The separate nine-policy comparison revisits one KNOWN SELECTED DK1 TRAIN
development condition. All45 lower bounds allocate .05/45 and use 32768 fresh
paths per frozen policy. Predictable drift confirms the original feedback
controller and final16-exposure constant policies lasting16 or48 periods;
their lower bounds are8.06449767, .32596860 and 4.54279468 EUR. All four
comparators abstain. The search helper retains its definition-time .05/720
heuristic allocation; actual confirmation correctly uses .05/45. This source
behavior is disclosed without rerunning or retuning. In the larger524288-path
matched comparison, simple delayed48 constant trading also certifies under the
older capped residual: feedback is not necessary in that empirical example.

The16-case Pareto suite uses p1.5/2.5, fixed/proportional fees, two accounts and
alpha .8/.95. Infinite price variance and unbounded proportional costs are
covered. Both predictable and capped MEAN-cost residual confirm the SAME 3/16;
predictable sample variance is higher in 4/16, and all terminal cases abstain.
The eight .95 controls obey the analytic uniform repair thresholds. The32
lower bounds allocate .05/32 as a separate family. There is no universal power
or variance dominance and no unique heavy-tail sign-detection claim.

Known first-moment access and oracle arithmetic are extra information/work.
Shared paths provide equal logical simulator budgets, not equal CPU costs.
The held-out run uses 473948288 shared transitions and 533725901 curve queries,
120.703 CPU/122.962 wall seconds on one numerical thread. The fixed-nine run
uses 22487040 active transitions,20829763 oracle points and 5.328 CPU seconds.
Pareto validation uses 5695585 active curve evaluations and 7 CPU seconds.

## Files and safe verification

- `predictable_audit.py`: frozen finite-kernel implementation and path/range
  calculations. Controllers observe clipped ledger; compensation uses the
  internal accounting ledger.
- `EXPLORATORY_PLAN.md`, `heldout_outputs_attempt2/`: frozen held-out audit
  protocol, all 72 cases, search scores, independent confirmation and hashes.
- `FIXED_POLICY_PLAN.md`, `fixed_policy_outputs/`: separate frozen9-policy
  estimator comparison.
- `UNBOUNDED_PLAN.md`, `unbounded_outputs/`: all 16 constructed cases, oracle
  checks, bounds, caps, controls, counts and replay provenance.
- `exact_predictable_checks.py/json`:32 exact finite-law account/policy/charge
  scenarios and counterexamples; `exact_harm_corollary.py/json`:120 additional
  optional-trade checks, including the zero-cost exception.
- `SOFTWARE_ATTEMPTS.md`, attempt-one files and the Pareto replay note retain
  serialization failures. Successful replay changes only scalar serialization,
  retaining plans, seeds, numerical design and core source bytes.

Read-only checks from the code root:

```powershell
python exploratory_novelty_2026-10-16/predictable_verification/verify_predictable.py
python tools/verify_studies.py
```

The independent evidence audit checks806 bounds across separate families and
18482 assertions plus hashes, without fresh draws. It recomputes intervals
from stored moments and independently checks supports; it does not reconstruct
every path moment from new simulations. The repository verifier passes552
checks, preserving all prior 501. For simulation reproduction use a disposable
copy/new output location: completed outputs and frozen sources must remain
byte-exact. Do not use `--write` to replace verification records casually.

Closest prior art: Henderson--Glynn2002 potential-drift martingales,
Vadori etal2020 predictable reward decomposition, Bernardo--Ledoit2000 and
Biagini--Pinar2013 gain-loss ratios, Thomas etal2015 independently selected
truncation, and Maurer--Pontil2009 bounded confirmation. Full verified primary
links and novelty limits are in `Paper1/AI_NOVELTY_COMPENSATOR_LITERATURE.md`.
