# Every accounting rule is an option

**Reinforcement-learning traders learn to exploit loss forgiveness, and a sign-paired reward prevents it.** Code, data pipelines, pre-registrations and aggregated results for the paper of the same title.

## The question
An accounting rule decides what an agent's owner receives once losses exceed equity, and that claim is an option. Trading simulators often floor an account's cash at zero, end an episode at bankruptcy, or let losses accumulate until a final settlement; real settlement systems book losses with a lag and hold collateral. Limited liability rewards risk, most strongly near zero equity ("gambling for resurrection"). This repository asks which option each rule writes, whether reinforcement-learning (RL) agents *learn* to exercise it, whether standard evaluation then reports the behavior as profit, whether acquisition can be anticipated, how to prevent it when the simulator's ledger is not exposed, and how large the option is in real Nordic imbalance settlement.

## Main results
- **Five accounting rules form a ladder of options.** Full liability ≤ capped final loss ≤ settlement lag ≤ terminal limited liability (Merton's call) ≤ the zero floor. For fixed positions without fees the floor's subsidy is between one and two times the terminal call, reaching two in the diffusion limit; reported equity is the booked ledger reflected at zero, and reported equity alone cannot identify booked performance. Charging every injection restores the booked objective with continued trading.
- **Origin.** In an audit of a validated electricity-market engine, a legacy protocol reported a learned controller first (**+270%**) while its booked P&L was **−535%**. A missing liquidity cap and missing solvency rules were jointly necessary; reconciling reported against booked returns flagged 53 of 54 affected runs and none of 246 others.
- **Agents learn to exercise the options** (placebo markets with sign-randomized imbalance spreads, where no policy can profit):
  - floor-trained agents raised leverage as equity fell and reported **+38% to +148%** while booking **−15% to −57%** (Study B; replicated with growing false profit in Study C);
  - with equity as the only input, **all ten floor-trained PPO agents chose maximal leverage at the low-equity probe**, 11.1 units of 16 above full-liability agents (Cohen's d = 2.6; Study F); A2C and DQN also gambled (Study I); Finland reproduced the pattern and calm southern Norway did not (Study J);
  - charging injections in full cut low-equity leverage by 9.6 units [4.5, 13.8] with matched continued trading (Study L);
  - under a **settlement lag, all ten forgiven-deficit agents took maximal leverage at every insolvent probe** (Study N, 6 of 9 hypotheses).
- **Anticipating acquisition (Studies O, P, Q; R failed).** A pre-training advantage-to-noise index, with a threshold fixed before Study O, classified 5 of 6 confident treatments (O 4/4); the threshold fell with training budget as N^−0.43 [−0.63, −0.22] (P 3/4; four times more training found an exploit missed at one million steps); the ordering transferred to A2C and DQN (Q 6/6). Simple cost-aware scores rank the arms equally, and Study R's prospective cost-transfer prediction failed (R 0/1). A reanalysis with the exploit's shape L(0.02) − L(2) confirms the orderings (`exploratory_claude_2026-10-16/reanalysis_shape/`).
- **Preventing the exploit without the ledger (Study S, 180 agents, 8/8 registered hypotheses supported).** The sign-paired reward clones the simulator before each step, steps the clone with the sign of the exogenous price change reversed, and trains on (r − r_twin)/2 − own cost. In a fair market it equals the booked P&L in expectation at every state, and it is the unique such estimator that also keeps a legitimate edge (`exploratory_claude_2026-10-16/nfl_training/NFL_THEORY.md`). In an edge market, PPO trained on the reported reward gambled and booked −6.5% of capital per week; with the sign-paired reward it never gambled and booked +22.6% (ledger training +18.1%), while a KL penalty toward a safe policy kept the gamble or lost over half the edge at every strength. In Study O's environment it removed the exploit's shape for PPO and A2C and was no worse than ledger training. DQN is not claimed (exploratory failure).
- **Model selection.** In an independent certification cohort, the winners selected on reported return (+108.8% and +104.3%) were certified worst in expected booked return within their ten-agent sets, with one joint 95% guarantee (Study M).
- **Historical settlement-component stress test (exploratory).** A proxy for the historical eSett volume term was below a modeled three-week deterrence buffer in **36.8%** of post-March-2025 FI/SE/NO zone-months versus **8.6%** before. It excludes the invoice component, minimum collateral and monitoring, and applying the historical formula to Finland after November 2024 and to Denmark is counterfactual; it is not an assessment of total collateral adequacy or conduct. The fee-free floor/terminal ratio was 1.62–1.99 across 24 zone-years.
- **Boundaries.** With realistic trading costs (Study H), inside gym-mtsim (Study G) and with equity-scaled positions (Study E), agents did not learn to gamble, or did so only weakly. Reported returns still exceed booked ones wherever a floor binds.
- **Prevalence.** One of the 18 most-starred public RL trading environments meets both enabling conditions (Study D); 20 of 21 public electricity-RL environments have no capital account.
- **Exploratory frozen-controller audits.** Statistical incentive audits of the engine's 20 saved controllers were silent on the guarded engine (0/40 cases, no forgiven loss on any path) and certified the incentive for 10 of 20 controllers once the documented fault was restored (none under full charging); their learned components added nothing. Records: `exploratory_novelty_2026-10-15/`, `exploratory_novelty_2026-10-16/`.
- **Tools.** Reconciliation of reported against booked returns, a placebo-market test, simultaneous cost-based certificates, the sign-paired reward and an evaluation checklist (`CHECKLIST.md`).

All registered hypotheses are reported: B–K supported 24/46, N 6/9, O 4/4, P 3/4, Q 6/6, R 0/1 and S 8/8. These are shared-outcome decision counts, not independent replications or family-wise guarantees. The completed main-study cohort has **1,164 distinct agents** (B–K 594; L 20; N 40; O 80; P 90; Q 100; R 60; S 180), excluding the separate **280-agent Study A**. C's five snapshots per training path do not add agents, M reuses L's checkpoints, and P's 170-row CSV includes O's 80 reused runs.

## Studies and where to find them
| Study | Question | Folder | Pre-registration | Aggregated results |
|---|---|---|---|---|
| Engine audit | Which evaluation shortcuts decide the verdict on frozen controllers? | `engine/`, `audit/`, `results/` | `docs/PREREGISTRATION.md` | `results/` |
| A | When do simulator shortcuts reverse "learning beats the rule" (three environment designs)? | `GENERALIZATION_STUDY_2026-09-29/` | `PREREGISTRATION_A.md` | `results/` there |
| B | Do agents learn to gamble under a zero floor? | `LIABILITY_STUDY_2026-09-29/` | `PREREGISTRATION_B.md` | `results/` |
| C | Does the gambling grow with training (fresh seeds)? | `LIABILITY_STUDY_2026-09-29/scaling/` | `PREREGISTRATION_C.md` | `scaling/results/` |
| D | How common are the enabling conditions in public trading environments? | `LIABILITY_STUDY_2026-09-29/env_audit/` | `PREREGISTRATION_D.md` | `coding_d.csv` |
| E | Equity-scaled positions and the log reward | `LIABILITY_STUDY_2026-09-29/` | `PREREGISTRATION_E.md` | `results_e/` |
| F | Well-posed learning (PPO, equity as the only input) | `LIABILITY_STUDY_2026-09-29/` | `PREREGISTRATION_F.md` | `results_f/`, pilot `results_pilot_f/` |
| G | A public simulator (gym-mtsim) | `LIABILITY_STUDY_2026-09-29/mtsim_study/` | `PREREGISTRATION_G.md` | `results_g/` |
| H | Realistic trading costs | `LIABILITY_STUDY_2026-09-29/` | `PREREGISTRATION_H.md` | `results_h/` |
| I | Other learners (A2C, DQN) | `LIABILITY_STUDY_2026-09-29/` | `PREREGISTRATION_I.md` | `results_i/`, pilot `results_pilot_i/` |
| J | Other markets (Finland, southern Norway) | `LIABILITY_STUDY_2026-09-29/` | `PREREGISTRATION_J.md` | `results_j/` |
| K | Patience: does the gambling region widen with the discount factor? | `LIABILITY_STUDY_2026-09-29/` | `PREREGISTRATION_K.md` | `results_k/` |
| L | Reward correction with matched continued trading | `revision_experiments/2026-10-10/` | `continuation_protocol.txt`, `prospective_manifest.json` | `continuation_experiment/` |
| M | Independent model selection and certification on the L checkpoints | `revision_experiments/2026-10-10-selection/` | `frontier_certification/prospective_manifest.json` | `frontier_certification/`, `VALIDATION.json` |
| N | Settlement lag (zombie window) and terminal limited liability | `ACCOUNTING_OPTIONS_2026-10-14/study_n/` | `PREREGISTRATION_N.md`, `PREREGISTRATION_HASHES_N.txt` | `runs_n.csv`, `VERDICTS_N.md` |
| O | Does a pre-training learnability index predict whether PPO learns the exploit (partial liability)? | `ACCOUNTING_OPTIONS_2026-10-14/study_o/` | `PREREGISTRATION_O.md`, `PREREGISTRATION_HASHES_O.txt` | `runs_o.csv`, `VERDICTS_O.md` |
| P | Does the training budget shift the learnability threshold as 1/√N? | `ACCOUNTING_OPTIONS_2026-10-14/study_p/` | `PREREGISTRATION_P.md`, `PREREGISTRATION_HASHES_P.txt` | `runs_p.csv`, `VERDICTS_P.md` |
| Q | Does the learnability index transfer to A2C and DQN? | `ACCOUNTING_OPTIONS_2026-10-14/study_q/` | `PREREGISTRATION_Q.md`, `PREREGISTRATION_HASHES_Q.txt` | `runs_q.csv`, `VERDICTS_Q.md` (6/6 supported) |
| R | Does the frozen exact one-step score transfer across fees at the same charge share? | `ACCOUNTING_OPTIONS_2026-10-14/study_r/` | `PREREGISTRATION_R.md`, `PREREGISTRATION_HASHES_R.txt` | `runs_r.csv`, `analysis_r.json`, `VERDICTS_R.md` (R1 not supported) |
| S | Does a sign-paired reward prevent the exploit without the ledger, where a KL penalty toward a safe policy cannot? | `ACCOUNTING_OPTIONS_2026-10-14/study_s/` | `PREREGISTRATION_S.md`, `PREREGISTRATION_HASHES_S.txt` | `runs_s_A.csv`, `runs_s_B.csv`, `VERDICTS_S.md` (8/8 supported) |
| Settlement | The ladder and collateral in twelve Nordic zones (exploratory) | `ACCOUNTING_OPTIONS_2026-10-14/` | not pre-registered | `ladder12.csv`, `coverage12_*.csv`, `frontier12_*.csv`, `tails12.csv` |

Studies L and M are called "revision experiment L" and "revision audit M" in their frozen records. They were frozen before their runs but registered after the scorecard of Studies B–K, and are reported separately from its 46 hypotheses. Study N was registered on 14 October 2026; its `RUN_LOG_N.md` is append-only, and the registered version is the prefix before the line `## After hashing`.

Each study folder of A–K also holds `PREREGISTRATION_HASHES.txt` (SHA-256 of every registered file, with UTC timestamps) and `RUN_LOG.md` (launches, interruptions, relaunches and changes of job order). Theory solvers: `dp_theory.py` (Denmark), `dp_frictions.py` (Study H), `dp_j.py` (Study J), `dp_full.py` (full training distribution, all zones; supersedes the 3,000-sample solves, see `RUN_LOG.md`), `theory_k.py` and `reflection_barrier.py` (closed-form barrier), and `study_n/dp_n.py`, `dp_n2.py` (Study N). Best-moment theorem: `make_best_moment_figure.py` (writes `results_k/best_moment.csv` and the figure; `best_moment.py` is its earlier 1,000-draw version) and `best_moment_zones.py` (twelve zones). Other theory checks: `action_gap.py`, `false_profit_bound.py`, `partial_liability.py`, tables by `make_theory_tables.py`. Placebo-market test: `placebo_market.py`. Exploratory analyses are in `review_rigor.py` and `explore_i.py`.

## Repository layout
```
README.md, DESCRIPTION.md        overview; short description and repository metadata
CITATION.cff                     citation metadata
CHECKLIST.md                     evaluation checklist for RL trading agents, their simulators and settlement rules
LICENSE, LICENSE-docs.md         MIT (code); CC BY 4.0 (docs, results, derived tables)
THIRD_PARTY_LICENSES.md          sources, licenses and attribution for all third-party material
requirements*.txt                standalone analyses; engine stack; Studies A-S; Study G (gym-mtsim)
tools/verify_studies.py          registered hashes; completed counts and headlines; theory/settlement checks
tools/verify_accounting_revision.py   exact rational-arithmetic checks of the reflection identity and reward correction
tools/verify_reflected_control_revision.py   numerical check of the reflected-control candidate (partial charges)
tools/verify_release.py          recomputes every headline number of the engine audit
tools/ledger_check.py            standalone reported-vs-booked-P&L reconciliation
GENERALIZATION_STUDY_2026-09-29/ Study A: environments, runner, aggregation, Energinet data and panels, results
LIABILITY_STUDY_2026-09-29/      Studies B-K: environments, runners, aggregation, theory solvers, results
revision_experiments/            Study L (2026-10-10/, with all 20 checkpoints) and Study M (2026-10-10-selection/)
ACCOUNTING_OPTIONS_2026-10-14/   option-ladder theory checks, historical settlement-component analysis, Studies N-S
exploratory_codex_2026-10-15/    independent count/statistical audit and guarded completed-Q diagnostics
exploratory_novelty_2026-10-15/, exploratory_novelty_2026-10-16/   exploratory scores, witnesses and frozen-controller audits
exploratory_claude_2026-10-16/   sign-paired reward development (nfl_training/) and the exploit-shape reanalysis
engine/, audit/                  engine audit: simulator source (patched audit copy) and audit code
analysis/, code_audit/           cross-market liquidity test; code audit of 21 electricity-RL repositories
literature/                      pilot literature survey coding (no PDFs)
results/                         engine-audit results
data/                            Energinet data (CC BY 4.0), engine liquidity series, derived tables;
                                 fetch_esett.py downloads the eSett files of all twelve zones (not redistributed)
docs/                            engine-audit protocols, pre-registration and methods
```

## Reproducing
Python 3.10. CPU only.

**Verification without training (minutes).**
```
pip install -r requirements-studies.txt
python tools/verify_studies.py      # registered hashes, counts and completed headline checks
python tools/verify_accounting_revision.py
python tools/verify_reflected_control_revision.py
python tools/verify_release.py      # headline numbers of the engine audit
python tools/ledger_check.py results/ledger_examples/L3_marl_seed7_DK1.csv.gz   # a flagged run
```

**Theory and aggregation from the shipped per-run results.** In `LIABILITY_STUDY_2026-09-29/`:

Run commands that regenerate registered results only in a disposable copy. Preserve released scripts, pre-registrations, verdicts and `runs_*.csv` byte for byte; the verifier and the new independent audit read them without rewriting them.
```
python dp_theory.py                 # optimal policies, Denmark (Section 3 of the paper)
python aggregate_b.py               # and aggregate_e.py, aggregate_f.py, aggregate_h.py, aggregate_i.py, aggregate_j.py, aggregate_k.py
python dp_full.py --gammas 0.95 0.98 0.99 0.995   # optimal gambling regions on the full training distribution (all zones)
python make_best_moment_figure.py   # Theorem 1, four zones: results_k/best_moment.csv and the figure
python placebo_market.py --demo     # the placebo-market test on a floor and a full-liability agent
```
Theory checks of Supplementary Section S11 (exploratory; the eSett files and the Study J panels are needed, see below):
```
python best_moment_zones.py         # Theorem 1 in all twelve Nordic zones
python action_gap.py                # action gap of the DP in every training condition; DP policy at gamma 0.99
python false_profit_bound.py        # the cap of Theorem 2 against fixed, threshold, closed-form and learned policies
python partial_liability.py         # the gambling region when a share of every forgiven loss is booked (Corollary 3)
python make_theory_tables.py        # outputs/tables/tab_zones.tex, tab_cap.tex, tab_actiongap.tex
```
Figure and table scripts (`make_*.py`) write to `outputs/` inside each study folder.

**Studies L and M** (see the `README.txt` in each folder):
```
cd revision_experiments/2026-10-10 && python aggregate_continuation.py        # Study L, all 20 agents
python revision_experiments/2026-10-10-selection/summarize_frontier.py          # Study M, from the repository root
python revision_experiments/2026-10-10-selection/verify_frontier.py
```

**Accounting options, Study N and the settlement analysis.** See `ACCOUNTING_OPTIONS_2026-10-14/README.md`. In short:
```
cd ACCOUNTING_OPTIONS_2026-10-14
python verify_theory.py                         # exact checks: ladder, factor two, doubling counterexample, non-identifiability
python study_n/portable.py aggregate_n.py       # Study N verdicts from the shipped per-run results
python study_o/aggregate_o.py                   # Study O verdicts (learnability index); study_o/dp_o.py and calibrate_o.py give the predictions
python study_p/aggregate_p.py                   # completed P verdicts (90 new runs + reused O80); disposable copy only
python study_s/aggregate_s.py                   # Study S verdicts (sign-paired reward); disposable copy only
python study_s/make_studyS_outputs.py           # the Study S figure
python make_tables.py                           # the paper's settlement tables from the shipped outputs
```

**Independent exploratory audit (from the repository root, seconds).**
```
python exploratory_codex_2026-10-15/statistical_audit.py --seed-fit-bootstrap 199
```
This writes only new exploratory output and independently checks the pre-Q cohort, O/P point fits, baselines and seed sensitivity. The completed-Q diagnostic is `python exploratory_codex_2026-10-15/diagnose_completed_q.py`; its separate output covers the 924-agent cohort through Q. Completed R adds 60 agents, bringing the main-study count to 984. Q2 paired-seed exploratory intervals remain positive; exact common-arm permutations give p = 0.025 for each learner under that reference distribution. Training visitation and critic gaps were not serialized and cannot be reconstructed from greedy probe summaries. See `exploratory_codex_2026-10-15/STATISTICAL_CHANGES.md`.

The DP-free fixed-reference rollout analysis has completed: see `exploratory_codex_2026-10-15/model_free/EXPLORATORY_RESULTS.md` and its reproduction README. Across 26 existing conditions, the primary surrogate's DP-index rank correlation was 0.950 overall but 0.552 among legacy conditions. It matched O's 58/70 run classifications, with wide Monte Carlo uncertainty; three of seven O arms had intervals crossing the point threshold, and the transferred threshold missed both 4M arms. This demonstrates an affordable fixed-policy diagnostic, not an interchangeable estimate of the optimal gap or a confirmed scalable predictor. All references and failures are reported; that method did not lead to a registered study. The later exact one-step method and separately registered Study R are described below; they do not retrospectively rescue this failed design.

### Exploratory record: exact witnesses and Study R

`exploratory_novelty_2026-10-15/estimator_audit/` contains an explicitly exploratory
exact empirical one-step audit. It integrates every TRAIN magnitude/cap row and
both fair signs, so it needs neither a DP nor Monte Carlo. Its numerical analysis
took about 0.40 seconds. This is exact conditional on the finite empirical model,
not a guarantee about future markets or arbitrary black-box simulators. The
seven-probe predictor and the executable dense-grid witness are distinct.

The geometric calibration cutoff is 0.089669%, within a very wide separating
interval. Historical acquisition transfer remains mixed: O gives 6/7 arm gaps but
46/70 raw-run endpoints, and alpha .9 is misclassified. All failures and source
revisions remain visible in its `EXPLORATORY_RESULTS.md`.

`exploratory_novelty_2026-10-15/state_audit/` extends the feasible-policy question
to Study N's ledger/clock states. Positive one-action witnesses occur at all
24 insolvent lag probes, versus none in the 24 matching booked controls. Terminal
immediate/deadline policies give 69/108 and 78/108 witnesses; their booked controls
give none. These are empirical-model incentive witnesses, not transferred O
thresholds or proofs of learned optimal actions. A nonpositive witness is
inconclusive about multi-step exploitation.

**Study R is complete: all 60 PPO agents finished and the automatic registered
aggregation returned R1 NOT SUPPORTED.** New seeds 60–64, DK1/DK2, fees .40/1.50
EUR/MWh and alpha 0/.8/1 were fixed before training. Each run completed 1,007,616
actual steps for the requested 1M budget. The frozen exact-score convention
predicted acquisition at alpha .8 under the lower fee and non-acquisition under
the higher fee; both retain a positive one-action incentive. All four required
primary gates are reported:

| R1 gate | Full-cohort result | Met? |
|---|---|---|
| Low-fee treatment-control gap >=4 | 4.500 | Yes |
| High-fee treatment-control gap <4 | 4.475 | No |
| Mean paired-seed interaction >=4 | .025 | No |
| Exact one-sided seed-cluster sign p<=.05 | .8125; 2/5 positive | No |

The descriptive paired-seed bootstrap interval for the interaction is
[-4.475,5.125]; it does not replace the sign test. Full-forgiveness anchors chose
16 in 10/10 runs at both fees. At alpha .8 the mean leverages were 7.600/5.600
(4/10 and 3/10 maximal); matched booked controls averaged 3.100/1.125 (1/10 and
0/10 maximal). The raw fee difference therefore does not establish the registered
treatment-control interaction. The operational gap thresholds do not prove
population equivalence or absence of gambling.

The frozen protocol, predictions, full CSV, analysis and verdict remain in
`ACCOUNTING_OPTIONS_2026-10-14/study_r/`; the launch logs are retained. Its 60
agents are included in the completed 984-agent main cohort. This failed
acquisition prediction supplies no superiority claim over simple fee-aware
scores and does not alter the separately confirmed accounting witnesses.
Preserve all registered records and outputs byte for byte; rerun result writers
only in a disposable copy.

**Retraining (hours to days on a multi-core CPU).** Each runner is resume-safe and skips finished jobs.
```
cd GENERALIZATION_STUDY_2026-09-29 && python run_study.py --phase rules && python run_study.py --phase rl --workers 6
cd LIABILITY_STUDY_2026-09-29
python run_b.py --phase rules && python run_b.py --phase rl --workers 6   # Study B (the study relaunched with run_b_order.py; see RUN_LOG.md)
python scaling/run_c.py --workers 7                                       # Study C
python run_e.py --workers 7 ; python run_f.py --workers 7 ; python run_h.py --workers 7 ; python run_i.py --workers 7
python run_k.py --workers 7                                               # Study K (discount factor)
cd ../ACCOUNTING_OPTIONS_2026-10-14 && python study_n/portable.py run_n.py --workers 7   # Study N (about 75 minutes with 7 workers)
```
Study B took about 8 hours, Study I about 2 hours and Study K about 3 hours with 7 workers on an 8-core CPU. Study L: `python revision_experiments/2026-10-10/run_portable.py --workers 4` (preserve the released jobs first; see its README).

**eSett data (Study J and the settlement analysis).** eSett data are not redistributed:
```
python data/fetch_esett.py                                   # from the repository root: twelve zones, 4 March 2025 to 27 September 2026; records content hashes
python ACCOUNTING_OPTIONS_2026-10-14/fetch_esett_pre.py      # the same zones, November 2021 to March 2025
cd LIABILITY_STUDY_2026-09-29
python get_esett_j.py && python build_panel_j.py && python dp_j.py && python run_j.py --workers 7
python ../tools/verify_studies.py                            # reports revised-source panel hashes explicitly
```
eSett revises recent months. `data/esett/CONTENT_SHA256.json` records the hash of each file as first downloaded; a later download of a revised month is reported as differing.

**Study G (gym-mtsim)** needs its own environment, because gym-mtsim's bundled data load only with pandas 2.0.x:
```
cd LIABILITY_STUDY_2026-09-29/mtsim_study
python -m venv .venv && .venv\Scripts\pip install -r ../../requirements-mtsim.txt
.venv\Scripts\python run_g.py --workers 7 && .venv\Scripts\python aggregate_g.py
```

**Engine audit (hours).** The engine source is in `engine/`; its datasets and trained checkpoints (about 57 GB) are not included. `pip install -r requirements-engine.txt`, then see `docs/RUNNING_THE_CAMPAIGN.md`. `audit/setup_audit_engine.py` copies the engine and applies the three-block patch in `audit/engine_patch/`, which bypasses the train/eval contract-hash check only when `SIM_AUDIT_ALLOW_PROTOCOL_MISMATCH=1` and applies config overrides from `SIM_AUDIT_CFG_OVERRIDES`.

## Pre-registration
All pre-registrations are internal: written and hashed before the corresponding test-period evaluation, not deposited with an independent registry. Their records disclose smoke tests on training data, design changes made before registration (for Study N: the fee and claim reporting, after a failed first pilot), a change of job submission order during Study B, a configuration-selection rule for Study I written after its pilot had been seen, aggregation scripts written after the runs for Study C, and, in the engine audit, timestamp corrections and one post-hoc correction of an aggregation script. `tools/verify_studies.py` checks every registered file against its recorded hash. The settlement analysis is exploratory.

## Exploratory record: cost-aware comparators

The exploratory comparison in `exploratory_novelty_2026-10-15/cost_baselines/`
uses thirteen fixed scores, the original ten historical labels and common
balanced-accuracy calibration. It reports all calibration-optimal cutoff sets,
arm-level results separately by learner/budget, the bundled H cost intervention,
six-setting and revised-FI sensitivities, and all failures. Dose-aware simple
scores tie the DP/exact indices' ranking in the two-class O/P/Q cohorts. H already
separates under nominal fee. The completed studies do not establish superior
acquisition prediction by the diagnostic.

```powershell
python exploratory_novelty_2026-10-15/cost_baselines/compare_completed_cost_baselines.py
python exploratory_novelty_2026-10-15/matched_moments/exact_tail_counterexample.py
```

The second command exactly verifies a bounded pair with identical fee, dose,
equity, variance and gross/net absolute-payoff means but opposite one-action
accounting-incentive signs. Full booking removes the positive proxy reward.
This explains a concrete information requirement for the feasible-policy audit;
it is not a result about learner acquisition. No new agents are trained. Neither
script uses Study R outcomes or changes its frozen records. Manuscript table generation is
`tools/generate_cost_baseline_tables.py` in the companion manuscript repository.

## Exploratory record: adaptive witness synthesis

The exploratory audit now extends the isolated-action diagnostic to executable
ledger/clock feedback policies. `exploratory_novelty_2026-10-15/adaptive_witness/`
contains the assumptions, proofs and exact rational counterexamples. A negative
immediate prefix can reach a profitable continuation: at equity .7, cost .4 and
fair gross +/-1, a two-step adaptive policy earns charged .0985 while booking
-.598; isolated action and constant two-step trading both lose. A second example
needs a low-equity gate before insolvency. The terminal-account clock compiler
places each fixed local controller at the deadline and reuses its certificate
across outer horizons by deterministic discount scaling. It does not optimize
all policies or transfer to time-varying markets or information gained while flat.

The accounting identity gives `J_alpha = E[(1-alpha)*F-C]` and
`J_book = -E[C]` under predictable no-edge gross payoffs. Bounded executed costs
give the residual a known lower endpoint even with integrable unbounded prices;
independent upper-capped confirmation can certify a positive expectation or
abstain. Full booking analytically removes positive expected monetary reward
under these assumptions. Search, martingale controls, clipping and concentration
are established tools; the proposed contribution is the accounting-specific
construction and certificate. Residual variance can increase, and failure to
find a witness is not safety or a learner-acquisition prediction.

The fixed-budget empirical families in `adaptive_witness_empirical/` preserve all
failures. The full-clock family confirms 38/70 selected state-policy pairs, but
structured, random, barrier and forced-prefix-myopic methods each confirm 8/14
states; none adds a state beyond the exact isolated-action reference's 9/14.
A separately planned selected-case follow-up treats those outputs as development
data. It freezes one DK1 96-step controller and a search-selected cap before
524,288 fresh confirmation paths. Its one-sided conditional 99% charged lower
bound is **11.09292566 EUR**, with exact expected booked return bounded by the
negative mandatory initial fee, **-3.33553183 EUR** (rounded). Every nonflat
isolated action at that state is negative (best: -.01433268 EUR). This establishes
one selected empirical-model multi-step capability, without revising the 38/70
benchmark or establishing superiority over random or constant multi-step search.

`adaptive_witness_heavytail/` validates the certificate on two constructed Pareto
laws. At alpha .8, infinite-variance shape 1.5 is confirmed with lower mean
.02335552 versus true .04509525; shape 2.5 is inconclusive at that charge, with
true mean -.07454469. The two-law error budget is .05 and covers each fixed
action's whole charge interval, separately from the empirical families. These
are generative mathematical cases, not independent markets or training results.

```powershell
python exploratory_novelty_2026-10-15/adaptive_witness/exact_adaptive_benchmark.py
python exploratory_novelty_2026-10-15/adaptive_witness_empirical/selected_case_followup.py
python exploratory_novelty_2026-10-15/adaptive_witness_heavytail/bounded_cost_certificate.py
```

The empirical README also lists all three earlier benchmark commands, plans,
selection records, bounds and failures. These method analyses did not use R's
interim endpoints. **Completed R1 is not supported**, and its 60 agents are now
included in the 984-agent main-study count; the policy-specific certificates
answer a separate incentive question. Current build QA is reported in the
companion manuscript README; earlier page and abstract counts do not describe
the latest method revision. Changes are uncommitted.

## Exploratory record: conditional-kernel accounting audit

The 16 October exploratory extension is in
`exploratory_novelty_2026-10-16/predictable_accounting_audit/`. It replaces each
floor injection or terminal debt increment by its exact conditional accounting
drift. Known finite uniform conditional stop-loss and mean-cost bounds make the
whole-path estimator bounded and unbiased without upper clipping, even when
realized prices and costs are unbounded. This requires conditional integration
over every piece of information the controller observes; a sample-only simulator,
an unconditional row average under revealed risk, or an uncalibrated fitted critic
does not provide that oracle. Finite moments without known uniform bounds are
insufficient. Predictable compensation and the gain/loss ratio are established
tools, credited to Henderson--Glynn and Bernardo--Ledoit/Biagini--Pinar; the
contribution is the explicit floor/terminal envelope and its executable audit.

A finite uniform ratio `rho >= h_t(0)/c_t` gives the model-specific all-policy
bound `J_alpha <= ((1-alpha)*rho-1)*E[B]`, for charges in `[0,1]` under no-edge
gross payoffs. It proves nonpositive expected charged reward when
`alpha >= 1-1/rho`. Below this threshold, a positive charged lower bound `ell`
also yields `J_book <= -ell/((1-alpha)*rho-1) < 0`, without a mandatory first trade
or a second cost test. Zero-cost positive-stop-loss actions preclude finite
`rho`. The global threshold is sharp only for the exact cash-independent action
laws with the offending action admissible at zero; a conservative conditional
envelope supplies sufficiency. This is a declared-kernel result, not future-market
coverage or proof that an unconfirmed controller class is safe.

The completed comparisons use five representations: raw reward, capped raw,
capped realized-cost residual, capped mean-cost residual, and uncapped predictable
drift. Each comparison equalizes logical simulator-transition budgets and shares
paths; exact conditional integration adds arithmetic work.

| Separately allocated family | Predictable result | Comparator and limit |
|---|---|---|
| 720 bounds on TEST row kernels excluded from audit development | 36/72 selected; 33/72 independently fixed-controller slots | Both capped residuals: 24/72 in each arm. All predictable discoveries lie in the 12/24 conditions already positive under exact isolated-action integration; no extra condition discovered |
| 45 bounds on nine fixed development policies, 32,768 confirmation paths each | 3/9 positive, including feedback and simple deadline policies | All four comparators abstain. A larger 524,288-path capped-residual comparison also confirms a simple delayed policy, so feedback is not necessary in this case |
| 32 bounds on 16 Pareto cases | 3/16 positive | Capped mean-cost residual confirms the same 3/16; predictable sample variance is higher in 4/16; all terminal cases abstain |

Each row has its own 95% simultaneous lower-bound allocation, not a paper-wide
guarantee. The held-out cases are two existing markets' empirical row partitions,
not independent new markets; the three fixed neutral controllers are distinct
controllers, not replications of one policy. Pareto cases include infinite price
variance and unbounded proportional costs, with exact analytic moment access.
The eight high-charge controls satisfy their analytic all-policy repair bounds.
Retained ties, variance increases and non-discoveries rule out universal variance,
power, optimizer or acquisition superiority.

Exact scripts retain 32 original finite-law cases and 120 additional optional-trade
harm checks. Independent read-only verification in
`exploratory_novelty_2026-10-16/predictable_verification/` reports **552/552 checks
passing** (the prior 501 plus 51 new checks). The original failed serialization
attempts, frozen input manifests and scalar-serialization-only replays remain
available; replay does not add independent observations. Recorded plans, raw
results, registered study artifacts and earlier audit records are preserved.

## Not included
- Trained checkpoints and raw per-run logs of Studies A–K and N–R (size). Per-run summaries are in each completed study's results folder (`jobs/` and `runs_*.csv`). The 20 Study L checkpoints are included.
- Engine datasets and checkpoints (about 57 GB) and raw engine logs (about 105 GB).
- Raw eSett Open Data and the Study J panels derived from them (no explicit redistribution license); `data/fetch_esett.py` and `ACCOUNTING_OPTIONS_2026-10-14/fetch_esett_pre.py` re-download them.
- Third-party source code from the audits (copyright); the scan scripts regenerate the excerpts from pinned commits.
- The paper PDFs of the literature survey (copyright).
