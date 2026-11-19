# Reproducing the results

How to verify, re-aggregate and re-train, kept from the repository's earlier README and extended, with the exploratory records that the supplement cites.

## Reproducing
Python 3.10. CPU only.

**Verification without training (minutes).**
```
pip install -r requirements-studies.txt
python -m unittest discover -s tools -p test_verify_studies.py
python tools/verify_studies.py      # registered hashes, counts and completed headline checks
python tools/verify_accounting_revision.py
python tools/verify_reflected_control_revision.py
python tools/verify_release.py      # headline numbers of the engine audit
python tools/ledger_check.py results/ledger_examples/L3_marl_seed7_DK1.csv.gz   # a flagged run
```

The historical B-K manifest includes trailing explanatory notes on three entries.
The verifier checks these entries too. Its `aggregate_c.py` entry refers to the
byte-identical script now shipped at `LIABILITY_STUDY_2026-09-29/scaling/aggregate_c.py`;
the recorded manifest and digest are unchanged.

**Before committing a release.** Review intended additions and deletions first.
The `.gitattributes` exceptions preserve registered and exploratory records byte
for byte. After staging the intended files, run `git add --renormalize .` so
previously tracked files also acquire the current attributes; inspect the staged
diff again before committing. A checkout of the staged snapshot must pass the
verifier above, not just the working copy. Do not blanket-stage deletions that
have not been reviewed.

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
python study_t/aggregate_t.py --print           # Study T verdicts (twin-identified reward, ORPO); prints, writes nothing
python study_u/aggregate_u.py --print           # Study U verdicts (settlement lag, terminal liability); prints, writes nothing
python study_v/aggregate_v.py --print           # Study V verdicts (ruin penalty, mean-variance, PMR, audit); prints, writes nothing
python study_v/make_studyV_outputs.py           # the Study V figure and tables
python study_x/aggregate_x.py --print           # Study X verdicts (two-write floor); prints, writes nothing
python study_x/make_studyX_outputs.py --tables DIR --figures DIR   # the Study X figure and tables (default: the sibling manuscript repository)
python study_y/aggregate_y.py --print           # Study Y verdicts (A2C); prints, writes nothing
python study_y/make_studyY_outputs.py --tables DIR --figures DIR   # the Study Y figure and tables (default: the sibling manuscript repository)
python study_z/test_envs_z.py; python study_zb/test_envs_zb.py      # exact checks of the split-account and isolated-margin proposition
python study_z/aggregate_z.py --print; python study_zb/aggregate_zb.py --print   # Study Z per-arm tables (Parts A and B); print, write nothing (the verdicts are in VERDICTS_Z.md and VERDICTS_ZB.md; without --print the scripts rewrite them: disposable copy only)
python study_zb/make_tab_hidden.py --out DIR                        # the main-text table of Study Z (default: the sibling manuscript repository's sections/)
python study_rs/portable_rs.py test_envs_rs.py; python study_rs/portable_rs.py aggregate_rs.py --print   # Study RS: exact checks; registered analysis, printed (writes nothing)
python study_rs/make_studyRS_outputs.py --tables DIR --figures DIR   # the Study RS figure and tables (default: the sibling manuscript repository)
python ../exploratory_claude_2026-11-05/rs_reanalysis.py --print; python ../exploratory_claude_2026-11-05/bracket_limits.py --print   # exploratory; print, write nothing
python study_ext/aggregate_ext.py --print; python study_ft/aggregate_ft.py --print   # Studies EXT and FT per-arm tables; print, write nothing (the verdicts are in VERDICTS_EXT.md and VERDICTS_FT.md; without --print the scripts rewrite them: disposable copy only)
python study_ext/fetch_data.py                  # the example price file of Gym-Trading-Env, by hash; only needed to re-run Study EXT
python twin_reward/test_reward_diagram.py       # the reward diagram: 88,702 exact checks in rational arithmetic
python study_w/aggregate_w.py --results ../study_v/results_v --demo   # Study W's analysis on Study V's runs (software check; writes nothing)
python twin_reward/test_twin_reward.py          # tests of the TIR/SPR estimators and the Gymnasium wrapper
python make_tables.py                           # the paper's settlement tables from the shipped outputs
```

**Independent exploratory audit (seconds).**
```powershell
# Run from the repository root:
python exploratory_codex_2026-10-15/statistical_audit.py --seed-fit-bootstrap 199
```
This independently checks the pre-Q cohort, O/P point fits, baselines and seed sensitivity. The completed-Q diagnostic is `exploratory_codex_2026-10-15/diagnose_completed_q.py`; its separate output covers the 924-agent cohort through Q. Completed R adds 60 agents, bringing the main-study count to 984. Q2 paired-seed exploratory intervals remain positive; exact common-arm permutations give p = 0.025 for each learner under that reference distribution. Training visitation and critic gaps were not serialized and cannot be reconstructed from greedy probe summaries. See `exploratory_codex_2026-10-15/STATISTICAL_CHANGES.md`.

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
# Run from the repository root:
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
# Run from the repository root:
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
the latest method revision.

## Studies added from 14 to 17 November 2026
These folders ran next to the repository and were copied in byte for byte; their scripts address each other by relative paths, so keep the folder names. Their registered analyses are re-run by the verifier in a scratch copy. To re-train, create the environments named in each protocol (gym-mtsim 2.0.0 for GC and DR, as for Study G; backtesting.py 0.6.6, backtrader 1.9.78.123 and vectorbt 1.0.0 for XE and the control), then:
```
cd study_gc_2026-11-14 && python test_envs_gc.py && python run_gc.py --workers 8 && python aggregate_gc.py && python verify_completed_gc.py
cd study_dr_2026-11-15 && python hash_prereg_dr.py --verify && python run_dr.py --workers 8 && python aggregate_dr.py && python verify_completed_dr.py
cd study_xe_2026-11-16 && powershell -File run_xe.ps1          # the registered commands; then python aggregate_xe.py
cd xe_patch_control_2026-11-17 && python check_control.py && python rl_engine_pilot.py --steps 500000 --seeds 301 302 303 304 305 306 307 308 --engines btpyfl btpy --workers 8 --levels 0 200 --tag _ctl --test-seed 20261211 --out results_ctl && python score_ctl.py && python verify_ctl.py
cd monitoring_law_2026-11-17 && python predict.py && python test_a_sim.py && python predict_v2.py && python test_a2_sim.py && python test_c_engine.py 8000 6 && python test_d_existing.py && python test_d4b_nordic.py
```
Run aggregators and test scripts in a disposable copy: they rewrite the stored outputs. The closed-form tests read the Nordic eSett files (download them as below). Each folder's README gives the details, the run times and every failure.
