# accounting-options-rl

**Simulators that forgive losses teach reinforcement-learning traders to gamble: theory, detection and removal, with an application to electricity trading.** Code, data pipelines, hashed protocols and results of the paper, submitted to *Engineering Applications of Artificial Intelligence*.

A simulator's accounting decides what happens to a loss beyond an agent's capital. Many trading simulators forgive it: they floor equity at zero, close the account and write the overshoot off, or leave the bankruptcy step unrewarded. That hands the agent a free option. This repository holds the evidence that reinforcement-learning (RL) agents learn to exercise it, the theory that prices it, a black-box method that reads a simulator's accounting from its reward, and training rewards that remove the exploit.

## Quick start (no training)
```
pip install -r requirements-studies.txt
python examples/quickstart.py      # reads a hidden floor from a toy simulator's reward; every printed number is asserted
python tools/verify_studies.py     # registered hashes, exact checks, re-run registered analyses, headline numbers (about 8 minutes)
```
`tools/verify_studies.py` ends with "all checks pass" (about 1,530 checks). On a fresh clone the three checks that need the non-redistributed eSett panels of Study J report "not shipped" instead. Further verification, re-aggregation and re-training: [`docs/REPRODUCING.md`](docs/REPRODUCING.md). Every study's results in detail: [`docs/RESULTS.md`](docs/RESULTS.md).

## The paper's claims and where the evidence is
| Claim (section of the paper) | Evidence | Where |
|---|---|---|
| Accounting rules form an ordered ladder of options, with closed forms and a time-step limit (3) | Exact checks of the theorems; exploratory tests of the closed forms in simulation and in unmodified backtesting.py | `ACCOUNTING_OPTIONS_2026-10-14/verify_theory.py`, `monitoring_law_2026-11-17/` |
| Agents learn the exploit where training reaches it (4) | Studies B to K and N in the electricity sleeve; EXT (Gym-Trading-Env); G, GC and DR (gym-mtsim); XE and the same-engine control (backtesting.py, backtrader); RE and O (the certificate) | `LIABILITY_STUDY_2026-09-29/`, `ACCOUNTING_OPTIONS_2026-10-14/study_*`, `study_gc_2026-11-14/`, `study_dr_2026-11-15/`, `study_xe_2026-11-16/`, `xe_patch_control_2026-11-17/` |
| The accounting can be read from the reward (5) | The reward-diagram reader and its exact checks; the gym-mtsim records; the engine's real clones | `ACCOUNTING_OPTIONS_2026-10-14/twin_reward/`, `exploratory_line_2026-11-07/`, `exploratory_claude_2026-11-06/engine_scan/` |
| Rewards from a sign-reversed clone step remove the exploit (6) | Studies S to Z, RS and FT; the twin-reward package with its tests | `ACCOUNTING_OPTIONS_2026-10-14/study_s/` to `study_ft/`, `ACCOUNTING_OPTIONS_2026-10-14/twin_reward/` |
| An engine audit and an acceptance check (7) | The engine's source, the audit code and results; the reconciliation tool | `engine/`, `audit/`, `results/`, `tools/verify_release.py`, `tools/ledger_check.py` |

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
| S | Does a sign-paired reward meet the registered prevention criteria, compared with three tested KL strengths? | `ACCOUNTING_OPTIONS_2026-10-14/study_s/` | `PREREGISTRATION_S.md`, `PREREGISTRATION_HASHES_S.txt` | `runs_s_A.csv`, `runs_s_B.csv`, `VERDICTS_S.md` (8/8 supported) |
| T | Do sign-paired and twin-identified rewards replicate on a new market draw and in two further settings, compared with ledger training and five ORPO strengths? | `ACCOUNTING_OPTIONS_2026-10-14/study_t/` | `PREREGISTRATION_T.md`, `PREREGISTRATION_HASHES_T.txt` | `runs_t.csv`, `VERDICTS_T.md` (16/16 supported) |
| U | Do SPR and TIR prevent the settlement-lag and terminal-liability exploits of Study N? | `ACCOUNTING_OPTIONS_2026-10-14/study_u/` | `PREREGISTRATION_U.md`, `PREREGISTRATION_HASHES_U.txt` | `runs_u.csv`, `VERDICTS_U.md` (6/6 supported) |
| V | Do a ruin penalty, a quadratic-utility ("mean-variance") reward and a clone-free pooled-magnitude reward (PMR) match the twin rewards, and can twin estimates audit a trained agent without the ledger? | `ACCOUNTING_OPTIONS_2026-10-14/study_v/` | `PREREGISTRATION_V.md`, `PREREGISTRATION_HASHES_V.txt` | `runs_v.csv`, `VERDICTS_V.md` (14/17 supported; VA3, VA5, VC6 not) |
| W | Can the TIR audit choose the ruin-penalty strength, in two settings outside Study V's grid? | `ACCOUNTING_OPTIONS_2026-10-14/study_w/` | `PREREGISTRATION_W.md`, `PREREGISTRATION_HASHES_W.txt` | `runs_w.csv`, `VERDICTS_W.md` (5/6 supported; W4 not) |
| X | Do SPR and TIR still prevent the exploit when the reward is a two-write (not single-floor) censor? | `ACCOUNTING_OPTIONS_2026-10-14/study_x/` | `PREREGISTRATION_X.md`, `PREREGISTRATION_HASHES_X.txt` | `runs_x.csv`, `VERDICTS_X.md` (6/6 supported) |
| Y | Do SPR and TIR transfer to a second learner (A2C), in the single-floor and two-write sleeves? | `ACCOUNTING_OPTIONS_2026-10-14/study_y/` | `PREREGISTRATION_Y.md`, `PREREGISTRATION_HASHES_Y.txt` | `runs_y.csv`, `VERDICTS_Y.md` (12/13 supported; YC1 not) |
| Z, Part A | Does a ruin penalty measured on the portfolio still work when each account has its own floor, and do SPR and TIR? | `ACCOUNTING_OPTIONS_2026-10-14/study_z/` | `PREREGISTRATION_Z.md`, `PREREGISTRATION_HASHES_Z.txt`, `THEORY_Z.md` | `runs_z.csv`, `VERDICTS_Z.md` (11/13 supported; ZA10, ZB3 not) |
| Z, Part B | The same when the floors are hidden from the trainer (isolated margin), and can a ledger-free finder find them? | `ACCOUNTING_OPTIONS_2026-10-14/study_zb/` | `PREREGISTRATION_ZB.md`, `PREREGISTRATION_HASHES_ZB.txt`, `THEORY_ZB.md`, `ERRATA_ZB.md` | `runs_zb.csv`, `VERDICTS_ZB.md` (11/13 supported; ZIB2, ZIB3 not) |
| RS | Do the twin rewards hold with real signs? A stress test on one zone and a reused test panel | `ACCOUNTING_OPTIONS_2026-10-14/study_rs/` | `PREREGISTRATION_RS.md`, `PREREGISTRATION_HASHES_RS.txt` (disclosed run-log digest defect) | `runs_rs.csv`, `VERDICTS_RS.md` (6/8 supported; RS3, RS6 not), `independent_verification/` |
| EXT | Does the exploit, and do the remedies and the reading, carry over to an unmodified public simulator (Gym-Trading-Env)? | `ACCOUNTING_OPTIONS_2026-10-14/study_ext/` | `PREREGISTRATION_EXT.md`, `PREREGISTRATION_HASHES_EXT.txt`, `ERRATA_EXT.md` | `runs_ext.csv`, `VERDICTS_EXT.md` (16/17 supported; EA9 not) |
| FT | What do agents learn where a sign query cannot see the gross (unrealized bucket, carried deficit)? | `ACCOUNTING_OPTIONS_2026-10-14/study_ft/` | `PREREGISTRATION_FT.md`, `PREREGISTRATION_HASHES_FT.txt` | `runs_ft.csv`, `VERDICTS_FT.md` (20/23 supported) |
| RE | Does the reach of the floor alone decide whether training can differ (moves and fees scaled)? | `ACCOUNTING_OPTIONS_2026-10-14/study_re/` | `PREREGISTRATION_RE.md`, `PREREGISTRATION_HASHES_RE.txt` | `runs_re.csv`, `cells_re.csv`, `VERDICTS_RE.md` (3/3 supported) |
| GC | Does PPO learn gym-mtsim's zero-balance floor in the package's own crypto market (forex control, reading-trained arm)? | `study_gc_2026-11-14/` | `PREREGISTRATION_GC.md`, `PREREGISTRATION_HASHES_GC.txt` | `results_gc/runs_gc.csv`, `results_gc/VERDICTS_GC.md` (6/7 supported; GC7 not), `results_gc/policies/` |
| DR | Does a widened training distribution switch on gym-mtsim's dormant forex floor, and does a pre-training edge predict where? | `study_dr_2026-11-15/` | `PREREGISTRATION_DR.md`, `PREREGISTRATION_HASHES_DR.txt` | `results_dr/runs_dr.csv`, `results_dr/VERDICTS_DR.md` (1/4 supported; DR1, DR2, DR3 not), `results_dr/policies/` |
| XE | Do two mainstream backtesting engines (backtesting.py, backtrader) agree inside the range and part past the kink, and does PPO trained through them learn the forgiveness? | `study_xe_2026-11-16/` | `PREREGISTRATION_XE.md`, `PREREGISTRATION_HASHES_XE.txt` | `results_xe/`, `xe_census.csv`, `VERDICTS_XE.md` (4/7 supported; XE2, XE3, XE7 not) |
| Control | backtesting.py with its two floor lines replaced in a subclass, against the published engine (exploratory, predictions fixed before the runs) | `xe_patch_control_2026-11-17/` | predictions in `README.md` | `results_ctl/`, `score_ctl.json`, `verify_ctl_out.txt` (4/4 predictions met) |
| Closed forms | What each accounting rule forgives, and the simulator's time step (exploratory, predictions written before each test) | `monitoring_law_2026-11-17/` | `PREDICTIONS.md`, `PREDICTIONS_v2.md` | `test_*` outputs (two pre-stated criteria failed, reported in `README.md`) |
| Settlement | The ladder and collateral in twelve Nordic zones (exploratory) | `ACCOUNTING_OPTIONS_2026-10-14/` | not pre-registered | `ladder12.csv`, `coverage12_*.csv`, `frontier12_*.csv`, `tails12.csv` |

Studies L and M are called "revision experiment L" and "revision audit M" in their frozen records. They were frozen before their runs but registered after the scorecard of Studies B–K, and are reported separately from its 46 hypotheses. Study N was registered on 14 October 2026; its `RUN_LOG_N.md` is append-only, and the registered version is the prefix before the line `## After hashing`.

Each study folder of A–K also holds `PREREGISTRATION_HASHES.txt` (SHA-256 of every registered file, with UTC timestamps) and `RUN_LOG.md` (launches, interruptions, relaunches and changes of job order). Theory solvers: `dp_theory.py` (Denmark), `dp_frictions.py` (Study H), `dp_j.py` (Study J), `dp_full.py` (full training distribution, all zones; supersedes the 3,000-sample solves, see `RUN_LOG.md`), `theory_k.py` and `reflection_barrier.py` (closed-form barrier), and `study_n/dp_n.py`, `dp_n2.py` (Study N). Best-moment theorem: `make_best_moment_figure.py` (writes `results_k/best_moment.csv` and the figure; `best_moment.py` is its earlier 1,000-draw version) and `best_moment_zones.py` (twelve zones). Other theory checks: `action_gap.py`, `false_profit_bound.py`, `partial_liability.py`, tables by `make_theory_tables.py`. Placebo-market test: `placebo_market.py`. Exploratory analyses are in `review_rigor.py` and `explore_i.py`.

## Repository layout
```
README.md, DESCRIPTION.md, CITATION.cff   overview, repository metadata, citation
CHECKLIST.md                              evaluation checklist for RL trading agents, their simulators and settlement rules
LICENSE, LICENSE-docs.md                  MIT (code); CC BY 4.0 (documentation, results, derived tables)
THIRD_PARTY_LICENSES.md                   sources, licenses and attribution of all third-party material
requirements*.txt                         analyses and studies; the engine stack; gym-mtsim (Studies G, GC, DR)
examples/quickstart.py                    reading a hidden floor from a toy simulator's reward (a smoke test of twin_reward/)
tools/                                    verify_studies.py and the other verifiers; ledger_check.py (reconciliation)
docs/                                     results in detail, reproduction guide, engine-audit protocols and methods
GENERALIZATION_STUDY_2026-09-29/          Study A
LIABILITY_STUDY_2026-09-29/               Studies B-K
revision_experiments/                     Studies L and M
ACCOUNTING_OPTIONS_2026-10-14/            theory checks, settlement analysis, Studies N-Z, RS, EXT, FT and RE; the twin_reward/ package
study_gc_2026-11-14/, study_dr_2026-11-15/, study_xe_2026-11-16/   Studies GC, DR and XE (registered)
novelty_search_2026-11-16/                Study XE's exploratory pilots (its census also feeds the closed-form tests)
xe_patch_control_2026-11-17/              the same-engine backtesting.py control (exploratory)
monitoring_law_2026-11-17/                the closed forms of the subsidies and their tests (exploratory)
exploratory_*/                            exploratory records cited by the paper's supplement
engine/, audit/, results/                 engine audit: simulator source (patched audit copy), audit code, results
analysis/, code_audit/, literature/       cross-market liquidity test; audit of 21 electricity-RL repositories; literature survey coding (no PDFs)
data/                                     Energinet data (CC BY 4.0), engine liquidity series, eSett download scripts
```
Folders hold their registered files byte for byte (`.gitattributes`), because their hash manifests cover them; run any script that rewrites outputs in a disposable copy.

## Pre-registration
All pre-registrations are internal: written and hashed before the corresponding test-period evaluation, not deposited with an independent registry. Their records disclose smoke tests on training data, design changes made before registration (for Study N: the fee and claim reporting, after a failed first pilot), a change of job submission order during Study B, a configuration-selection rule for Study I written after its pilot had been seen, aggregation scripts written after the runs for Study C, and, in the engine audit, timestamp corrections and one post-hoc correction of an aggregation script. `tools/verify_studies.py` checks every registered file against its recorded hash. The settlement analysis is exploratory.

Study RS was registered the same way, with one defect: its hasher matched the marker quoted in the run log's introduction, so the registered digest of `RUN_LOG_RS.md` covers its first 147 bytes and not the intended 3,555. The other 16 registered files match, and `tools/verify_studies.py` checks and reports this. It ran next to this repository and was copied in byte-exact, so its manifest and scripts name the original layout: `tools/verify_studies.py` maps the manifest and `study_rs/portable_rs.py` runs the scripts.

Studies RE, GC, DR and XE were registered the same way, each hash manifest taken before its full run; the manifests of Studies XE and DR include files of Study GC by relative path. The backtesting.py control and the tests of the closed forms are exploratory, with their predictions written to files before the runs.

## Not included
- Trained policies of most studies. Shipped: the 20 checkpoints of Study L, the models of Study R and the policies of Studies GC and DR. Every study ships its per-run results (`jobs/`, `results_*/`, `runs_*.csv`), so every decision can be recomputed exactly, but the other policies cannot be replayed without re-training, whose bit-exact reproducibility has not been checked.
- The engine's datasets and checkpoints (about 57 GB) and raw engine logs (about 105 GB); 110 of the 115 step traces of the engine twin check and 21 of the 26 scans of the engine's real-clone check, whose digests are listed in `exploratory_claude_2026-11-05/engine_twin/results/trace_sha256.txt` and `exploratory_claude_2026-11-06/engine_scan/results/scan_sha256.txt`.
- The example price file of Gym-Trading-Env (no license stated); `ACCOUNTING_OPTIONS_2026-10-14/study_ext/fetch_data.py` fetches it by hash.
- Raw eSett Open Data and the Study J panels derived from them (no explicit redistribution license); `data/fetch_esett.py` and `ACCOUNTING_OPTIONS_2026-10-14/fetch_esett_pre.py` re-download them and record their hashes.
- Third-party source code from the audits (the scan scripts regenerate the excerpts from pinned commits) and the paper PDFs of the literature survey (copyright).
- Development scratch work, AI-assisted review sessions and the exploratory pilots of Studies GC and DR (whose protocols disclose their designs and results), kept in the authors' private archive.

## Citation
See [`CITATION.cff`](CITATION.cff). Code: MIT; documentation and results: CC BY 4.0 (see `LICENSE`, `LICENSE-docs.md` and `THIRD_PARTY_LICENSES.md`).
