# Complete Experimental Suite Guide (Studies A–FT)

Companion guide for the paper *"Reading hidden loss forgiveness from the reward of reinforcement-learning trading simulators, with an electricity-trading engine"* submitted to *Engineering Applications of Artificial Intelligence* (EAAI).

This document provides a comprehensive mapping of every empirical study and theoretical benchmark presented in the manuscript, including paper sections, pre-registrations, reproduction scripts, and cohort sizes.

---

## 1. Study Overview & Cohort Summary

The full experimental cohort comprises **2,365 trained agents** (280 in Study A + 2,085 across Studies B through FT), 300 campaign runs on the Prototype5 engine, and multiple independent verification suites:

| Study | Topic | Agents / Runs | Paper Section | Pre-registration | Location | Reproduction Script |
|---|---|---|---|---|---|---|
| **Engine Audit** | Origin +795% vs −3,655% inversion | 300 runs | Section 3 | `docs/PREREGISTRATION.md` | `engine/`, `audit/` | `python tools/verify_release.py` |
| **Study A** | Generalization across environment designs | 280 agents | Section 4 | `PREREGISTRATION_A.md` | `GENERALIZATION_STUDY_2026-09-29/` | `python aggregate_study.py` |
| **Study B** | Zero-floor gambling in placebo markets | 40 agents | Section 5 | `PREREGISTRATION_B.md` | `LIABILITY_STUDY_2026-09-29/` | `python aggregate_b.py` |
| **Study C** | Scaling exploit with training steps | 24 agents (120 checkpoints) | Section 5 | `scaling/PREREGISTRATION_C.md` | `LIABILITY_STUDY_2026-09-29/scaling/` | `python aggregate_c.py` |
| **Study D** | Audit of 18 public RL trading environments | 18 repos | Section 5 | `env_audit/PREREGISTRATION_D.md` | `LIABILITY_STUDY_2026-09-29/env_audit/` | Inspect `coding_d.csv` |
| **Study E** | Equity-scaled positions & log returns | 40 agents | Section 5 | `PREREGISTRATION_E.md` | `LIABILITY_STUDY_2026-09-29/` | `python aggregate_e.py` |
| **Study F** | Well-posed learning (equity sole input) | 40 agents | Section 5 | `PREREGISTRATION_F.md` | `LIABILITY_STUDY_2026-09-29/` | `python aggregate_f.py` |
| **Study G** | Replication in public simulator (gym-mtsim) | 50 agents | Section 5 | `mtsim_study/PREREGISTRATION_G.md` | `LIABILITY_STUDY_2026-09-29/mtsim_study/` | `python aggregate_g.py` |
| **Study H** | Realistic trading friction costs | 80 agents | Section 5 | `PREREGISTRATION_H.md` | `LIABILITY_STUDY_2026-09-29/` | `python aggregate_h.py` |
| **Study I** | Learner generality (A2C and DQN) | 160 agents | Section 5 | `PREREGISTRATION_I.md` | `LIABILITY_STUDY_2026-09-29/` | `python aggregate_i.py` |
| **Study J** | Multi-zone transfer (Finland, Norway) | 80 agents | Section 5 | `PREREGISTRATION_J.md` | `LIABILITY_STUDY_2026-09-29/` | `python aggregate_j.py` |
| **Study K** | Horizon sensitivity and discount factor $\gamma$ | 80 agents | Section 5 | `PREREGISTRATION_K.md` | `LIABILITY_STUDY_2026-09-29/` | `python aggregate_k.py` |
| **Study L** | Full loss charging with continued trading | 20 agents | Section 6 | `continuation_protocol.txt` | `revision_experiments/2026-10-10/` | `python aggregate_continuation.py` |
| **Study M** | Independent certification & model selection | 20 checkpoints | Section 6 | `prospective_manifest.json` | `revision_experiments/2026-10-10-selection/` | `python audit_frontier.py` |
| **Study N** | Settlement lag and terminal liability | 40 agents | Section 7 | `PREREGISTRATION_N.md` | `ACCOUNTING_OPTIONS_2026-10-14/study_n/` | `python aggregate_n.py` |
| **Study O** | Advantage-to-noise learnability index | 80 agents | Section 7 | `PREREGISTRATION_O.md` | `ACCOUNTING_OPTIONS_2026-10-14/study_o/` | `python aggregate_o.py` |
| **Study P** | Budget-dependent threshold scaling | 90 agents | Section 7 | `PREREGISTRATION_P.md` | `ACCOUNTING_OPTIONS_2026-10-14/study_p/` | `python aggregate_p.py` |
| **Study Q** | Learnability transfer to A2C and DQN | 100 agents | Section 7 | `PREREGISTRATION_Q.md` | `ACCOUNTING_OPTIONS_2026-10-14/study_q/` | `python aggregate_q.py` |
| **Study R** | Prospective cross-cost score transfer | 60 agents | Section 7 | `PREREGISTRATION_R.md` | `ACCOUNTING_OPTIONS_2026-10-14/study_r/` | `python aggregate_r.py` |
| **Study S** | Sign-paired reward (SPR) prevention | 180 agents | Section 8 | `PREREGISTRATION_S.md` | `ACCOUNTING_OPTIONS_2026-10-14/study_s/` | `python aggregate_s.py` |
| **Study T** | Twin-identified reward (TIR) on new draw | 170 agents | Section 8 | `PREREGISTRATION_T.md` | `ACCOUNTING_OPTIONS_2026-10-14/study_t/` | `python aggregate_t.py` |
| **Study U** | Remedies under settlement lag & terminal | 80 agents | Section 8 | `PREREGISTRATION_U.md` | `ACCOUNTING_OPTIONS_2026-10-14/study_u/` | `python aggregate_u.py` |
| **Study V** | Ruin penalty, PMR, and ledger-free audit | 144 agents | Section 8 | `PREREGISTRATION_V.md` | `ACCOUNTING_OPTIONS_2026-10-14/study_v/` | `python aggregate_v.py` |
| **Study W** | Selecting penalty strength via TIR audit | 48 agents | Section 8 | `PREREGISTRATION_W.md` | `ACCOUNTING_OPTIONS_2026-10-14/study_w/` | `python aggregate_w.py` |
| **Study X** | Two-write floors violating single-floor | 32 agents | Section 8 | `PREREGISTRATION_X.md` | `ACCOUNTING_OPTIONS_2026-10-14/study_x/` | `python aggregate_x.py` |
| **Study Y** | Second learner (A2C) under twin rewards | 80 agents | Section 8 | `PREREGISTRATION_Y.md` | `ACCOUNTING_OPTIONS_2026-10-14/study_y/` | `python aggregate_y.py` |
| **Study Z** | Multi-account portfolio & isolated margin | 136 agents | Section 9 | `PREREGISTRATION_Z.md`, `ZB` | `ACCOUNTING_OPTIONS_2026-10-14/study_z/`, `zb` | `python aggregate_z.py`, `zb` |
| **Study RS** | Real price signs stress test (DK2) | 80 agents | Section 9 | `PREREGISTRATION_RS.md` | `ACCOUNTING_OPTIONS_2026-10-14/study_rs/` | `python aggregate_rs.py` |
| **Study EXT** | Unmodified public Gym-Trading-Env | 70 agents | Section 9 | `PREREGISTRATION_EXT.md` | `ACCOUNTING_OPTIONS_2026-10-14/study_ext/` | `python aggregate_ext.py` |
| **Study FT** | Non-identifiable accountings (bucket/deficit) | 81 agents | Section 9 | `PREREGISTRATION_FT.md` | `ACCOUNTING_OPTIONS_2026-10-14/study_ft/` | `python aggregate_ft.py` |

---

## 2. Theory & Verification Packages

### Dynamic Programming and Analytical Option Pricing
- `LIABILITY_STUDY_2026-09-29/dp_theory.py`: Solves optimal trading policies under the five ladder rules for the Danish electricity market.
- `LIABILITY_STUDY_2026-09-29/make_best_moment_figure.py`: Implements Theorem 1 (Best-Moment Bound) showing that the zero floor subsidizes the agent between 1 and 2 times Merton's terminal call option.
- `ACCOUNTING_OPTIONS_2026-10-14/twin_reward/reward_diagram.py`: Reads the piecewise-linear reward diagram along ray queries $input(t) = reference + t \cdot (input - reference)$ to extract hidden floors, caps, and margins.

### Automated Release Verification
Run the unified verification scripts from the repository root:
```bash
# Verify all pre-registered studies and statistical criteria
python tools/verify_studies.py

# Verify the Prototype5 engine audit headline figures
python tools/verify_release.py

# Reconcile reported vs booked P&L on a sample episode trace
python tools/ledger_check.py results/ledger_examples/L3_marl_seed7_DK1.csv.gz
```
