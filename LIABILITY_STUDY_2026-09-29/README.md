# Studies B–K: The Option Ladder of Accounting Rules

Part of the experimental suite for the paper *"Simulators that forgive losses teach reinforcement-learning traders to gamble: theory, detection and removal, with an application to electricity trading"* submitted to *Engineering Applications of Artificial Intelligence* (EAAI).

## Overview
This directory contains the experimental code, environments, theory solvers, and pre-registered hypotheses for Studies B through K (594 trained agents). These studies establish that when trading simulators floor cash balances at zero, reinforcement-learning agents learn to exploit this implicit call option ("gambling for resurrection"), so that evaluation reports profits where the booked ledger records losses.

## Study Mapping
| Study | Objective & Scope | Key Scripts | Results Folder |
|---|---|---|---|
| **Study B** | Do agents learn to gamble under a zero floor in placebo markets? | `run_b.py`, `aggregate_b.py`, `envs_b.py` | `results/` |
| **Study C** | Does the exploit grow with additional training steps? | `scaling/run_c.py`, `scaling/aggregate_c.py` | `scaling/results/` |
| **Study D** | Prevalence of loss forgiveness in 18 public RL trading environments | `env_audit/` | `env_audit/coding_d.csv` |
| **Study E** | Can equity-scaled positions or log returns eliminate gambling? | `run_e.py`, `aggregate_e.py`, `envs_e.py` | `results_e/` |
| **Study F** | Well-posed learning: PPO with equity as the sole state input | `run_f.py`, `aggregate_f.py`, `pilot_f.py` | `results_f/` |
| **Study G** | Replication in an unmodified public Forex simulator (`gym-mtsim`) | `mtsim_study/run_g.py`, `aggregate_g.py` | `mtsim_study/results_g/` |
| **Study H** | Do realistic trading friction costs deter exploitation? | `run_h.py`, `aggregate_h.py`, `envs_h.py` | `results_h/` |
| **Study I** | Does gambling occur across other RL algorithms (A2C, DQN)? | `run_i.py`, `aggregate_i.py`, `select_pilot_i.py` | `results_i/` |
| **Study J** | Cross-market generalization: Finland (FI) and Norway (NO2) | `run_j.py`, `aggregate_j.py`, `build_panel_j.py` | `results_j/` |
| **Study K** | Horizon sensitivity: does gambling widen with discount factor $\gamma$? | `run_k.py`, `aggregate_k.py`, `theory_k.py` | `results_k/` |

## Dynamic Programming & Analytical Solvers
- `dp_theory.py`: Solves optimal policies and option values for Denmark (the theory of the paper).
- `dp_full.py`: Calculates optimal gambling regions across the full training distribution for all zones.
- `reflection_barrier.py` & `reflection_theory.py`: Analytical closed-form solutions for the reflected diffusion barrier.
- `make_best_moment_figure.py`: Computes and plots the best-moment theorem (a supplementary theorem of the paper) across zones.
- `placebo_market.py`: Evaluates policy behavior in sign-randomized placebo markets without edge.

## Reproduction
Aggregation and figure scripts may rewrite the registered outputs they read: run them in a disposable copy of the folder (see `docs/REPRODUCING.md`); `python tools/verify_studies.py` checks the shipped results without writing.

To verify all statistical aggregations and hypothesis verdicts from the completed runs:
```bash
python aggregate_b.py
python aggregate_e.py
python aggregate_f.py
python aggregate_h.py
python aggregate_i.py
python aggregate_j.py
python aggregate_k.py
python make_liability_outputs.py
```
To reproduce dynamic programming theory benchmarks:
```bash
python dp_theory.py
python make_best_moment_figure.py
```
All scripts verify against pre-registered criteria without modifying pre-registered hashes.
