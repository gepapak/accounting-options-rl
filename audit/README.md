# Forensic Engine Audit: Reproducing the Reported-vs-Booked Inversion

Part of the codebase for the paper *"Simulators that forgive losses teach reinforcement-learning traders to gamble: theory, detection and removal, with an application to electricity trading"* submitted to *Engineering Applications of Artificial Intelligence* (EAAI).

## Overview
This directory contains the forensic audit harness that uncovered and reproduced the severe evaluation inversion in the Prototype5 renewable fund engine:
- In earlier protocols, learned MAPPO controllers were ranked first, reporting apparent returns of **+162% to +795%**, while actual booked portfolio losses reached **−1,822% to −3,655%**.
- The audit proves that two simulator shortcuts—a missing liquidity participation cap and missing cash solvency enforcement—were jointly necessary to create this illusion.

## Key Files
- `setup_audit_engine.py`: Configures the audit engine with identical historical configurations.
- `audit_run_variant.py`: Evaluates individual protocol variants (L1 through L7, cap-on vs cap-off).
- `run_all_variants.py`: Automated batch execution across all protocol variants and seeds.
- `run_scarcity_tail_campaign.py` & `aggregate_scarcity_tail_campaign.py`: Runs the full 300-run campaign testing the edge of extreme price scarcity events.
- `decision_replay.py`: Step-by-step trace replayer inspecting individual cash and margin writes.

## Reproduction
Aggregation and figure scripts may rewrite the registered outputs they read: run them in a disposable copy of the folder (see `docs/REPRODUCING.md`); `python tools/verify_studies.py` checks the shipped results without writing.

To verify the engine audit headline numbers against recorded campaign results:
```bash
python tools/verify_release.py
```
To run a standalone ledger reconciliation on an individual run trace:
```bash
python tools/ledger_check.py results/ledger_examples/L3_marl_seed7_DK1.csv.gz
```
