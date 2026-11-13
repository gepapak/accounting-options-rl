# Studies L & M: Reward Correction and Model Selection Certification

Part of the experimental suite for the paper *"Reading hidden loss forgiveness from the reward of reinforcement-learning trading simulators, with an electricity-trading engine"* submitted to *Engineering Applications of Artificial Intelligence* (EAAI).

## Overview
This directory contains the prospective revision experiments conducted to evaluate policy continuation mechanisms and out-of-sample frontier certification:
1. **Study L (`2026-10-10/`)**: Evaluates full loss charging with matched continued trading. Tests whether charging cash injections in full eliminates low-equity gambling while preserving normal trading incentives across 20 distinct training checkpoints.
2. **Study M (`2026-10-10-selection/`)**: Conducts prospective out-of-sample certification on the Study L models. Demonstrates that naive model selection based on reported return selects controllers that are certified to be the worst in actual booked economic performance.

## Structure
- `2026-10-10/` (Study L):
  - `continuation_protocol.txt`: Frozen execution protocol.
  - `run_continuation.py`: Training and continuation evaluation runner.
  - `aggregate_continuation.py`: Computes leverage reductions and booked P&L comparisons.
  - `continuation_experiment/`: Checkpoints, logs, and prospective manifest.
- `2026-10-10-selection/` (Study M):
  - `audit_frontier.py`: Audits reported vs booked returns across policy candidates.
  - `certify_frontier.py`: Evaluates statistical dominance and joint 95% certification bounds.
  - `summarize_frontier.py`: Compiles the validation scorecard.
  - `VALIDATION.json`: Frozen validation metrics.

## Reproduction
To verify Study L and Study M results without re-training:
```bash
# Study L aggregation
python revision_experiments/2026-10-10/aggregate_continuation.py

# Study M certification
python revision_experiments/2026-10-10-selection/audit_frontier.py
python revision_experiments/2026-10-10-selection/certify_frontier.py
```
All checks run in seconds and verify against the frozen prospective manifests.
