# Study A: Generalization Across Electricity-Trading Environments

Part of the experimental suite for the paper *"Reading hidden loss forgiveness from the reward of reinforcement-learning trading simulators, with an electricity-trading engine"* submitted to *Engineering Applications of Artificial Intelligence* (EAAI).

## Overview
Study A evaluates reinforcement-learning trading agents across three distinct environment designs (280 trained agents) using real 15-minute imbalance market data from Energinet (Denmark DK1 and DK2 zones). It investigates whether evaluation shortcuts and reward formulations systematically reverse the apparent ranking between learned policies and fixed baseline rules.

## Contents
- `PREREGISTRATION_A.md`: Frozen pre-registered hypotheses and evaluation protocol.
- `PREREGISTRATION_HASHES.txt`: SHA-256 integrity digests recorded before test-period execution.
- `RUN_LOG.md`: Execution log recording runs and timestamps.
- `envs.py`: Gym-compatible trading environment implementations across the three designs.
- `build_panel.py`: Pipeline converting raw Energinet market feeds into reproducible evaluation panels.
- `run_study.py`: Main experimental runner for training and evaluating Study A agents.
- `aggregate_study.py`: Statistical aggregation and hypothesis testing script.
- `make_studyA_outputs.py`: Generates publication figures and summary tables for the manuscript.
- `data_energinet/`: Raw Energinet imbalance and spot price data (CC BY 4.0).
- `data_panel/`: Preprocessed train/test panel splits (`panel_DK1.pkl`, `panel_DK2.pkl`).
- `results/`: Aggregated evaluation metrics and per-agent test logs.
- `outputs/`: Manuscript-ready plots and statistical summaries.

## Reproduction
To verify and reproduce Study A results from the shipped artifacts:
```bash
python aggregate_study.py
python make_studyA_outputs.py
```
This reads the frozen result files in `results/`, verifies all registered criteria, and regenerates figures in `outputs/` without modifying pre-registered hashes.
