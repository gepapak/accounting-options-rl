# Engine Audit Evaluation Results

Part of the codebase for the paper *"Simulators that forgive losses teach reinforcement-learning traders to gamble: theory, detection and removal, with an application to electricity trading"* submitted to *Engineering Applications of Artificial Intelligence* (EAAI).

## Contents
- `campaign/`: Evaluation outputs from the 300-run scarcity tail campaign (`results_long.csv`), tracking reported vs booked returns, cash floor hits, and solvency violations.
- `controlled_variants/`: Aggregated metrics for isolated protocol variations (protocols L1 through L7, testing the impact of liquidity participation caps and solvency enforcement).
- `cross_market/`: Empirical liquidity depth and order-book consumption metrics across Nordic price zones.
- `decision_replay/`: Replayed controller traces validating individual cash writes and action selections.
- `ledger_examples/`: Gzip-compressed step-by-step cash accounting traces (e.g., `L3_marl_seed7_DK1.csv.gz`), suitable for testing with `tools/ledger_check.py`.
