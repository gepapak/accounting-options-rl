# Automated Verification and Audit Suite

Part of the codebase for the paper *"Reading hidden loss forgiveness from the reward of reinforcement-learning trading simulators, with an electricity-trading engine"* submitted to *Engineering Applications of Artificial Intelligence* (EAAI).

## Overview
This directory contains the independent automated verification testbed for the paper. Reviewers can verify the mathematical, statistical, and empirical claims in minutes without training models.

## Verification Tools
- `verify_studies.py`: Primary verifier. Checks SHA-256 pre-registration hashes, completed agent counts (2,085 distinct agents across Studies B–FT, plus 280 in Study A), headline figures, and statistical hypothesis verdicts.
- `verify_release.py`: Recomputes all headline numbers, parameter dumps, and protocol variants from the Prototype5 engine audit.
- `ledger_check.py`: Standalone CLI tool for detecting loss forgiveness. Takes any episode cash trace CSV/GZ and tests for the presence of the +795% reported vs −3,655% booked inversion.
- `verify_accounting_revision.py`: Validates mathematical reflection identities and loss-charging formulas in exact rational arithmetic.
- `verify_reflected_control_revision.py`: Validates numerical bounds for partial loss-charging controllers.
- `test_verify_studies.py`: Unit test suite ensuring verifier routines operate correctly.

## Running the Verification Suite
From the repository root:
```bash
# 1. Run unit tests
python -m unittest discover -s tools -p test_verify_studies.py

# 2. Verify all pre-registered studies (Studies A through FT)
python tools/verify_studies.py

# 3. Verify engine audit headline numbers
python tools/verify_release.py

# 4. Verify exact mathematical identities
python tools/verify_accounting_revision.py
python tools/verify_reflected_control_revision.py

# 5. Test ledger reconciliation on a sample flagged trace
python tools/ledger_check.py results/ledger_examples/L3_marl_seed7_DK1.csv.gz
```
Expected output: all scripts exit with code `0` and print `all checks pass` or `ALL CHECKS PASSED`.
