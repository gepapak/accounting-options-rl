# Automated Verification and Audit Suite

Part of the codebase for the paper *"Simulators that forgive losses teach reinforcement-learning traders to gamble: theory, detection and removal, with an application to electricity trading"* submitted to *Engineering Applications of Artificial Intelligence* (EAAI).

## Overview
This directory contains the independent automated verification testbed for the paper. Reviewers can verify the mathematical, statistical, and empirical claims in minutes without training models.

## Verification Tools
- `verify_studies.py`: Primary verifier. Checks the SHA-256 pre-registration hashes of every registered study (2,295 distinct agents in Studies B to FT, RE, GC, DR and XE, plus 280 in Study A), re-runs the exact checks, re-runs the registered aggregators of Studies RE, GC, DR and XE on their stored results in a scratch copy, and re-checks headline numbers and hypothesis verdicts. About 1,530 checks in some eight minutes; on a fresh clone the three checks that need the non-redistributed eSett panels of Study J report "not shipped" instead.
- `verify_release.py`: Recomputes all headline numbers, parameter dumps, and protocol variants from the Prototype5 engine audit.
- `ledger_check.py`: Standalone CLI tool for detecting loss forgiveness. Takes an episode cash trace (CSV or GZ) and reconciles reported against booked return.
- `verify_accounting_revision.py`: Validates mathematical reflection identities and loss-charging formulas in exact rational arithmetic.
- `verify_reflected_control_revision.py`: Validates numerical bounds for partial loss-charging controllers.
- `test_verify_studies.py`: Unit test suite ensuring verifier routines operate correctly.

## Running the Verification Suite
From the repository root:
```bash
# 1. Run unit tests
python -m unittest discover -s tools -p test_verify_studies.py

# 2. Verify all pre-registered studies (Studies A to FT, RE, GC, DR and XE)
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
