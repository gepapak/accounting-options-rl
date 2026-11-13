# Exploratory records, 2 November 2026 (Claude)

Not registered and not part of any study's decision rules.

| File | What it does |
|---|---|
| `check_identification_statements.py` | Data-free brute-force check of the corrected identification statements: the identified set of the gross magnitude for each observed twin pair (420 pairs, including no-credit steps), the non-sharpness of SPR's magnitude, TIR's pathwise error dominance over SPR (5,200 grid points), the minimax regret `Q A B / (A + B)` against a brute-force search over policies, and the finite-pool counterexamples of the pooled-magnitude reward. Imports no project code; exit code 0 means every check passes. |
| `make_fig_overview.py`, `fig_overview.png` | Draws the overview figure of the manuscript (`figures/fig_overview.pdf` in the manuscript repository). |

Related records: the independent review's exact audit (`../exploratory_codex_2026-11-01/review_theory/`), the ledger-free selection motivating Study W (`../ACCOUNTING_OPTIONS_2026-10-14/study_w/retrospective_v.py`) and the errata appended to `../exploratory_claude_2026-10-31/identification/THEORY_TIR.md` and `.../study_v_development/THEORY_PMR.md`.
