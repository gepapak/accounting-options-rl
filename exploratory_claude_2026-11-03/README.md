# Exploratory analyses after the fourth AI-assisted review (3 November 2026)

The fourth AI-assisted review of the submission (Codex, 3 November 2026; report kept outside this repository) confirmed the registered results of
Studies S to X and listed claims that were stated without their conditions or denominators, a false promise in the PMR docstring and a cost-shift
test whose checks were necessary but not sufficient. Everything in this folder was written after that review. None of it is registered; it reads
stored results (and, for the contract benchmark, the shipped market panels) and trains nothing.

| Path | What it does | Output |
|---|---|---|
| `exact_cost_shift/exact_cost_shift.py` | development version of the exact cost-shift test (`single_floor_fit`), now in `../ACCOUNTING_OPTIONS_2026-10-14/twin_reward/twin_reward.py` | -- |
| `exact_cost_shift/validate_exact.py` | exactness against an independent brute force (five cost grids, 68,637 generated and 19,744 perturbed sequences, no disagreement), no false alarm in 200,000 random single-floor worlds, power by violation family, the review's cases | `validate_exact_results.json` |
| `exact_cost_shift/rerun_contract_with_exact.py` | the benchmark of `../exploratory_claude_2026-11-02/contract_test.py` (single-floor sleeve, two-write sleeve, the public engine's cash-writing methods) with both versions of the test: identical counts | `contract_exact_results.json`, `rerun_contract.log` |
| `exact_cost_shift/make_costshift_table.py` | the supplementary table of power by family | manuscript `sections/tab_costshift_power.tex` |
| `audit_cohorts.py` | the ledger-free audit by cohort (Studies V, W, X, Y): agents, gamblers, largest TIR errors, reported-return errors | `audit_cohorts.json`, `audit_cohorts.md`, manuscript `sections/tab_audit_cohorts.tex` |
| `w_common_criterion.py` | Study W's six cells under the common significance-based criterion next to W2's mean-based one | `w_common_criterion.md`, manuscript `sections/tab_w_common.tex` |
| `twin_bracket.py` | the twin bracket `[2*SPR - r, r]` (any accounting that only forgives) checked on every stored audit episode of Studies V, W, X and Y (304 agents, 60,800 episodes: none outside) and on the public engine's six hand cases (all covered; TIR wrong in four) | `twin_bracket_results.json`, `twin_bracket.md`, manuscript `sections/tab_twin_bracket*.tex` |
| `holm_sensitivity.py` | Holm adjustment of every registered significance test of Studies S to Z and RS, within each study and across all (extended on 5 November 2026; the result for S to Y alone is on the last line of its output) | `holm_sensitivity.md`, manuscript `sections/tab_holm.tex` |

Corrections to hashed protocol files are in `ERRATA_T.md`, `ERRATA_W.md` and `ERRATA_X.md` next to them; the protocols themselves are unchanged.
`../tools/verify_studies.py` re-checks the numbers that the manuscript quotes from these outputs.
