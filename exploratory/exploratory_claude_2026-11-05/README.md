# Exploratory analyses of 5 November 2026

Written after a read-only audit of the submission (AI-assisted, 5 November 2026). None of this is registered and the registered verdicts are unchanged. The two scripts below read stored results and the shipped market panels and train nothing; with `--print` they write nothing.

| Path | What it does | Output |
|---|---|---|
| `rs_reanalysis.py` | Study RS beyond its threshold decisions: the head-to-head contrast of TIR and SPR, uncertainty from the reused test panel (window overlap, seeds with windows, seeds with calendar months, the edge by month) and what the 8-row lag and the 672-row episodes are in calendar time | `rs_reanalysis.json`, manuscript `sections/tab_studyRS_months.tex` |
| `bracket_limits.py` | What happens outside the premises of the bracket and of TIR: a carried deficit (the settlement lag's zombie window), split accounts with legs of opposite sign, a charge on one branch only (a liquidation fee), a payoff convex in the sign; and the width against the forgiveness realized on the path | `bracket_limits.json` |

`engine_twin/` is a further exploratory check with its own hashed rules and README: the sign twin on the engine's own evaluation paths, and the correction of the audit's uncapped percentage payoff.

`../tools/verify_studies.py` re-runs both scripts with `--print`, compares them with the stored outputs and checks the numbers the manuscript quotes.
