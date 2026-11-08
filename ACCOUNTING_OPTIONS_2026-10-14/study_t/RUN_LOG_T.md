# Study T run log (append-only after "## After hashing")

All times are local (UTC+2), 31 October 2026 unless stated.

- **Before this study (31 October).** Exploratory development in `exploratory_claude_2026-10-31/` (pilot seeds 90-92): the censored-reward
  theory and its exact checks (`identification/`), one-step economics of candidate settings (`study_t_development/calibrate_settings.py`), a
  pilot of the ORPO baseline and of the sign-twin rewards (`study_t_development/pilot_*`). A first pilot launch (a `Win32_Process` chain with
  `&&`) was killed by a console-close event (`forrtl: error (200)`, BrokenProcessPool) and rerun with a detached `Start-Process`. Pilot findings
  that shaped the design are listed in `PREREGISTRATION_T.md`: the ORPO grid was extended downwards (ORPO at 10, 1 and 0.1 returned a
  reference-level policy), and a weaker-edge setting (DK1, p_edge 0.55) was replaced by DK1, p_edge 0.58 because its ORACLE edge was too small.
- **Before 03:36.** `PREREGISTRATION_T.md`, `envs_t.py`, `spr_t.py`, `orpo_t.py`, `run_t.py` and `aggregate_t.py` finalized.
  `aggregate_t.py` tested on synthetic run files and on the smoke results (it labels hypotheses "not testable" when the exploit is absent, and
  TA6 "not testable" when the ORACLE edge is not positive). An edit script that changed `envs_t.py` before `run_t.py` failed halfway
  (an assertion on the arm list); the files were brought into agreement before any registered run. The smoke test `run_t.py --smoke`
  (20,000 steps; every arm in `S_DK1`, four arms in each of `T_DK2` and `T_DK1M`; 17 runs) passed at 03:36, all greedy-policy exactness checks
  passed. Smoke results (`results_t/smoke/`) are excluded.
- **Evaluation windows.** The pilots drew the test windows from the run seed. The registered version fixes them (edge market seed 9000,
  placebo market seed 9500; 100 episodes) so that all runs are evaluated on the same windows and start equities.
- **03:38-03:48.** A 4-run pilot of `T_DK1M` (PLAIN and ORACLE, seeds 90-91, full budget; results in
  `exploratory_claude_2026-10-31/study_t_development/pilot_results_m/`) checked that the exploit is present and the edge positive in that
  setting before the hashes were taken: booked edge PLAIN -6.4% and -0.1% of K0 per episode against ORACLE +10.2% and +13.8%.
- **About 03:50.** The hashes of Study T and Study U are taken next, then the Study T full run (170 runs, 8 workers) is launched, with the
  Study U launch chained after it.

## After hashing (not part of the registered prefix)
- **03:49.** Full run launched detached (`Start-Process -WindowStyle Hidden`, PID 16560, `run_full.log`): 170 runs, 8 workers. Correction to the header: local time on this date is UTC+1 (CET; summer time ended on 25 October), not UTC+2; the hash file records UTC.
- **09:34.** Part A complete (90/90 runs, no failure; last at 09:32). The registered `aggregate_t.py` (hash checked) was run in print-only mode on the partial results folder; Part A verdicts do not depend on Part B (the Part A bootstrap draws come first). TA1-TA6 all SUPPORTED: X PLAIN 0.600 vs ORACLE 0.025; SPR 0.000 (E +19.40 vs ORACLE +20.46, ratio 0.948, bound 0.852); TIR 0.025 (E +21.34, ratio 1.043, bound 0.983); ORPO c = 1, 0.1, 0.05: X 0.000 and E +7.70; c = 0.02: X 0.287, E +0.94; c = 0.01: X 0.537, E -0.07. Part B (80 runs) is still running; `VERDICTS_T.md` is written only once all 170 runs are complete.
- **11:55.** All 170 runs complete (last at 11:54:26), no FAILED entry, empty `run_full.err`. The registered `aggregate_t.py` (hash checked against `PREREGISTRATION_HASHES_T.txt`) was run here and wrote `runs_t.csv` and `VERDICTS_T.md`: all 16 hypotheses SUPPORTED. Part A as at 09:34. Part B, DK2: X PLAIN 0.394 vs ORACLE 0.000 (p = 0.0029), SPR 0.000, TIR 0.013; E SPR +33.34 and TIR +33.97 vs ORACLE +32.66 (ratios 1.02 and 1.04, bounds 0.93 and 0.97). Part B, DK1 weaker edge: X PLAIN 0.388 vs ORACLE 0.000 (p = 0.0004), SPR 0.000, TIR 0.000; E SPR +13.45 and TIR +13.25 vs ORACLE +11.63 (ratios 1.16 and 1.14, bounds 1.00 and 0.99). Study U was launched automatically at 11:54 (see `RUN_LOG_U.md`).
