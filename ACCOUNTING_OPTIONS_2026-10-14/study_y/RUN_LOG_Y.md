# Study Y run log (append-only after "## After hashing")

All times are local (UTC+1, CET), 3 November 2026.

- **Before this study.** The fourth AI-assisted review (Codex, 3 November; `Paper1/_archive/codex_review_2026-11-03/REPORT.md`) confirmed the
  registered results of Studies S-X and listed, among the remaining objections, that every edge-market test of SPR and TIR used PPO. Study Y
  asks whether the results transfer to A2C (`run_y.py`), with Study X's measures and decision rules.
- **Software validation.** `run_y.py --smoke` (20,000 steps; PLAIN and TIR, seed 99, the single-floor sleeve without zero starts and the
  two-write sleeve; output in a scratch folder outside this repository) ran end to end at 07:34 and is excluded. After the settings were
  changed, the final `run_y.py --smoke` ran the new zero-start single-floor setting (PLAIN and TIR) and the setting without zero starts
  (PLAIN) end to end at 07:52-07:53 (output outside this repository, excluded).
- **Pilot (exploratory, seed 99, eight runs).** Launched 07:34 as a detached process (6 workers) with an earlier version of `run_y.py` whose
  settings were the single-floor sleeve without zero starts (`S_DK1`) and the two-write sleeve (`X_DK1+Z`), four arms each; finished 07:50
  (6 to 10 minutes per run). Results in `pilot_results/` (not part of the registered analysis) and in `PREREGISTRATION_Y.md`. The pilot changed
  the settings (the single-floor main setting now has zero starts; the setting without zero starts keeps PLAIN and ORACLE as an acquisition
  check). `aggregate_y.py --results pilot_results --print` was run on the pilot as a software check (it prints and writes nothing).

## After hashing (not part of the registered prefix)
- **07:53.** Hashes taken at 06:53:34 UTC (`hash_prereg_y.py`). Full run launched detached (`Start-Process -WindowStyle Hidden`, PID 24716,
  `run_full.log`): 80 runs (seeds 200-207), 7 workers. As in Study X, `run_full.log` prints each run's booked return when the run finishes;
  those lines may be read to follow progress. The registered analysis (`aggregate_y.py`) is run once, after all runs have finished.
- **08:11.** While testing a new descriptive script for the manuscript's audit table (`exploratory_claude_2026-11-03/audit_cohorts.py`, which reads
  every result folder of Studies V to Y), it was run once with 9 of the 80 Study Y runs finished; it printed, for those 9 runs of the
  single-floor setting with zero starts, the number with X >= 0.25 (2) and their largest audit errors. This is more than the booked-return progress
  lines and is disclosed here. Nothing in the hashed files, the design or the decision rules was changed; the registered analysis has not been run.
- **08:20-09:14.** The computer slept for about 50 minutes; the seven runs in progress took about 56 minutes of wall-clock time instead of
  8 to 9 and finished normally. The session's keep-awake setting was switched on at 09:19. No run failed and none was replaced.
- **10:13.** All 80 runs finished (the last at 10:13:31).
- **10:13.** The registered analysis (`python aggregate_y.py`) was run once on 80 of 80 runs and wrote `runs_y.csv` and `VERDICTS_Y.md`. Twelve of
  thirteen decisions were supported: YA1-YA5 and YB1 in both main settings; YC1 was not supported (without zero starts no reported-reward A2C
  agent chose maximal leverage at a low-equity probe), as the prior belief anticipated. `make_studyY_outputs.py` (the manuscript's tables and
  figure) and the exploratory scripts of `exploratory_claude_2026-11-03/` were run afterwards.
