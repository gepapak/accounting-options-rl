# Study X run log (append-only after "## After hashing")

All times are local (UTC+1, CET), 2 November 2026 unless stated.

- **Before this study (2 November, after the Study V write-up, the third independent review and the cost-shift test).** The review found that the
  public engine writes its cash in several floored steps, so the single-floor assumption behind SPR and TIR does not hold there. The cost-shift
  contract test (`exploratory_claude_2026-11-02/contract_test.py`) detects the violation in a two-write variant of the sleeve and in the engine's
  own source methods. Study X asks the learning question on the two-write sleeve (`envs_x.py`).
- **Software validation.** `run_x.py --smoke` (20,000 steps, PLAIN and TIR, seed 190, output outside this folder) ran end to end and is excluded.
- **Pilot (exploratory, seed 90, four runs).** Launched 22:28 as a detached process (4 workers) while Study W was finishing; finished 22:50 (about
  18-20 minutes per run under load). Results (`pilot_results/`, not part of the registered analysis): PLAIN X 0.875, booked -0.9%; ORACLE +22.1%;
  SPR +22.0%; TIR +22.0%; see `PREREGISTRATION_X.md`. The decision rules, arms and measures were not changed after the pilot; only XB1's bound
  was tightened from 2.0 to 1.0 point, before hashing.

## After hashing (not part of the registered prefix)
- **22:51.** Hashes taken at 21:51:07 UTC (`hash_prereg_x.py`). Before hashing, `aggregate_x.py --results pilot_results --print` was run on the four pilot runs as a software check (it prints and writes nothing). Full run launched detached (`Start-Process -WindowStyle Hidden`, PID 3444, `run_full.log`): 32 runs (4 arms x 8 seeds, seeds 190-197), 7 workers.
- **23:09.** The first seven runs finished (15.5 to 17 minutes each). `run_full.log` prints each run's booked return as the run finishes, and those lines were visible while the study was running (they were read to follow progress). The registered analysis (`aggregate_x.py`) was not run before all 32 runs had finished. Other work on the machine (a verifier run and LaTeX builds) shared the CPU for a few minutes and lengthened some runs; nothing in the design, the decision rules or the hashed files was changed.
- **00:02 (3 November).** All 32 runs finished (the last at 00:01:55); no run failed and none was replaced. Runs took 9.4 to 17 minutes, longer while other work shared the machine.
- **00:03.** The registered analysis (`python aggregate_x.py`) was run once on 32 of 32 runs and wrote `runs_x.csv` and `VERDICTS_X.md`. All six decisions were supported (XA1 to XA5 and XB1). `make_studyX_outputs.py` (the manuscript's figure and tables) and `explore_x.py` (`EXPLORATORY_X.md`, not registered) were run afterwards.
