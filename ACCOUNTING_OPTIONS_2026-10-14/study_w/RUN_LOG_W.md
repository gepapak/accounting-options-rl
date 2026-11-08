# Study W run log (append-only after "## After hashing")

All times are local (UTC+1, CET), 2 November 2026 unless stated.

- **Before this study (2 November, after the Study V write-up and two independent reviews).** An independent review (Codex review 3, 1 November)
  found that the manuscript claimed the working ruin-penalty strength "can be identified only from booked returns of trained agents", and showed
  retrospectively on Study V's stored arrays that the TIR-audited return selects strength 5 in both regimes. This repository's
  `study_w/retrospective_v.py` re-computes that selection (strength 5 by booked, TIR-audited and, on the full sample, reported return in both
  regimes; the reported return picks strength 1 on the odd-numbered half of Part A). The data were already analysed, so the look is not a test.
  Study W is the prospective test in two settings that were not part of the ruin-penalty grid.
- **Design choices made before launch.** Strength grid, seeds (180-187), settings and decision rules are those of `PREREGISTRATION_W.md`;
  `run_v.py` is used unchanged with explicit settings, arms and seeds. The decision-rule tolerance (1.0 point) is the audit-accuracy bound of
  Study V (VD1). No pilot runs were made for this study.
- **Software validation.** `aggregate_w.py --results ../study_v/results_v --demo` reproduces Study V's selection tables, its largest audit errors
  (0.385 and 0.129 points) and the software check passes; the demo writes nothing.

## After hashing (not part of the registered prefix)
- **21:00.** Hashes taken at 20:00:22 UTC. A 20,000-step smoke test of two jobs (`T_DK2` and `T_DK1M`, RUIN5, seed 999, output outside this folder) ran end to end beforehand and is excluded. Full run launched detached (`Start-Process -WindowStyle Hidden`, PID 1212, `run_full.log`): 48 runs (2 settings x 3 strengths x 8 seeds, seeds 180-187), 6 workers; about 12.5 minutes per run.
- **21:28.** The registered analysis script was run by mistake (instead of its `--demo` mode) while 12 of the 48 runs were complete. It wrote `runs_w.csv` and `VERDICTS_W.md` for the 12 runs; they were deleted unread at 21:30 (only the header line "Runs: 12 of 48" was displayed) and will be regenerated from the complete set. Nothing in the design, the decision rules or the code was changed.
- **22:51.** All 48 runs finished (the last at 22:50:28). Each took 16-17 minutes instead of the planned 12.5 because the Study X pilot (four workers, exploratory) and, from 22:51, the Study X run shared the machine. No run failed and none was replaced.
- **22:55.** The registered analysis (`python aggregate_w.py`) was run on 48 of 48 runs and wrote `runs_w.csv` and `VERDICTS_W.md`; `make_studyW_outputs.py` then wrote the manuscript tables from them. Decisions: W1 and W2 supported in both settings, W3 supported (largest audit error 0.077 point in the edge market, 0.063 in the placebo market), W4 not supported (the reported return selects the same strength as the booked return in both settings, as the prior belief anticipated). The audit-selected strength is 5 in DK2 and 1 in the weaker-edge DK1 setting.
- **Disclosure.** `run_full.log` prints each run's booked return as the run finishes, and those lines were visible while the study was running (they were read to follow progress). The decision rules, the analysis script and the hashed files were not changed after the first run completed.
