# Study Q run log (append-only)

All times are local (UTC+2), 15 October 2026.

- **About 01:20.** `run_q.py` and `aggregate_q.py` written, from Study O's runner and Study I's learner configurations.
- **01:29.** Smoke test `run_q.py --smoke` (16,384 steps, TRAIN split, control arm, DK1, seed 50, A2C-n64 and DQN-default): both runs completed, excluded.
- **About 01:33.** `aggregate_q.py` tested on synthetic run files. The pre-registration is written; the hashes are taken next. The full run (100 runs) starts automatically when Study P's 90 runs are complete.

## After hashing (not part of the registered prefix)
- **01:35.** Hashes taken (23:31:13 UTC). Detached launcher `launch_after_p.ps1` started; it launches the full run when Study P is complete.
- **02:08.** Study P complete; full run launched automatically by launch_after_p.ps1 (PID 27524, `run_full.log`): 100 runs, 7 workers.
- **02:34.** The computer restarted (Windows boot time 02:34:22), stopping the run after 14 of 100 runs; no run had failed. The log of the first part is kept as `run_full_part1.log`.
- **08:58.** Relaunched unchanged (PowerShell Start-Process, PID 7960); completed runs are kept and skipped, interrupted runs restart from the beginning with the same seeds.

- **15:54 (15 October 2026).** All 100 full runs completed (last job at 15:52:23), with no FAILED entry in the full/resumption logs. The unchanged registered aggregate_q.py was run by the manuscript review and wrote runs_q.csv and VERDICTS_Q.md. All six registered hypotheses were supported: Q1, Q2, Q3 for A2C-n64 and DQN-default. DQN control leverage is high (5/10 at leverage 16 at the low-equity probe); any further diagnosis is exploratory and does not alter these verdicts.
