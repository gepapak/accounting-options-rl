# Study U run log (append-only after "## After hashing")

All times are local (UTC+2), 31 October 2026 unless stated.

- **Before this study.** The theory for carried-deficit and zombie-window rules (`exploratory_claude_2026-10-31/identification/THEORY_TIR.md`,
  `verify_identification.py`: 200,000 exact random cases, an LP showing the 4/9 regret bound) and the random-action identification study on
  Study N's environments (`ladder_identification_study.py`). No training of the twin-based arms in Study N's environments was run before
  this study.
- **Environments.** `../study_t/envs_tn.py` is a copy of `../study_n/envs_n.py` with a sign-override added and the per-step P&L and own cost
  returned in `info`; `../study_t/unit_test_envs_tn.py` checked it against the original on 36,000 steps (0 mismatches).
- **Smoke test.** `run_u.py --smoke` (16,384 steps, seed 140, DK1, all eight arms) ran and the greedy NumPy forward pass matched
  `model.predict`; smoke results (`results_u/smoke/`) are excluded.
- **Order.** Study U starts after Study T completes (170 runs): `launch_u_after_t.ps1` polls for the 170 Study T result files and then
  starts `run_u.py` with 8 workers. Study U's hashes are taken before Study T's results are known; Study T's verdicts do not enter any Study U
  rule.

## After hashing (not part of the registered prefix)
- **03:49.** Hashes taken (UTC 02:49, local UTC+1) together with Study T's. Study U is launched by `launch_u_after_t.ps1` when Study T has 170 result files.
- **03:49.** Waiter started (PID 13996, hidden powershell).
- **11:54.** Study T complete; full run launched automatically by launch_u_after_t.ps1 (PID 16596, `run_full.log`): 80 runs, 8 workers.
- **14:39.** All 80 runs complete (last at 14:39:20), no FAILED entry, empty `run_full.err`. The registered `aggregate_u.py` (hash checked against `PREREGISTRATION_HASHES_U.txt`) was run here and wrote `runs_u.csv` and `VERDICTS_U.md`: all six hypotheses SUPPORTED. Zombie window: X FG 16.00, BK 1.18, SPR 1.45, TIR 1.21 (FG - BK +14.82, bound +14.26; FG - SPR +14.55, SPR - BK +0.27; FG - TIR +14.79, TIR - BK +0.04). Terminal limited liability: X FG 10.84, BK 1.59, SPR 0.79, TIR 0.38 (FG - BK +9.25, bound +6.38; FG - SPR +10.05, SPR - BK -0.80; FG - TIR +10.46, TIR - BK -1.21).
