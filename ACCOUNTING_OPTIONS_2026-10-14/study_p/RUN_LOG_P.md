# Study P run log (append-only)

All times are local (UTC+2), 14 October 2026.

- **About 20:55.** `run_p.py`, `aggregate_p.py` and `design_p.py` written while Study O was running.
- **21:00.** Provisional design on 56 of Study O's 70 treatment runs (c50 = 0.437%); power simulation of three designs; arms fixed at 250k {0, 0.25, 0.5, 0.7, 0.8, control} and 4M {0.8, 0.9, control}.
- **21:28.** Study O complete. `design_p.py` rerun on all 70 treatment runs (c50 = 0.429%, slope 5.20) -> `design_p.json`.
- **21:30.** Smoke test `run_p.py --smoke` (16,384 steps, TRAIN split, control arm, DK1, seed 40, both budget labels): completed, excluded.
- **21:31.** P4's pooled arms widened to {0.5, 0.7, 0.8} for power. The pre-registration is written; the hashes are taken next, then the full run is launched (90 runs).

## After hashing (not part of the registered prefix)
- **21:30:43.** Hashes taken (19:30:42 UTC). Full run launched as an independent process (PowerShell Start-Process, PID 23880, `run_full.log`): 90 runs, 7 workers, 4M-step runs first.
- **Erratum.** The registered entry "21:31" is approximate: the widening of P4 and the writing of the pre-registration were completed before the hashes were taken at 21:30:42.
- **About 02:05.** Full run complete: 90 of 90 runs, no failures. `aggregate_p.py` (registered) run next.
- **02:10.** `aggregate_p.py` wrote `VERDICTS_P.md` and `runs_p.csv`: P1, P2 and P3 supported; P4 not supported.
