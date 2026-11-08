# Study RE run log (append-only)

All times are local (UTC+1), 7 November 2026. Written in an AI-assisted session (Claude in Cursor).

- **14:40-14:55.** `envs_re.py`, `reach_re.py`, `run_re.py`, `aggregate_re.py` and `PREREGISTRATION_RE.md` written. `reach_re.py`
  run (20,000 simulated episodes per scale): predicted reach 856, 443, 149, 5.7 and 0.0 floor-reaching episodes per 10^6-step run
  at c = 1, 0.3, 0.1, 0.03, 0.01.
- **14:57.** Smoke test `run_re.py --smoke` (16,384 steps, seed 0, c in {1, 0.01}, 4 workers): four runs in about 28 s each, every
  greedy-policy exactness check passed; the c = 0.01 pair identical (parameter digest 9b9ae50f...), the c = 1 pair different.
  Smoke results (`results_re/jobs_smoke/`) are excluded.
- **15:00.** The hashes are taken next; then the full run (50 runs, 8 workers) starts.
