# Study RS run log

Append-only after the line "## After hashing (not part of the registered prefix)". Times are local (Central European Time).

## Before hashing
- **4 November 2026, 21:43.** Design chosen after an independent review named "toy sleeve with simulated signs" as the first attack a hostile
  EAAI reviewer would make, and after confirming that no registered twin-reward study used real signs (Study T's unused "Part C" code for
  Study N's environments was never registered or run). Design-stage diagnostics on the real data (no training): sign statistics of lagged
  spreads (training split; also lags 2, 8 and 96 on the test split, disclosed in the pre-registration) and a myopic witness (`witness_rs.py`).
- **21:50.** `test_envs_rs.py`: 270,331 checks pass (40,320 twin steps; 27,089 identified, 13,231 not identified).
- **21:51-21:56.** `run_rs.py --smoke` (20,000 steps, every arm and setting, seed 300): 10 of 10 runs end to end; the step bracket held on all
  53,760 audited smoke steps and TIR was exact on all 31,572 identified ones. Smoke results (`results_rs/smoke/`) are excluded.
- **21:58.** Testbed pilot launched: `run_rs.py --pilot` (PLAIN and ORACLE, seeds 98-99, both settings, 10^6 steps, 8 workers).
- **22:35.** Pilot 1 finished (8 of 8 runs, about 2,010-2,035 s each; `pilot_results/`, `pilot_run.log`). The exploit was acquired by one
  seed of two per setting (results in the pre-registration). Decision: add zero-equity training starts as in Study V Part C. Checked and
  rejected: `kappa = 4` for DK1 (a test-split evaluation of the myopic booked policy, -9.5 percent per week, disclosed) and a signal lag of 4
  (training-split myopic witness only). `run_rs.py`'s pilot flag now names pilot 2; `--pilot1` reproduces pilot 1.
- **22:40.** `envs_rs.py` gains `zero_starts` (the draw of `rewards_v.EdgeEnvZ`); `run_rs.py` uses it for every training environment;
  `aggregate_rs.py` gains the ORACLE lower-bound condition for the edge tests and restricts RS-D1 to the ORACLE, SPR and TIR runs (Study X's
  XB1 precedent, 1.0 point). `test_envs_rs.py`: 309,221 checks pass. Synthetic-data check of the three decision branches (normal; ORACLE
  edge not reliably positive; exploit not acquired): as registered.
- **22:41-22:44.** `run_rs.py --smoke` for the changed code: 10 of 10 runs (`results_rs/smoke2/`, excluded).
- **22:45.** Pilot 2 launched: `run_rs.py --pilot` (zero-equity training starts; PLAIN and ORACLE, seeds 98-99, both settings, 8 workers).
- **23:25.** Pilot 2 finished (8 of 8 runs, 2,282-2,324 s each; `pilot2_results/`, `pilot2_run.log`). DK2: both PLAIN runs gambled; DK1:
  one of two, and DK1's ORACLE edge cannot be told from zero (results in the pre-registration). Decision: register DK2 only with 16 seeds
  (same 80 runs), judge both twin rewards by Study V's edge criterion (Monte Carlo from the pilot edges: pass probability 0.83 against 0.48
  for Study T's TIR bound if TIR matched ORACLE), keep everything else. `run_rs.py` (`REGISTERED_SETTINGS`, seeds 300-315) and
  `aggregate_rs.py` (setting, tags RS1-RS6, TIR bound) changed accordingly; `hash_prereg_rs.py` now also hashes Study A's `envs.py` and the
  two market panels.
- **23:28.** Registration: the files listed in the pre-registration are hashed with `python -B hash_prereg_rs.py` (UTC time in `PREREGISTRATION_HASHES_RS.txt`) and checked with `--verify`; the full run is then launched with `python -B run_rs.py --workers 8 --finish`.

## After hashing (not part of the registered prefix)

- **23:29.** Full run launched detached: `python -B run_rs.py --workers 8 --finish`, PID 5564, started 23:29:08 local (4 November 2026), 8 worker processes. `run_full.log` reads "80 jobs, 0 complete, running 80 with 8 workers"; `run_full.err` is empty. Expected finish about 06:00-07:30 on 5 November (about 10 batches of about 38-45 min). When all 80 runs are done, `--finish` runs `aggregate_rs.py` once.
- **02:22.** All 80 runs finished. The registered analysis (`python aggregate_rs.py`) was run once, automatically by `run_rs.py --finish` (exit code 0), and wrote `runs_rs.csv` and `VERDICTS_RS.md`.
