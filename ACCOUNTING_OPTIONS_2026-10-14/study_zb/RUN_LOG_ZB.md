# Study Z, Part B run log (append-only after "## After hashing")

All times are local (UTC+1, CET), 3 November 2026.

- **Before this part.** Study Z (split accounts, both balances visible) was registered at 16:03 UTC and launched at 17:03. At about 17:40 the
  user passed on an interim AI-assisted review of Study Z's design (GitHub Copilot): registration, exact checks, penalty units and theory
  confirmed; main objection that the agent sees both balances, so a per-account ruin penalty is always available and the portfolio penalty's
  blindness is built in; recommendation to add a case in which the floors exist but only the balance is visible (per-position isolated
  margin). Part B was designed in response, from 17:50.
- **Calibration (exploratory, no learning).** On the DK1 training panel (28,259 quarter-hours; E|s| = 76.7 EUR/MWh, median 43; P(|s| > 25,
  50, 100) = 0.634, 0.448, 0.162), with p = 0.6 and kappa 4, the reported advantage of leverage 16 over leverage 4 in isolated margin is
  +68.4, +56.0, +39.7 and +22.6 EUR per quarter-hour for M = 10, 25, 50 and 100 EUR/MWh, against a booked advantage of -14.3; M_ISO = 100 was
  chosen as the most conservative of these margins.
- **Exact checks.** `test_envs_zb.py`: 322,396 checks, no failure (17:55).
- **Scripted witness (`witness_zb.py`).** First run with HONEST, CROSS_LOW and ISO16 (isolated maximal leverage at every equity): ISO16 had
  7.6 account ruins per edge-market test week (repeated liquidations drain a low-equity account) and was deterred by the ruin penalty
  (-0.5 at strength 5). The funded-only policy ISO16_HIGH was then added: no account ruin, 21.0 liquidations per week, booked -9.5% against
  +22.0% for HONEST, reported +38.7% against +22.0%, reward unchanged under the ruin penalty at 5 and 25, -66.3 under the liquidation penalty
  at 5. This changed the registered exploit measure to the funded probes (equity 0.25 to 2 allocations) and removed zero starts from training.
- **Software validation.** `run_zb.py --smoke` (20,000 steps; REP and TIR, seed 220; output in a scratch folder outside this repository)
  ran end to end at 17:55-17:56 and is excluded. `aggregate_zb.py --print` and its verdict code (output to a scratch folder) were run on the
  smoke output; this set ZIB2's bound at 0.5 points (an untrained smoke agent's TIR audit error was 0.18 points, from steps of an account below
  its own cost in cross margin).
- **Pilot (exploratory, seed 99, two runs).** Launched 17:56 as a detached process (2 workers, `run_zb.py --out pilot_results --seeds 99
  --arms REP RUINP25`), sharing the processors with Study Z; REP finished at 18:21:42 and RUINP25 at 18:22:15. Results in `pilot_results/` and
  `PREREGISTRATION_ZB.md`; `aggregate_zb.py --results pilot_results --print` was run as a software check. At about 18:15 the user passed on a
  second AI-assisted review (Copilot) of the proposed next steps, which recommended a ledger-free exploit finder in settings where forgiveness
  is hidden; the FINDER arm and hypotheses ZIC1-ZIC2 were added and smoke-tested (18:22-18:23, one run, 20,000 steps, scratch output, excluded)
  before the pilot results were read.

## After hashing (not part of the registered prefix)
- **18:26.** Hashes taken at 17:26:12 UTC (`hash_prereg_zb.py`). `chain_zb.py` launched detached (PID 12412): it holds the computer's idle
  sleep, waits for Study Z's run (`run_z.py`, PID 21892) to exit, then launches the full run (64 runs, seeds 220-227, 7 workers) and, when
  all 64 have finished, runs the registered analysis once and records it here.
- **22:40.** Study Z's run had exited; `chain_zb.py` launched the full run (`run_zb.py --workers 7`, `run_full.log`): 64 runs (seeds 220-227). As in Studies X to Z, `run_full.log` prints each run's booked returns when the run finishes; those lines may be read to follow progress. The registered analysis (`aggregate_zb.py`) is run once, after all runs have finished.
- **01:11.** All 64 runs finished (the last at 01:10:57). The registered analysis (`python aggregate_zb.py`) was run once, automatically by `chain_zb.py`, and wrote `runs_zb.csv` and `VERDICTS_ZB.md`: 11 of 13 decisions supported. `make_studyZB_outputs.py` (the manuscript's tables and figure) was run afterwards.
- **4 November, 08:20.** Reading the verdicts showed that the hashed protocol text of ZIB1 and ZIB2 still says `For all 56 runs` (written before
  the FINDER arm was added), while the hashed analysis tests all 64; documented in `ERRATA_ZB.md`. No hashed file was changed and the registered
  analysis was not re-run.
