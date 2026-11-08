# Study Z run log (append-only after "## After hashing")

All times are local (UTC+1, CET), 3 November 2026.

- **Before this study.** After the twin bracket was added, the user asked for a genuinely novel result. A literature search (reward hacking
  in simulators, limited liability and learning agents, loss mutualization, judgment proofing, isolated margin) found no study of learning
  agents and per-account floors. The split-account proposition and its exact checks (`THEORY_Z.md`, `test_envs_z.py`) were written and run
  (216,000 checks, no failure) before any agent was trained. At about 16:47, while the pilot was running and before any pilot run had finished,
  part 3 was extended to losses and drawdowns (a step that trades only from empty accounts has a non-negative portfolio reward) and the park
  check to every leverage level; the extended checks (244,000) passed.
- **Software validation.** `run_z.py --smoke` (20,000 steps, seed 210, all ten arms; output in a scratch folder outside this repository)
  ran end to end at 16:35-16:36 and is excluded. `aggregate_z.py --print` was run on the smoke output as a software check.
- **Scripted witness (exploratory, no learning; `witness_z.py`, `witness_z.json`).** Run at about 16:55, while the pilot was training and
  before any pilot run had finished. On the registered test weeks, a scripted "park and gamble" policy (a hedge from two empty accounts,
  otherwise the richer account flat and the other at maximal leverage) incurred no portfolio ruin and 63 account ruins per week from zero
  equity in the placebo market, reporting +33.0% of the allocation per week while booking -164.1%; under the portfolio ruin penalty at 5 or 25
  its reward is unchanged, above flat (0) and the honest policy (-94.0 and -524.8), while the account penalty at 5 gives it -282.1. From zero
  equity in the edge market the portfolio penalty at 25 scores it +53.5 against -52.8 for the honest optimum (leverage 4 with the signal in both
  accounts). From the registered (funded) starts in the edge market the honest optimum scores higher (+43.5 against +21.7 at strength 5), so the
  incentive lies at empty and low accounts, where the probes are taken. The witness changed no arm, measure or decision rule.
- **Pilot (exploratory, seed 99, eight runs).** Launched 16:37 (8 workers, `run_z.py --out pilot_results --seeds 99`, arms REP, LEDGER,
  RUINP5, RUINP25, RUINA5, SPR, P_REP, P_RUIN5); seven runs finished at 16:58-16:59 and SPR at 16:59:58 (21-22 minutes per run). Results in
  `pilot_results/` and `PREREGISTRATION_Z.md`; `aggregate_z.py --results pilot_results --print` and the verdict code (with its output written
  to a scratch folder outside this repository) were run on the pilot as software checks. The pilot changed the design before hashing: the
  portfolio penalty at strength 1 was dropped (72 runs), ZB2's threshold rule was replaced by a comparison with REP, ZB3 (the exploit's form)
  was added, and the verdicts are printed in ID order.

## After hashing (not part of the registered prefix)
- **17:03.** Hashes taken at 16:03:22 UTC (`hash_prereg_z.py`). Full run launched detached (`Start-Process -WindowStyle Hidden`, PID 21892,
  `run_full.log`): 72 runs (seeds 210-217), 7 workers. As in Studies X and Y, `run_full.log` prints each run's booked returns when the run
  finishes; those lines may be read to follow progress. The registered analysis (`aggregate_z.py`) is run once, after all runs have finished.
- **17:19.** `finish_z.py` (not registered; written after launch and before any run had finished) launched detached (PID 20716): it holds the
  computer's idle sleep while `run_z.py` works, and when `run_z.py` exits it runs the registered analysis once (if all 72 runs are present),
  then `make_studyZ_outputs.py`, and records both here. A first launch at 17:19:14 was stopped within a minute to correct a time stamp in its text.
- **22:40.** All 72 runs finished (the last at 22:40:14). The registered analysis (`python aggregate_z.py`) was run once, automatically by `finish_z.py` (written at 17:19, before any run had finished), and wrote `runs_z.csv` and `VERDICTS_Z.md`: 11 of 13 decisions supported. `make_studyZ_outputs.py` (the manuscript's tables and figure) was run afterwards.
