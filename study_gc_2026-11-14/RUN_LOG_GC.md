# Study GC run log (append-only)

Local time (UTC+1), 14 November 2026. Written in an AI-assisted session (Claude in Claude Code) at the author's request; the author asked
for the study to be registered after reading the pilot's results.

- **21:04-21:05.** `envs_gc.py`, `run_gc.py` and `aggregate_gc.py` written, from Study G's runner and aggregator and the pilot in
  `Paper1/gym_mtsim_pilot_2026-11-14/`. The session then paused (usage limit) until 22:14.
- **22:15.** `test_envs_gc.py` written and run: all 18 checks pass (installed package equals the wheel's RECORD, 22 files; the control
  deletes exactly the three floor lines; placebo paths; margin limit at 27 balances; exposure probe against 280 opened or refused orders
  per market; on 300 random trajectories per market the booked ledger equals the control's equity within 4.8e-7 USD and the reading
  equals the booked step profit within 2.3e-5 USD, with 297 floor steps in crypto and 38 in forex).
- **22:17.** `reach_gc.py` run (first attempt cut by a 3-minute time limit after the crypto half; one summary statistic renamed and
  corrected to the standard deviation of the signed placebo return, then rerun in full): output `reach_gc.json`, quoted in the protocol.
- **22:21.** Smoke test, first attempt: the five worker processes were killed by a 3-minute time limit placed around the command, not by
  an error. Outputs deleted.
- **22:27-22:30.** Smoke test rerun without the limit: five runs of 16,384 steps finished in 102 to 121 s; `aggregate_gc.py --smoke`
  ran; the reading's largest error in LINE training 1.2e-9 USD. Smoke outputs kept in `results_gc/jobs_smoke/`, excluded.
- **22:33.** `PREREGISTRATION_GC.md` written with the numbers of `reach_gc.json`. The hashes are taken next; then the full run (40 runs,
  8 workers) starts.

## After hashing
- **22:32.** Hashes taken (`PREREGISTRATION_HASHES_GC.txt`, 2026-11-14T21:32:29+00:00, nine files); `hash_prereg_gc.py --verify`:
  all registered files match.
- **22:33.** Full run started: `run_gc.py --workers 8`, 40 jobs, the ten LINE runs first. Output `run_full.log`, `run_full.err`.
- **22:36.** Correction to the last entry above the marker: it should read 22:32, not 22:33. The protocol was finished at 22:32 and the
  hashes were taken at 22:32:29; the hashed text is left unchanged.
- **23:57.** No run finished after 84 minutes. The eight workers are computing (about 50 CPU-minutes each at 23:34). A timing check outside
  the study (3,000 steps per condition, PPO-like initial actions, while the run was using all cores) gave 2.0 ms per environment step for
  ORIG and 6.2 ms for LINE, so the LINE runs take about two hours each, not the 45 minutes assumed when the run was launched. Nothing was
  changed; expected finish of all 40 runs about 02:30 to 03:00.
- **00:27-00:38 (15 November).** First three LINE runs finished (seeds 15, 17, 12; 6,763 to 7,425 s).
- **01:30.** The computer was in Modern Standby from 00:37:21 to 01:29:50 (Windows Kernel-Power events 506 and 507); the eight worker
  processes survived and accrued about 4 CPU-minutes each during it, so the run paused for about 52 minutes and then continued. No run
  was restarted or changed.
- **00:41-01:42.** Five more LINE runs finished (seeds 11, 14, 16, 10, 13), so eight of the ten LINE runs are done.
- **02:55.** A second Modern Standby, 02:00:35 to 02:54:27, paused the run again (about 54 minutes). The computer is on battery
  (69% at 02:55), which is why the app's keep-awake did not hold. No run was restarted or changed.
- **03:01-04:25.** The other 32 runs finished (the computer was on mains power from about 03:00). All 40 runs finished; none failed and
  none was rerun. Wall time per run: LINE 6,763 to 11,223 s, the others 748 to 8,578 s (the longest include the standby pauses).
- **04:27.** `aggregate_gc.py` run once: GC1 to GC6 supported, GC7 not supported (the reading's largest error against the booked step
  profit is 4.398 USD, on one test step of crypto LINE seed 10). Output `results_gc/VERDICTS_GC.md`, `results_gc/runs_gc.csv`.
- **04:35.** `verify_completed_gc.py` (written during the run, not registered): hashes, the 40 runs, steps, test bank, package versions,
  saved policies, unique parameter digests and the 18 exact checks all pass; an independent recomputation agrees with all seven decisions
  (bootstrap lower bounds with 100,000 resamples and another generator: GC1 0.055, GC2 0.094, GC3 1.125, GC5 0.511; GC4 p = 0.00088).
- **04:40.** Post hoc, not registered: `diagnose_gc7.py` (output `gc7_diagnosis.txt`) replays crypto LINE seed 10 on the test bank. The
  only failing step (path 286, bar 95) holds a short of 0.16 BTC opened at 47,303.93 while the placebo price has risen to 125,578.65.
  gym-mtsim fixes the margin at the entry price (75.69 USD), so at a margin level of 0.202 the equity (15.29 USD) is 0.076% of the price
  away from the floor, inside the reading's bump of 0.1%. The upward bump crosses the floor, the slope comes out at 0.141 instead of 0.160
  USD per USD, and the reading gives -32.43 USD against a booked -36.83 USD. It is the only step above 0.001 USD among 1,356,000 audited
  test steps and 10,076,160 LINE training steps (largest training error 4.5e-9 USD), so it did not enter any LINE agent's training reward.
