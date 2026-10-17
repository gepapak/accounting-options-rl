# Study S run log (append-only)

All times are local (UTC+2), 17 October 2026.

- **Before this study (16-17 October).** Exploratory development in `exploratory_claude_2026-10-16/nfl_training/` (seeds 80-89):
  pilot v1 (testbed failed, episode-level placebo variant over-conservative), pilot v2 (SPR passed on the zero floor, DK1), stage 2
  (four accounting rules, KL baseline), stage 3 (TERM/LAG with insolvent starts, DK2, A2C: no learned exploit or invalid), stage 4
  (Study O/Q environment; pre-written rule failed for PPO and DQN against the registered controls) and a post-hoc same-seed diagnostic.
  All of it is recorded in that folder's PLAN.md, including a correction of its low-equity probe measure (unreachable negative-equity
  probes), which is why Part A of this study uses reachable probes only.
- **About 06:55.** `PREREGISTRATION_S.md`, `envs_s.py` (frozen copy of the exploratory edge-market testbed and its market draw), `spr_s.py`,
  `run_s.py` and `aggregate_s.py` written. `aggregate_s.py` tested on synthetic run files in a scratch folder (no file written here).
- **06:58.** Smoke test `run_s.py --smoke` (20,000 steps, seed 100 for Part A, seed 110 and DK1 for Part B, all 12 arm-learner cells)
  started. The computer slept during the test (about 07:00 to 12:41; Windows resume events), which suspended it; it resumed and completed
  at 12:44: all 12 runs completed, every greedy-policy exactness check passed. Smoke results (`results_s/smoke/`) are excluded.
- **12:45.** The hashes are taken next; then the full run (180 runs, 8 workers) starts, with the app keeping the computer awake.

## After hashing (not part of the registered prefix)
- **12:46.** Full run launched detached (Win32_Process Create, cmd PID 19780, `run_full.log`): 180 runs, 8 workers.
- **14:51.** Part A complete (60/60 runs, no failure). The unchanged registered `aggregate_s.py` (hash checked) was run on a scratch copy of `results_s/A` so that `VERDICTS_S.md` is written here only once all 180 runs are complete; Part A verdicts do not depend on Part B (the Part A bootstrap draws come first). SA1, SA2, SA3, SA4 all SUPPORTED: X_A PLAIN 0.850 vs ORACLE 0.056 (p = 0.0001); SPR 0.000; E SPR +22.58 / ORACLE +18.05 = 1.251 (LB 0.917); no KL beta did both.
- **17:59.** All 180 runs complete (last at 17:57:44), no FAILED entry. The unchanged registered `aggregate_s.py` (hash checked) was run here and wrote `runs_s_A.csv`, `runs_s_B.csv` and `VERDICTS_S.md`. All eight hypotheses SUPPORTED. Part A as at 14:51. Part B: PPO shape ZF +12.15, SPR +2.73, LEDGER +1.39 (SB1 mean +9.43, p = 0.0005; SB2 mean +1.34, UB +3.51); A2C-n64 ZF +7.76, SPR +0.72, LEDGER +3.00 (SB1 +7.04, p = 0.0020; SB2 -2.27, UB +0.33).
