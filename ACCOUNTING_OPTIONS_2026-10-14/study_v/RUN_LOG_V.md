# Study V run log (append-only after "## After hashing")

All times are local (UTC+1, CET), 31 October 2026 unless stated.

- **Before this study (31 October, after Studies T and U).** A fresh read of the manuscript for referee objections (Claude) found three gaps: the
  remedies compared with the twin rewards were regularization toward a safe policy only (no ruin penalty, no risk-adjusted reward), the twin
  needs a clone of the simulator, and in the simple sleeve a trainer could compute the booked profit itself. Exploratory development in
  `exploratory_claude_2026-10-31/study_v_development/`: the pooled-magnitude theory (`THEORY_PMR.md`), its accuracy on random-action
  rollouts (`pmr_accuracy.py/.txt`), and two pilots (seeds 90-92): Part A arms (24 runs, `pilot_results/`) and Part C (8 runs,
  `pilot_results_c/`). A first launch of the Part A pilot (17:21, without the ledger-free audit) was stopped after 12 minutes when the audit
  was added to the runner (orphaned worker processes were stopped too), and relaunched with the audit at 17:34 (finished 18:18).
- **Pilot findings that shaped the design** (listed in `PREREGISTRATION_V.md`): PMR reached the ledger-trained policy in 3 of 3 pilot runs;
  mean-variance rewards destroyed the edge (the weight grid was moved down to {0.05, 0.2, 1}); the ruin penalty was sensitive to its strength;
  reported-reward PPO collapsed in one of three runs; the audit was within 0.10 point of the booked return for every pilot agent. The pooled
  mean of the PMR agents was 81.4 EUR/MWh against 76.7 for the training market's true mean: censored favourable branches can only raise the pool
  (Proposition on pooled magnitude, part 1). The PMR reward in the registered version uses the pooled mean before the current step's draw is
  added (the pilot added the step's own draw first, a change of weight 1 in 10^6).
- **Part C pilot** (launched 18:22, finished 18:50; 8 runs, seeds 90-91; the zero-start training distribution made each run about 27 minutes
  against 15 in Part A): reported-reward PPO gambled at all 16 probes in 2 of 2 runs; ORACLE, TIR and PMR trained to the same policy (booked
  +22.0%); the ruin penalty at 25 stopped the gambling and most of the trading (booked +1.8% and +11.9%). The design therefore tests all three
  ruin strengths in Part C and states VC6 as "no tested strength does both".
- **Smoke tests.** `run_v.py --smoke` (20,000 steps, one seed per arm; also with the audit and the zero-start test) passed; smoke results are
  excluded. The audit asserts that it replays the booked and reported returns of the test.
- **Not registered, for the record.** The clone-free wrapper `PooledMagnitudeWrapper` was added to `twin_reward/` with its tests (fair-sign
  expectation -c in every state, the pooled mean in [E m, 1.05 E m] on a toy environment). A toy test with large costs showed the upward bias
  of the pooled mean by censored favourable branches (0.315 against 0.239), as predicted.

## After hashing (not part of the registered prefix)
- **18:53.** Hashes taken at 17:53:53 UTC. Full run launched detached (`Start-Process -WindowStyle Hidden`, PID 25560, `run_full.log`): 144 runs (Part A 72, Part B 16, Part C 56), 8 workers.
- **21:10.** Part A complete (72 of 72 files). A descriptive look at those result files at 21:12 (before the aggregate was run) showed that the ruin penalties of strengths 5 and 25 would satisfy the registered conjunction in Part A, so VA5 would fail; nothing in the design or the analysis was changed.
- **23:16.** All 144 runs complete (about 4.4 hours; no errors, `run_full.err` empty, every audit assertion passed). `aggregate_v.py` was run at 23:17: **14 of 17 hypotheses supported; VA3, VA5 and VC6 not supported.**
  - VA3: the mean edge ratio of PMR in Part A was 0.861 (criterion 0.80) but its one-sided bootstrap lower bound was 0.478 (criterion 0.60), because one of the eight PMR agents (seed 151) acquired the exploit (X 0.5, booked -10.1%).
  - VA5 and VC6: the ruin penalty of strength 5 satisfied "both" in Part A (X 0.000, E +18.97 against ORACLE +20.89) and in Part C (X 0.000, E +17.19 against +19.84), and strength 25 in Part A (X 0.000, E +17.63); in Part C strength 25 stopped trading (E +4.18) and strength 1 kept gambling in both parts (X 0.336 and 0.641). The registered prior belief for VC6 (holds) was therefore wrong for strength 5.
  - All other hypotheses supported, including VA4 (no mean-variance weight both), VB (PMR in DK2 and DK1 weaker edge), VC1-VC5 and the audit (VD1: largest TIR error 0.385 and 0.129 points; VD2: 28 agents with X >= 0.25, median reported-return error 68.25 points, median ratio 0.0011).
- **23:20.** `make_studyV_outputs.py` written outputs (figure and two tables). Descriptive and post hoc, not registered: one-sided Mann-Whitney of the booked edge of PMR against RUIN5 gave p = 0.042 (Part C) and 0.113 (Part A); TIR against RUIN5 in Part C p = 0.193. In a later edit the header of the audit table was set in wrapping columns (cosmetic).
