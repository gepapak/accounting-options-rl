# The reward diagram of the running engine: protocol

**Status: exploratory, not a registered study.** No agent is trained: every run re-evaluates a frozen checkpoint of the engine audit. The rules below, the job list and the code are fixed and hashed (`PROTOCOL_SCAN_HASHES.txt`) before the full evaluation.

**Seen first.** Four pilot runs (MAPPO seed 7, DK1: L3, L7, L7 without the percentage payoff, L7 with the cap on; `line_pilot/`, `analyze_line.py`) were analysed before this file was written, on a sample of steps and with an earlier version of the estimator. The rules generalize what those pilots showed, so they are not out-of-sample predictions for those four runs, which are also part of this job list. A tolerance error in `reward_diagram.read_diagram` that made one step of the L7 pilot err by 116 points (slope equality was absolute, on slopes of order 7e8 DKK) was found and corrected on the pilots, before this file.

## Question
On the engine's own paths the sign twin's point estimates each failed on one payoff type, and the bracket is wide (Supplementary Section "The sign twin on the engine's own paths"). That check replayed the cash writes. Here every query runs the engine's full step on a real clone with the market input moved along a line. Can the booked P&L, the hidden balance and the contract's cap be read from the reward alone?

## Runs
`SCAN_JOBS.txt`, 26 runs: V, volume payoff, floor bound (10); K, percentage payoff with its cap on (4; the floor binds in seed 5001); N, nothing forgiven (6); P, percentage payoff uncapped as in the audit (6; floor bound in seeds 7 and 123, not in seed 42).

## Method
`scan_launcher.py`. Every step with `i % 6 == 0` is read adaptively: t = +1, -1 from real clones; if the bracket's width is non-zero also t = 0, +-2, +-4, +-8; if that shows no stable slope although the pair differs, also +-16, +-64, +-256, +-1024. A sample of the steps (3% funded, 10% empty) also gets the full grid of 23 points up to |t| = 2^22, zero-cost clones and funded clones (cash raised by 1e13 DKK). Clones form reference cycles, so the launcher runs the garbage collector every 25 read steps; a development smoke test (`scan_smoke/`, stopped early, excluded) found this before the campaign, when memory grew to 2 GB in 3,000 steps.

The estimator (`analyze_scan.py`, using `reward_diagram.read_diagram`) sees only the clones' rewards and the declared own cost C (access, roll, transaction, funding, less the battery flow): where the width is within tolerance it returns phi(1), otherwise tail gross minus C. The ledger is used only to score it. A run's audit is the sum of the estimates over the read steps minus C on the other steps. In the volume payoff every non-zero gross lies on the read steps (checked on the audit's traces); in the percentage payoff the gross lies on every step, so for group P the errors are reported per read step, and no run-level audit is claimed.

## Gates (for a run to count)
- **G1.** The clones leave the run unchanged: its trace equals the audit trace in `engine_twin_2026-11-05/traces/` within 1e-3 DKK in r and in cash at every step (for the cap-on group, the trace of the same job there).
- **G2.** phi(1) equals the engine's reported change within 1e-3 DKK at every read step.

## Rules
Tolerances on the width and on equality are in DKK, 1e-3 plus 1e-9 of the account. Points are percent of the initial sleeve.
- **S1, the twin is real.** The clone's phi(-1) equals the earlier replay's r_twin within 1 DKK at every read step of groups V and N.
- **S2, the bracket on real clones.** -phi(-1) - 2C <= x <= phi(1), within 1 DKK, at every read step of all 26 runs.
- **S3, reading the booked return.** In every run of group V the run-level audit is within 0.01 points of the booked return, and closer than the TIR and the SPR audit built from the same clones.
- **S4, step exactness.** In every run of group V the estimate is within 1 DKK of the booked P&L on at least 98% of the read steps.
- **S5, no harm where nothing is forgiven.** In every run of group N, and in the K runs of seed 7, the run-level or read-step error is at most 0.01 points.
- **S6, with cap and floor together.** In the K runs of seed 5001 the summed absolute error over read steps is at most 1 point. (Least certain: no pilot had both.)
- **S7, the percentage payoff.** In every floor-bound run of group P the summed absolute error over read steps is below TIR's, and at most 2 times SPR's.
- **S8, the configuration error, from the reward alone.** The share of sampled moved steps with a concave kink at |t| <= 2 is at least 90% in all four K runs and at most 5% in all six P runs, so a threshold of 50% separates the capped from the uncapped protocol in all ten.
- **S9, the hidden balance, from the reward alone.** In group V, on sampled steps with cash above 1,000 DKK and a flat stretch below phi(0) in the full diagram, the flat level equals minus (cash + bucket) within 1 DKK on at least 99% of the steps.
- **S10, cost of the reading.** Mean clones per read step at most 8 in group V and at most 2.5 in group N.

Reported without a rule: the funded and zero-cost clones as comparators (they need write access to the balance or the cost settings, which the reading does not); the kink counts; the share of steps at each stage; for group P, whether the steps the reading misses have a kink beyond the range the input's clip admits, which the range law predicts.

## Reading
Every outcome is reported, whichever way it goes. The engine exposes its position and its gross P&L, so no outcome here can show that a reading is *needed* on this engine; the direct calculation with the formula in force is exact (E6 of the earlier check). The check asks whether the reading is correct on realistic accounting and on a real, unmodified step. Technical corrections are allowed only before outcomes are read and are recorded in `RUN_LOG_SCAN.md`.
