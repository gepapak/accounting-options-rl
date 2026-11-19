# Same-engine control for Study XE: backtesting.py with its balance floor removed (exploratory, 17 November 2026)

**Not registered (exploratory, with predictions fixed before the runs).** Written in an AI-assisted session (Claude Fable 5.2 in Claude
Code). Reported in the paper and copied byte for byte into the code repository on 17 and 18 November 2026.

## Why

Study XE (`../study_xe_2026-11-16/`, registered) compared PPO trained through backtesting.py with PPO trained through a different
engine, backtrader. The cleanest control is the same engine with only its floor changed, as in Studies G, GC and DR (FLPATCH).

## What changes

`rl_engine_env.py` and `rl_engine_pilot.py` are copies of Study XE's registered files with two additions:

- `BookedBroker`: backtesting.py 0.6.6's own `_Broker.next()` with its two floor lines, `self._cash = 0` and `self._equity[i:] = 0`,
  replaced by `self._equity[i:] = self._cash`, the equity the closed trades left. The account still closes at ruin; the loss beyond
  the account is reported instead of forgiven. `verify_ctl.py` found two other differences after the run (this text first said
  "line for line"): the package's comment line above the branch is dropped, and one added line clears the cached unrealized P&L after
  the trades close. That line changes no number: the booked equity is identical to the published engine's on every path.
- `BtpyBookedEnv` (engine name `btpyfl`), which runs the wrapper on that broker.

The installed package is not modified.

## Exact checks before training (`check_control.py`, `check_control_out.txt`)

On 300 fair paths with "always 200x":
- the booked equity is identical in the two engines on every path (difference 0.0), and the ruin flags are identical;
- the published engine reports +18.55% where it books -20.06%;
- the control reports -20.06%, equal to its own trade records on every path; its lowest reported equity is -18,420 USD.

## Design

Study XE's part C, unchanged: two actions (flat or 200x), fair placebo on gym-mtsim's EURUSD magnitudes, PPO with 507,904 steps.
Both arms use the same eight fresh seeds (301 to 308), so each pair starts from the same weights and the same path draws. The test
seed is 20261211.

## Predictions, written before the runs (17 November, 06:29)

This file was first saved at 06:29:15 and the run was launched in the same minute (`run_ctl_err.txt` was created at 06:29:15). The
heading first gave the time as 06:40, an estimate that was wrong; it was corrected at 06:36 without changing the predictions.

A learner bets 200x at half or more of the nine low-equity probes, as in XE5.

- Published engine: at least 5 of 8 learners (Study XE had 12 of 16).
- Floor removed: at most 1 of 8 learners (backtrader had 1 of 16).
- Floor removed: reported equals booked on every test path (difference below 1e-6 points).
- Published engine: reported minus booked above 15 points from 10,000 USD (Study XE: 26.9).

## Results (run finished 07:52 on 17 November; scored by Claude Opus 5.5 after the session's model was switched)

**All four predictions met** (`score_ctl.py`, `score_ctl_out.txt`). The definitions are those of Study XE's registered aggregator.

| Prediction | Outcome |
|---|---|
| Published engine: at least 5 of 8 learners | **6 of 8** |
| Floor removed: at most 1 of 8 learners | **0 of 8** |
| Floor removed: reported = booked on every test path | **largest difference 0.0 points** |
| Published engine: reported - booked above 15 points from 10,000 USD | **22.80** (seed-cluster lower bound 9.12) |

What each agent learned (probes: equity 0.05, 0.1, 0.25, 0.5, 1 and 2, three time points each; `#` = 200x, `.` = flat):

| Seed | Published engine | Over-report from 10k / 1k (points) | Floor removed |
|---|---|---|---|
| 301 | `### ### ### ### ### ##.` | 36.48 / 36.48 | flat everywhere |
| 302, 303, 306, 307 | `### ### ### ### ### ###` | 36.48 / 36.48 | flat everywhere |
| 304 | `### ### ### ##. ... ...` (200x only when equity is low) | 0.00 / 36.48 | flat everywhere |
| 305 | `... ... ... ... ... ...` (bets on some paths from 1k) | 0.00 / 23.47 | flat everywhere |
| 308 | `... ... ... ... ... ...` (bets on some paths from 1k) | 0.00 / 19.77 | flat everywhere |

Seed 304 is the gamble-for-resurrection policy in its plainest form: flat when the account is healthy and 200x when it is near ruin.
Its twin with the floor removed, from the same initial weights and training paths, never trades.

Post hoc:
- Fisher one-sided p = 0.0035 (6 of 8 against 0 of 8).
- Paired by seed: 6 pairs learn only with the floor and none only without; exact sign test p = 0.016.
- With the registered Study XE: the published engine 18 of 24 against the floor removed 0 of 8, p = 2.9e-4 (backtrader had 1 of 16).
- The six learners from 10,000 USD report +28.96% and book -1.44%.

Independent checks (`verify_ctl.py`, `verify_ctl_out.txt`):
- The broker difference is as described above.
- Study XE's two wrapper files match the registration's hash manifest (`PREREGISTRATION_HASHES_XE.txt`). The control's copies change
  one line each and otherwise only add code.
- Fresh engine runs on the regenerated test bank reproduce the recorded draws exactly. Seeds 302, 303, 306 and 307 equal "always 200x"
  from both starts; seed 304 equals "always flat" from 10k and "always 200x" from 1k; all eight controls equal "always flat".
- The monitoring law, written at 01:50 before this run, gives 37.47 points forgiven at 200x with P(ruin) 0.828. This test bank gives
  36.48 (standard error 2.72) with a ruin share of 0.825.

What it adds: Study XE compared two different engines. Here the engine, the agent, the seeds and the paths are the same, and only
the floor lines differ. Learning to bet 200x near ruin appears with the floor (6 of 8) and never without it (0 of 8). The evidence is
stronger; the claim is the paper's existing one.

## Rerun

```powershell
venv-vbt\Scripts\python -B check_control.py
venv-vbt\Scripts\python -B rl_engine_pilot.py --steps 500000 --seeds 301 302 303 304 305 306 307 308 --engines btpyfl btpy --workers 8 --levels 0 200 --tag _ctl --test-seed 20261211 --out results_ctl
python -B score_ctl.py
venv-vbt\Scripts\python -B verify_ctl.py
```
