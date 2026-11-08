# Study EXT run log (append-only after "## After hashing")

- 6 November 2026, evening: harness written (`envs_ext.py`, `run_ext.py`, `scan_ext.py`), official source vendored byte for byte from the archived pinned copy, scripted tests of the accounting (`scripted_ext.py`, `scripted_ext2.py`) and smoke tests (`smoke_results/`, 20,000 steps, excluded). The first smoke test was interrupted when the laptop's lid closed and the machine restarted; it is not part of the study.
- Pilot at 10^6 steps, seed 99, arms REP, LEDGER, FINDER (`pilot_results/`, `pilot.log`). `run_ext.py` then gained the width in the test and the FINDER_DEF arm, and the first version of `scan_ext.py`'s jump search (wrong interval) was replaced before any result of it was read for a decision.
- `test_envs_ext.py` passed: 60,000 steps, 133 busts, 6,000 deep-copy comparisons.

## After hashing
- 2026-11-06 20:50 local: hashes written; scan_ext.py run; launched run_ext.py --workers 3, PID 2792.
- 2026-11-06 20:55 local: the runner reports 70 jobs where the hashed text says 62: a counting error in the text, see ERRATA_EXT.md. Nothing else changed.
- 2026-11-07 06:47 local: all 70 runs complete, 0 failed; aggregate_ext.py run once: 16 of 17 decisions supported, EA9 not. make_tab_ext.py added after the analysis; it computes nothing new.
- 2026-11-07 07:26 local: exploratory and not registered, written after the registered analysis: explore_log_reward.py, the engine's default logarithmic reward, seeds 300-307, 8 runs, none failed (results_log/SUMMARY.md): X at most 0.011, no bust. The hashed files are imported unchanged.
