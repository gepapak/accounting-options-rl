# Study XE run log (append-only)

Local time (UTC+1). Written in an AI-assisted session (Claude Opus 5.5 in Claude Code).

- **16 November, 13:00 to 13:30.** Package drafted (as `study_xe_draft/`) from the exploratory pilots in `../novelty_search_2026-11-16/`.
  - Scripts were copied from the pilot folder. `reading_engines.py` now also writes `reading_xe.json`; `census_mtsim.py` takes a seed
    and an output name; the census and PPO scripts take fresh seeds and an output folder.
  - The copies reproduce the pilot exactly: the census on the pilot's seed, 40 paths, gives a difference of 0.0.
  - `aggregate_xe.py` was dry-run on copies of the pilot outputs, which were then deleted. It ran end to end: XE1 to XE4 and XE6
    passed, and XE5 and XE7 failed on the seed counts, as they must with 4 seeds.
- **16 November, about 20:45.** The author approved registration ("Yes, register it").
  - The package was copied to `study_xe_2026-11-16/`, with an identical-copy check, and the draft folder was removed.
  - The status text in the protocol, the README, the runner and the aggregation docstring was changed from draft to registered. No
    design, seed, threshold or code logic was changed.
  - The hashes are taken next.

## After hashing
- **20:46.** Hashes taken (`PREREGISTRATION_HASHES_XE.txt`, 2026-11-16T19:46:10+00:00, 13 files).
  - `hash_prereg_xe.py --verify`: all registered files match.
  - Study GC's `envs_gc.py` equals Study GC's own registered hash (66d0195c...).
  - The computer is on mains power (battery 99%) and the app's keep-awake hold is granted for the session.
- **20:47.** Full run started: `run_xe.ps1` with the two environments, in the background. The console output goes to `run_full_xe.log`.
- **20:47 to 23:05 (139 minutes).** The run completed: the hash check, the census, the gym-mtsim census, the reading, 32
  two-action PPO runs, 24 direction PPO runs and the aggregation. No error in any log, no standby.
- **23:05. `aggregate_xe.py` (registered): 4 of 7 decisions supported.**
  - XE1 is supported: max |backtesting.py - backtrader| at L <= 5 is 9.4e-16.
  - XE2 is NOT supported. At L = 200, backtesting.py minus backtrader is +17.29 points, with a lower bound of -5.26. The other two
    parts held: reported minus own trade records +37.82 (lower bound +36.19), and backtrader minus ledger 1.2e-14.
  - XE3 is NOT supported. backtesting.py picks L = 200 in 100% of resamples, but backtrader avoids 200 in only 14% (60% needed).
  - XE4 is supported: the reading classifies every engine, gym-mtsim as published forgives +21.81 points at L = 200, and
    floor-deleted it is exact.
  - XE5 is supported: 12 of 16 backtesting.py seeds against 1 of 16 backtrader seeds, Fisher one-sided p = 8.5e-5.
  - XE6 is supported: reported minus booked is +26.94 points from 10,000 USD (seed-cluster lower bound +18.66) and +31.36 from 1,000
    USD (lower bound +24.56); backtrader is exact to 9e-13.
  - XE7 is NOT supported: backtesting.py has 6 of 12 short-majority seeds and 2 long-majority; backtrader has 4 and 3; Mann-Whitney
    p = 0.153.
- **23:10. `verify_completed_xe.py` (not registered; output `verify_completed_xe.log`).**
  - Hashes match, all 56 runs are present with the registered steps, actions, seeds, test seeds and tags, and there are no problems.
  - The decisions recomputed by a separate code path agree: XE5 at 12 against 1 with exact p 8.54e-5; XE7 at 6 and 2 against 4 and 3,
    U = 90 of 144; XE1 at 9.4e-16; backtrader exact.
  - The census's fresh paths have an upward drift: the ledger at L = 1 is +0.074% where the expected value, minus the fees, is
    -0.040%. At L = 200 this makes the booking engine report +14.76%.
  - **Post hoc reading:** XE2's engine difference and XE3's backtrader half compare reported means of a long-only trade at 200x, where
    the noise of 2,000 paths (about +/-13 points) exceeds the fee effect. Their thresholds came from one pilot sample whose drift was
    negative. The forgiveness itself (+37.8) is unaffected. The verdicts stand as registered.
  - Learned policies in XE5:
    - backtesting.py: 11 seeds bet 200x at every equity, 1 only near ruin, 2 partly, 2 flat.
    - backtrader: 14 flat, 1 bets 200x everywhere (s104), 1 only at high equity (s108).
