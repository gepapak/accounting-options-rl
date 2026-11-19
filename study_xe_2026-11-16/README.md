# Study XE: cross-engine verdicts

**Status: registered 16 November 2026 on the author's approval (hashed 19:46 UTC); run 20:47 to 23:05; verified. 4 of 7 decisions
supported.** The folder was run outside both repositories, as Studies GC and DR were; on 18 November 2026 it was copied byte for
byte into the code repository, and the study is reported in the paper (Section 4.2 and the supplement).

## Results

| Decision | Registered prediction | Result | Verdict |
|---|---|---|---|
| XE1 | The engines agree exactly for L <= 5 | max difference 9.4e-16 on 2,000 paths | Supported |
| XE2 | At L = 200, backtesting.py exceeds backtrader and its own trade records; backtrader = ledger | difference +17.29 [lower bound -5.26]; forgiven +37.82 [+36.19]; ledger 1.2e-14 | Not supported (the engine difference) |
| XE3 | An optimizer picks L = 200 in backtesting.py (>= 95%) and avoids it in backtrader (>= 60%) | 100%; 14% | Not supported (the backtrader half) |
| XE4 | One query classifies every engine | as predicted; gym-mtsim forgives +21.81 at L = 200, floor-deleted exact | Supported |
| XE5 | PPO through the engines, two actions: backtesting.py >= 9 of 16 learners, backtrader <= 1 of 16, p < 0.01 | 12 of 16 against 1 of 16, p = 8.5e-5 | Supported |
| XE6 | Reported minus booked for the learned policies: positive in backtesting.py, exact in backtrader | +26.94 [+18.66] from 10,000 USD, +31.36 [+24.56] from 1,000; backtrader 9e-13 | Supported |
| XE7 | At the defaults, backtesting.py agents short (>= 7 of 12, <= 1 long), more than backtrader (p < 0.05) | 6 short, 2 long against 4 and 3; p = 0.153 | Not supported |

**What the registered results say.**
- PPO trained through an unmodified mainstream backtester learns its zero floor (XE5), and the same agent through an engine that books
  the loss does not. Learned agents report 27 to 31 points more than they book (XE6).
- The engines agree exactly inside the range (XE1), and one query past the kink classifies them (XE4).
- The direction bias at the default settings was not learned reliably (XE7).
- The engine-difference parts of XE2 and XE3 failed. On the census's fresh paths, the booking engine itself reports +14.76% at L = 200,
  because the sample drifts upward (+0.074% at L = 1, against -0.040% expected).

**Post hoc, not registered (see `RUN_LOG_XE.md`).**
- XE2 and XE3 compared reported means of a long-only trade at 200x. There, 2,000 paths leave noise of about +/-13 points, larger than
  the fee effect, and the thresholds came from a single pilot sample.
- The forgiveness itself (reported minus own trade records, +37.8 with a lower bound of +36.2) was not affected.
- Learned XE5 policies: in backtesting.py, 11 of 16 bet 200x at every equity; in backtrader, 14 of 16 stay flat.

- Protocol: `PREREGISTRATION_XE.md`.
- Exploratory pilots that set the thresholds: `../novelty_search_2026-11-16/README.md`.

The study asks whether two mainstream backtesting engines that agree on every solvent trade report opposite verdicts once a trade can
lose more than the account. backtesting.py sets negative equity to zero and stops; backtrader books the loss. It also asks:
- whether an optimizer of the reported return, a grid search or PPO trained through the unmodified engine, picks opposite policies;
- whether at the default settings the accounting picks the trade's direction;
- whether one query past the kink classifies the engines.

The decisions are XE1 to XE7.

To rerun (about 4 to 5 hours):

```powershell
venv-vbt\Scripts\python -B hash_prereg_xe.py --verify
.\run_xe.ps1 -vbt <path to venv-vbt python.exe> -mtsim <path to venv-mtsim python.exe>
```

The environments used are recorded in `PREREGISTRATION_XE.md`. They live in the session's scratch folder, like Study GC's
`venv-mtsim`.

| File | What it is |
|---|---|
| `PREREGISTRATION_XE.md` | Question, pilot disclosure, design, fresh seeds, decisions XE1 to XE7 |
| `PREREGISTRATION_HASHES_XE.txt` | SHA-256 of the registered files, including Study GC's `envs_gc.py` |
| `run_xe.ps1` | The exact commands, in order |
| `aggregate_xe.py` | The decisions, written to `VERDICTS_XE.md` |
| `hash_prereg_xe.py` | Writes or verifies the hashes |
| `engines_pilot.py`, `census_mtsim.py`, `reading_engines.py` | Parts A, B and D |
| `rl_engine_env.py`, `rl_engine_pilot.py` | Parts C and E: PPO through the unmodified engines |
| `eurusd_mags.npy`, `crypto_mags.npy` | Placebo magnitudes from gym-mtsim's bundle |
| `RUN_LOG_XE.md` | Append-only log, hashed up to its line "## After hashing" |
| `VERDICTS_XE.md`, `aggregate_xe_out.txt` | The registered decisions |
| `verify_completed_xe.py`, `verify_completed_xe.log` | After the run, not registered: hashes, completeness, and an independent recomputation of the decisions |
| `xe_census.csv`, `xe_census_mtsim.csv`, `reading_xe.json`, `results_xe/` | Outputs of parts A, D, C and E; the `*_out.txt` files hold the console output |
