# Study O run log (append-only)

All times are local (UTC+2), 14 October 2026.

- **About 16:35.** Design calculation `dp_o.py` launched for DK1 and DK2 with alphas {0, 0.5, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 1}.
- **16:47.** Smoke test `run_o.py --smoke` (16,384 steps, TRAIN split, alpha 0 and 1, DK1, seed 30): both runs completed, untrained policies (|leverage| 1 everywhere). Excluded from all results.
- **About 16:52.** Design calculation stopped after alpha = 0.8 (the index falls almost linearly in 1 - alpha, so the arms were moved to span the calibrated threshold) and relaunched with the final arm set {0, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 1}.
- **17:00.** Design calculation complete; `calibrate_o.py` run (c* = 0.513%); `aggregate_o.py` tested on synthetic run files.
- The pre-registration is written; the hashes are taken next, then the full run is launched (80 runs, TEST-split evaluation).

## After hashing (not part of the registered prefix)
- **17:03.** Hashes taken (15:03:37 UTC). Full run launched as an independent process (PowerShell Start-Process, PID 22864, `run_full.log`): 80 runs, 7 workers.
- **Erratum.** The registered entry "17:00 Design calculation complete" is imprecise: the relaunched design calculation finished at 16:54 (file times of `dp_o_*.json`); 17:00 is when the calibration and the synthetic aggregation test were run.
- **21:28.** Full run complete: 80 of 80 runs, no failures. `aggregate_o.py` (registered) wrote `VERDICTS_O.md` and `runs_o.csv`: O1, O2, O3 and O4 supported.
