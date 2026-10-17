# Study N run log

## 2026-10-14

**11:15 — DP benchmark (`dp_n.py`).** Computed before any training, for DK1 and DK2 at fee 0.10 EUR/MWh with lump-sum forgiveness at detection.
- ZFG: 16 at every insolvent probe; ZBK: flat.
- TFG: 16 at every probe with L ≤ 0.5·K0; TBK: flat.

**11:21 — Smoke test.** 16k steps, plumbing only.

**11:22–11:28 — Pilot 1.** Seed 0, DK1, 300,000 steps, TRAIN split, fee 0.10, lump-sum forgiveness. The design was frozen in `pilot1_archive/envs_n_pilot1.py` and `run_n_pilot1.py`; results are in `pilot1_archive/results_n_pilot1/`.
- **ZFG did not learn the DP policy.** |lev| was 1 at most insolvent probes and 4 at k = 12. The forgiveness arrives as one lump sum up to 96 quarter-hours after the losses.
- **ZBK took 16 at 8 of 12 insolvent probes and at most solvent probes.** This is the Study F problem: at 0.10 EUR/MWh the fee that makes flat optimal is too small against the reward noise to be learned (cf. F2 not supported, H2 supported at 1.0 EUR/MWh).
- **TFG learned the DP policy.** It took 16 at L ≤ 0.25·K0 for every τ, and at negative L except (−0.25, τ ≤ 4).
- **TBK stayed near flat.** It took 0.25 at most probes and 4 at τ ≥ 96 for low L.

**Design changes after pilot 1 (made before registration, disclosed):**
1. **Fee raised to 1.0 EUR/MWh in every arm** (`FEE_N`), the Study H level at which full-liability agents learned to stay flat (H2, 10/10). The DP at this fee (`dp_n2.py`) still predicts:
   - ZFG: 16 at every insolvent probe (action gaps 2–44 EUR) and 16 when solvent up to 0.5·K0;
   - TFG: 16 at L ≤ 0.25·K0 for τ ≤ 96, but **flat at τ = 192** (waiting is optimal far from the horizon);
   - ZBK and TBK: flat.
2. **ZFG reporting.** Once insolvent, the zombie arm reports the owner's claim max(L, 0) and is rewarded on its change; the deficit is not paid as a lump-sum bonus at detection. The terminal payoff is identical (max(L_close, 0)); only the timing of the reward within the window changes. ZBK, TFG and TBK are unchanged.

**11:50 — Pilot 2.** The registered design, seed 0, DK1, 1,000,000 steps, TRAIN split. The outcome is recorded below, before the hashes are taken.

**11:36–11:49 — Pilot 2 outcome.** Registered design, seed 0, DK1, 1,000,000 steps, TRAIN split; results in `results_n_pilot/`.
- **ZFG matched the DP.** |lev| 16 at all 12 insolvent probes. When solvent, 16 up to 0.5·K0 and 4 above.
- **ZBK was near flat.** It took 0.25 at 9 of 12 insolvent probes and 1 at the other 3, and 0.25 at every solvent probe.
- **TFG mostly matched the DP.**
  - It took 16 at L ∈ [−0.1, 0.25]·K0 for τ ≤ 96, and 16 at L = 0.5·K0 for τ ≥ 48.
  - It departed from the DP in two places: 16 at τ = 192 for L ≥ 0.02·K0, where the DP is flat; and flat at L = −0.25·K0 for every τ, where the DP gambles.
- **TBK was near flat.** It took 0.25 at most probes, and 4 at negative L or long τ.
- **TRAIN-split evaluation:** ZFG reported −68.1 % against −98.5 % booked; ZBK −15.5 % against −15.5 %.

No further design change. The hashes are taken next, then the full study is launched: 40 runs, TEST-split evaluation.

## After hashing (not part of the registered files)

- **11:49.** The full run was launched from a shell. It was stopped within a minute, before any job finished, and relaunched as an independent process (PowerShell Start-Process, PID 22628, `run_full.log`) so that the session's time limit cannot kill it. No job had completed; the runner skips only finished jobs.
- **Erratum.** The line "11:50 — Pilot 2" above gives the planned start time. Pilot 2 actually ran from 11:36 to 11:49, as its outcome entry says.
- **About 13:05 — Full run complete.** 40 of 40 runs, no failures. `aggregate_n.py` (registered) wrote `VERDICTS_N.md` and `runs_n.csv`.
  - Supported (6 of 9): N1, N2, N3, T1, T2, T3.
  - Not supported: N4, T4, X1.
  - Figure: `fig_studyN.png` (`make_studyN_outputs.py`, written after the runs, not registered).
