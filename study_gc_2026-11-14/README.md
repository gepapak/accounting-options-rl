# Study GC: does PPO learn gym-mtsim's zero-balance floor in the package's own crypto market?

Registered study, run outside both repositories (as Study RS was), 14 November 2026. Copied byte for byte into the code repository and
reported in the manuscript on 18 November 2026.
Protocol: `PREREGISTRATION_GC.md`; hashes: `PREREGISTRATION_HASHES_GC.txt` (taken 2026-11-14T21:32:29Z, before the full run); run log:
`RUN_LOG_GC.md` (append-only; its digest covers it up to the line `## After hashing`). The protocol, code and registration were produced
by an AI assistant (Claude) at the author's request; the exploratory pilot that motivated it is disclosed in the protocol and kept in the authors' archive.

## What it tests

Study G's design (registered, 0 of 3 hypotheses supported, agents never traded) inside unmodified gym-mtsim 2.0.0, but with the market of
the package's registered crypto environment instead of a synthetic forex market, plus an arm trained on the reading of the published
reward (LINE) and a forex control. Seven registered decisions, GC1 to GC7.

## Files

| File | Role | Registered |
|---|---|---|
| `PREREGISTRATION_GC.md` | Question, design, measures, decision rules | yes |
| `envs_gc.py` | The wrapper, the control simulator, the placebo paths, the reading, the exposure probe | yes |
| `run_gc.py` | Training and test of every run | yes |
| `aggregate_gc.py` | The decisions; writes `results_gc/runs_gc.csv` and `results_gc/VERDICTS_GC.md` | yes |
| `test_envs_gc.py` | Exact checks (18), run before hashing | yes |
| `reach_gc.py`, `reach_gc.json` | Reach and credit of the floor, computed before training | yes |
| `hash_prereg_gc.py` | Writes or verifies the hash manifest | yes |
| `RUN_LOG_GC.md` | Append-only log | prefix |
| `verify_completed_gc.py` | After the run: hashes, run set, versions, exact checks, an independent recomputation of every decision | no, written during the run |
| `diagnose_gc7.py`, `gc7_diagnosis.txt` | After the run: the one step on which the reading failed, and why | no, post hoc |
| `results_gc/jobs/` | One JSON per run | outputs |
| `results_gc/policies/` | Trained policies and normalization statistics | outputs |
| `results_gc/jobs_smoke/`, `VERDICTS_GC_SMOKE.md`, `smoke_*.log` | Smoke test, excluded | outputs |

## Reproduce

Python 3.10 with numpy 1.26.4, gymnasium 1.2.0, stable-baselines3 2.7.0, torch 2.8.0 (CPU) and scipy, plus:

```powershell
python -m venv --system-site-packages venv-mtsim
venv-mtsim\Scripts\python -m pip install --no-deps gym-mtsim==2.0.0 pandas==2.0.3
venv-mtsim\Scripts\python test_envs_gc.py
venv-mtsim\Scripts\python run_gc.py --workers 8
venv-mtsim\Scripts\python aggregate_gc.py
venv-mtsim\Scripts\python verify_completed_gc.py
```

The full run is 40 PPO runs of 1,007,616 steps; with 8 workers on 8 cores it takes about two and a half hours. Training uses one torch
thread per run; bit-identical re-training has not been checked.

## Results (finished 15 November 2026, 04:25; six of seven decisions supported)

All 40 runs finished, none failed. `verify_completed_gc.py` passes: hashes, run set, versions, saved policies, the exact checks, and an
independent recomputation of all seven decisions.

| Decision | Result | Verdict |
|---|---|---|
| GC1: exposure at equity 0.1, floor on minus floor deleted (crypto) | 0.427 vs 0.131; +0.296, 95% interval 0.050 to 0.514 | supported |
| GC2: low-equity exposure (e up to 0.25), same contrast | 0.278 vs 0.078; +0.200, interval 0.095 to 0.289 | supported |
| GC3: floor-trained agents, reported minus booked test return | +3.36 points, interval 1.13 to 6.26 | supported |
| GC4: the reading removes the exploit | low-equity exposure 0.054 vs 0.278 (Mann-Whitney p = 0.0009), within the control's 0.078 + 0.10 | supported |
| GC5: reported-minus-booked gap, floor-trained minus reading-trained | +2.86 points, interval 0.50 to 5.82 | supported |
| GC6: forex negative control | no agent opened a position; difference 0, gap 0 | supported |
| GC7: the reading equals the booked step profit within 0.001 USD everywhere | 4.40 USD on 1 of 11.4 million audited steps | **not supported** |

Per seed (crypto, seeds 10 to 19), low-equity exposure: floor on 0.27, 0.19, 0.23, 0.30, 0.29, 0.12, 0.31, 0.38, 0.40, 0.28 (every seed
above 0.1); floor deleted 0.05, 0.04, 0, 0, 0, 0, 0, 0.40, 0.28, 0; reading-trained 0, 0, 0, 0, 0, 0, 0.18, 0, 0.04, 0.33. From
10,000 USD, six of the ten floor-trained agents ended at a zero balance on 23% to 63% of the test paths; the others did not trade at that
equity. Ending at zero is not specific to floor training: five of the twenty control agents (two floor-deleted, three reading-trained)
also traded from 10,000 USD and ended at zero on 5% to 59% of the paths, and one floor-deleted agent's reported-minus-booked gap is 6.4
points. The registered evidence of the learned exploit is the exposure contrast (GC1, GC2, GC4), not the test busts. Post hoc: exposure
at low equity exceeds exposure at full equity in all ten floor-trained agents and in three of ten in each control arm; this is descriptive only, since a fixed requested volume alone makes the share rise as equity
falls. During training
the floor wrote in about 6,250 of 8,256 episodes per floor-trained run, against 6,178 predicted before training for PPO's initial
policy; reading-trained runs reached it in about 5,580 episodes, with the full loss charged, and did not learn the exploit. In forex the
floor wrote on 97 to 871 training steps per run, and no agent learned to trade.

**Why GC7 failed** (post hoc; `diagnose_gc7.py`, `gc7_diagnosis.txt`): on one test step a reading-trained agent held a short opened at
47,304 USD per BTC while the placebo price had drifted to 125,579. gym-mtsim fixes the margin at the entry price, so the account was 0.076%
of the price from the floor, closer than the reading's bump of 0.1%; the bump crossed the floor and the slope came out 12% low (error 4.40
USD). This is the reading's stated condition (the reward must be affine over the bump) failing at an extreme state, not a coding error.
It is the only step above 0.001 USD among 1,356,000 audited test steps and 10,076,160 reading-trained training steps (largest training
error 4.5e-9 USD), so no training reward was affected. A bump smaller than the distance to the floor, or a check that the two one-sided
slopes agree, would have avoided it; neither was registered.
