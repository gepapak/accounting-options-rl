# Study DR: does a wider training distribution switch on gym-mtsim's dormant forex floor?

Registered 15 November 2026: hashes taken at 2026-11-15T17:51:02Z, before the full run. The folder was run outside both repositories, like
Study GC; on 18 November 2026 it was copied byte for byte into the code repository, and the study is reported in the paper. Written in an AI-assisted session (Claude) at the author's request, after the exploratory
pilot, which the protocol discloses and the authors' archive keeps.

**Status: finished 15 November, 22:20 local time, and verified. One of the four registered decisions is supported.** All 64 PPO runs
finished without failure or standby: about 3.5 hours, eight at a time. `verify_completed_dr.py` confirms the hashes, the runs, the
versions and the exact checks, and it recomputes all four decisions independently with the same verdicts.

## The question in one paragraph

In Study GC, PPO never learned gym-mtsim's zero-balance floor in the package's forex market: at the default leverage of 100, one bar at the
largest position loses even under the published reward (-2.39% of the balance). Robust trading agents are often trained on randomized
market parameters. This study asks two things:

- Does widening the training distribution switch the dormant floor on?
- Does the published edge of the largest position, computed before training over the training distribution, predict where?

There are six arms:

| Arm | Change | Published edge C | Predicted |
|---|---|---|---|
| V2 | volatility randomized, k up to 2, hidden from the agent | -1.77% | dormant |
| V4 | volatility randomized, k up to 4 | +2.06% | intermediate |
| V8 | volatility randomized, k up to 8 | +12.53% | switches on |
| V16 | volatility randomized, k up to 16 | +34.42% | switches on |
| L30 | fixed leverage 30, the EU retail cap | -0.74% | dormant |
| L500 | fixed leverage 500, common offshore | +13.94% | switches on |

The decisions are registered as DR1 to DR4 in `PREREGISTRATION_DR.md`.

## Results

| Decision | Registered prediction | Result | Verdict |
|---|---|---|---|
| DR1 | V8 and V16 above the V16 floor-deleted controls | V8 +0.082 [-0.021, 0.160]; V16 +0.201 [0.077, 0.311] | Not supported (V8) |
| DR2 | L500 above its floor-deleted controls | +0.027 [-0.028, 0.085] | Not supported |
| DR3 | At most 1 learner of 6 in V2 and in L30 | V2 3 of 6; L30 0 of 6 | Not supported (V2) |
| DR4 | Exposure rises over V2 < V4 < V8 < V16 | J = 303 of 480 pairs, one-sided p = 0.035 | Supported, close to the threshold |

DR1 and DR2 are differences in mean low-equity exposure (xlow), with 95% bootstrap intervals. A learner has an xlow of at least 0.10.

| Arm | Published edge C | Training episodes reaching the floor | Learners | Mean xlow | Mean x(0.1) |
|---|---|---|---|---|---|
| V2 | -1.77% | 12.0% | 3 of 6 | 0.146 | 0.311 |
| V4 | +2.06% | 27.4% | 4 of 10 | 0.115 | 0.172 |
| V8 | +12.53% | 40.1% | 8 of 10 | 0.128 | 0.191 |
| V16 | +34.42% | 47.4% | 8 of 10 | 0.246 | 0.443 |
| L30 | -0.74% | 0% | 0 of 6 | 0.000 | 0.000 |
| L500 | +13.94% | 26.7% | 2 of 10 | 0.041 | 0.033 |
| V16, floor deleted | | 0% | 1 of 6 | 0.045 | 0.023 |
| L500, floor deleted | | 0% | 0 of 6 | 0.014 | 0.008 |

What the registered results say:

- Wider volatility ranges gave more low-equity exposure (DR4, close to the threshold), and the widest range beat its controls. The
  switch-on test failed for V8 because one control drifted (s604, xlow 0.271).
- Leverage 500 did not switch the floor on (2 learners of 10). The narrowest range, V2, which was predicted to stay dormant, had
  learners in half its runs.
- So the published edge computed before training failed as a predictor in both directions: L500 had a high edge and few learners, and
  V2 had a negative edge and many.
- From a full balance the agents barely trade. On the nominal test bank, reported and booked returns agree in every arm. Gaps appear
  only on the stressed bank, including in L30, which has no learners, so they are not specific to learners.

Post hoc, not registered (not tests):

- `posthoc_partial_size.py` computes the one-bar published edge at fractions of the largest order. The edge turns positive:
  - in L500 at 0.75 of the largest order, and below that it is the most negative of all arms;
  - in V8 at 0.5, and in V16 at 0.25;
  - in V4 only at full size;
  - in V2 never: it is negative at every size.

  This fits L500's shortfall but does not explain V2.
- The drifting control's exposure sits at equity 0.02 and 0.05, below and at the bottom of the training balance range (0.05 to 2). At
  equity 0.1, inside that range, every volatility arm is well above both control arms (table above). This is a post hoc choice of
  measure and changes no verdict. A later study should probe only inside the training range.
- `posthoc_2026-11-16/` (16 November) replays the 64 saved policies from 0.1 and 0.25 x 10,000 USD in gym-mtsim as published.
  - The registered learner label matches collecting forgiven loss in 63 of 64 runs.
  - The xlow magnitude misranks agents. Its points at 0.02 and 0.05 refuse orders that 9 of the 10 V8 agents request, so the drifting
    control (0.271) scores above every V8 learner, yet it collects less forgiven loss than six of them.
  - Two of V2's learners collect 4.6 and 9.0 points of forgiven loss and hold an order for 32 to 52 bars, which the one-bar edge does
    not count.
  - Across the arms, reach before training orders the share of learners (rank correlation 0.75) better than the one-bar edge does
    (0.38).

  Details are in `RUN_LOG_DR.md`.

## Files

| File | What it is |
|---|---|
| `PREREGISTRATION_DR.md` | Protocol: question, pilot disclosure, before-training numbers, predictions, design, measures, decisions DR1 to DR4 |
| `RUN_LOG_DR.md` | Append-only log; hashed up to its line "## After hashing" |
| `PREREGISTRATION_HASHES_DR.txt` | SHA-256 of the eleven registered files, including the two Study GC files imported |
| `envs_dr.py` | The arms, on Study GC's wrapper (imported read-only) |
| `test_envs_dr.py` | 38 exact checks (package unmodified, scaled placebo, training distribution, leverage, probe, ledger) |
| `before_dr.py`, `before_dr.json`, `before_dr_out.txt` | Before-training published edge, credit and reach per arm, and the hidden-leverage computation |
| `run_dr.py` | Runner (skips finished runs) |
| `aggregate_dr.py` | The registered decisions, writing `results_dr/VERDICTS_DR.md` and `results_dr/runs_dr.csv` |
| `hash_prereg_dr.py` | Writes or verifies the hashes |
| `verify_completed_dr.py` | After the run: hashes, the 64 runs, versions, saved policies, exact checks, and an independent recomputation of the decisions; not registered |
| `results_dr/` | One JSON per run (`jobs/`), saved policies (`policies/`); smoke test in `jobs_smoke/`, `policies_smoke/` and `VERDICTS_DR_SMOKE.md` (excluded) |
| `run_full.log`, `aggregate_full.log`, `verify_completed.log`, `smoke_run.log`, `smoke_aggregate.log` | Console output of the run, the aggregation, the verification and the smoke test |
| `posthoc_partial_size.py`, `posthoc_partial_size.json`, `posthoc_partial_size_out.txt` | Post hoc, not registered: the one-bar published edge at fractions of the largest order |
| `posthoc_2026-11-16/` | Post hoc, not registered (16 November). The fine probe and low-balance replay of the saved policies (`probe_actions_dr.py`, `tabulate_probe.py`), and an uninformative attempt at the initial policy's advantage (`initial_gradient_dr.py`), with their outputs |

## Rerun and check

Use Study GC's virtual environment (gym-mtsim 2.0.0, pandas 2.0.3; see `../study_gc_2026-11-14/README.md`):

```powershell
venv-mtsim\Scripts\python hash_prereg_dr.py --verify
venv-mtsim\Scripts\python test_envs_dr.py
venv-mtsim\Scripts\python run_dr.py --workers 8
venv-mtsim\Scripts\python aggregate_dr.py
venv-mtsim\Scripts\python verify_completed_dr.py
venv-mtsim\Scripts\python posthoc_partial_size.py      # post hoc, not registered
```
