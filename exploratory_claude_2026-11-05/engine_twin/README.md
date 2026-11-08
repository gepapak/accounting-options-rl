# The sign twin on the engine's own paths (exploratory, 6 November 2026)

**Status: exploratory, not a registered study.** No agent was trained: every run re-evaluates a frozen checkpoint of the engine audit, so the programme's agent count is unchanged. Rules, job list and code were fixed and hashed (`PROTOCOL.md`, `PROTOCOL_HASHES.txt`) before the full evaluation, after three pilot runs, which are disclosed there.

## Why it exists
A review of the paper raised two questions that the sleeves could not answer. Does the twin bracket hold on the engine's own evaluation paths, or only on hand states? And why would a trainer who can reverse the market input not compute the booked profit directly? While answering them, the tracer showed that **the audit's legacy protocol L7 had run its percentage payoff uncapped**, against the paper's description.

## Result
Decisions are in `results/VERDICTS.md`. Over 93 runs (54 in which the floor bound, 39 in which it did not; 3,655,923 steps):

| Rule | Outcome |
|---|---|
| G1, G2 gates | Met: each run reproduces the audit's return within 1e-12 points; the replay reproduces the engine's cash exactly; no cash write of unknown kind |
| E1 forgiving on the real path | Supported |
| E2 bracket contains booked P&L | Supported: 0 steps outside, 93 of 93 runs contained |
| E3 ledger-free reconciliation | Supported: reported minus summed TIR flags the same 53 of 54 runs as the ledger, and 0 of 39 |
| E4 width | Supported: the same 53 of 54, and 1 of 39 (no rule), where only the twin branch would have been floored, on one step |
| E5 point estimates split by payoff | **Not supported**: TIR closer in 24 of 27 volume-payoff runs (rule: all), SPR closer in 27 of 27 percent-payoff runs |
| E6 direct calculation | Supported: exact with the formula in force; over 5 points off with the other generation's formula in 53 of 54 |
| E7 cost-shift test | **Not supported**: flags 53 of 54 runs (rule: all; the other touched the floor for 0.7 points) and 99.9% of the steps where TIR errs by over 0.01 points |

**The bracket is what held.** Median audit errors, in points: with the volume payoff TIR 43 and SPR 618; with the percentage payoff TIR 608 and SPR 11. The engine exposes its position and its gross P&L, so none of this shows that a twin is *needed* on this engine: the check asks whether the bracket survives realistic accounting.

**The cap.** With the cap really on (22 runs), MAPPO reports means of -38.5% (DK1) and -45.0% (DK2) against -7.9% and -13.7% for the rule, and the floor binds in 2 of its 20 runs. As run, without the cap, L7 reported +270% and +122% against -36% and -41%. `config_dump/DIFFERENCES.md` lists every setting that differs from the strict protocol at runtime, for every protocol: the cap is the only setting that differed from the paper's description (the flat fee is set in L7 but acts only with the volume payoff).

**A tolerance artifact.** With its default tolerance the cost-shift test also fired in 26 of the 39 comparison runs, because that tolerance scales with the pair's own values while cash near 1e9 DKK rounds at about 1e-7 DKK. With the tolerance scaled to the account (`posthoc_shift_tolerance.py`, post hoc) it fires in 2 of the 39, on 2 steps, and still in 53 of the 54.

## Files
| Path | What it is |
|---|---|
| `PROTOCOL.md`, `PROTOCOL_HASHES.txt`, `JOBS.txt`, `make_jobs.py` | Rules, hashes and the job list. The hashes of `engine/*` refer to the audit engine built from the private `Prototype5/`, which differs in formatting from `../../engine/` |
| `twin_trace.py`, `twin_launcher.py`, `run_engine_twin.py` | Read-only tracer of every write to trading cash, its launcher and the job runner |
| `analyze_engine_twin.py`, `summarize_engine_twin.py`, `check_against_audit.py` | Per-run results, decisions and the gate |
| `dump_runtime_config.py`, `config_dump/` | The settings each protocol really ran with, read from the engine at its first step |
| `inspect_results.py`, `posthoc_shift_tolerance.py`, `mechanisms.py` | Post-hoc: the runs behind each decision, the test's tolerance, and the mechanisms the text gives for the errors |
| `results/` | `engine_twin_runs.csv` (one row per run), `VERDICTS.md`, `engine_twin_summary.json`, `tab_engine_twin.tex`, `check_against_audit.txt`, `shift_tolerance.*`, `mechanisms.json`, `run_log.txt`, `trace_sha256.txt` |
| `traces/` | Five of the 115 step traces (the others total about 290 MB; their digests are in `results/trace_sha256.txt`) |
| `pilot_2026-11-06/` | The tracer's first version, with the bug noted in `PROTOCOL.md`, and the pilot's run log |

## Reproducing
`../../tools/verify_studies.py` checks the hashes, re-derives the decisions from `results/engine_twin_runs.csv`, recomputes the five shipped traces and compares the run table with the audit's table. Re-running the evaluations needs the frozen checkpoints and the private `Prototype5/` folder next to a copy of this folder, as for the audit (`../../docs/RUNNING_THE_CAMPAIGN.md`):

```
python setup_audit_engine.py; python make_interpolated_eval_data.py   # from ../../audit/, copied beside these files
python run_engine_twin.py --manifest JOBS.txt --workers 6             # about 13 minutes per run
python analyze_engine_twin.py; python check_against_audit.py
python summarize_engine_twin.py --audit-gap <largest difference printed by check_against_audit.py>
```
