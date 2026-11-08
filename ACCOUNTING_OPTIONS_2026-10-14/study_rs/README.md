# Study RS: completed real-sign reward stress test

**Status: all 80 registered runs completed, and all eight registered decisions
were reproduced by a separate recount. Six were supported; two were not.**

The run started at 23:29 on 4 November 2026 and finished at 02:22 on
5 November (local time). The registered aggregator completed at 02:23, exit
code 0. There were no failed or skipped jobs, and the error log is empty.
No training process remains running.

It is reported in the manuscript (main-text paragraph "Real signs" and its own supplementary section).

**Location.** The study was designed and run in `Paper1/study_rs_2026-11-04`, next to the `github` folder, and this folder is a byte-exact copy made on 5 November 2026 (140 files). Only this README, `portable_rs.py`, `make_studyRS_outputs.py` and `fig_studyRS.png` were changed or added; none is a registered file. The registered files keep the paths of the original layout, so:

- run the scripts through `portable_rs.py` (see below), which finds the reused code in this repository;
- `tools/verify_studies.py`, at the repository root, checks the registered hashes from here, including the disclosed defect below;
- `hash_prereg_rs.py --verify` and `verify_completed_rs.py` read the original layout and do not run from here. Their recorded outputs are in `independent_verification/`.

## What was tested

The gap was that the reward interventions in Studies S-Z had been tested with
simulated market signs. Study RS retains the deliberately loss-forgiving,
zero-floor sleeve but trains on historical DK2 spread signs and magnitudes,
using the sign eight quarters earlier as the observation signal.

There are five PPO arms, each with 16 seeds (300-315), nominally one million
training steps per run. Half of the training episodes start at zero equity.
Evaluation uses the same 100 sampled test weeks and funded start equities for
each agent. The complete design, pilot-dependent choices and decision rules
are in [the registration](PREREGISTRATION_RS.md).

This is a **real-sign stress test of reward correction**, not a calibrated
exchange simulation or an evaluation of live trading profitability.

## Results

`X` is the fraction of 24 specified low-equity probes choosing maximum
absolute leverage. `E` is booked P&L per simulated week, as a percentage of a
fixed EUR 20,000 allocation. It is not a feasible investment return: the
simulator permits continued trading without sufficient capital, while its
reference ledger records the resulting losses.

| Training reward | Seeds | X | E (% fixed allocation/week) |
|---|---:|---:|---:|
| PLAIN: reported reward | 16 | 0.826 | -100.57 |
| ORACLE: booked reward | 16 | 0.000 | +8.52 |
| SPR | 16 | 0.000 | +5.85 |
| TIR | 16 | 0.000 | +8.49 |
| RUIN5 | 16 | 0.094 | -9.48 |

- **TIR removed the measured gambling and retained 99.7% of ORACLE's mean
  booked edge**, with a one-sided 95% bootstrap lower bound of 75.5%.
  Both registered retention thresholds (80% point estimate, 60% lower bound)
  were met.
- **SPR removed the measured gambling but failed edge retention**: 68.7%
  point estimate, 37.0% lower bound.
- **RUIN5 met the registered X tolerance but failed the joint criterion**:
  its mean booked P&L was negative. It was not completely ineffective.
- The ORACLE edge was testable: mean +8.5164, bootstrap lower bound +6.6039.
- Descriptively, all 16 TIR and ORACLE seeds had positive booked test P&L;
  all 16 PLAIN seeds had negative booked test P&L. SPR had 13 positive seeds
  and RUIN5 had eight. These are not additional registered decisions.

| ID | Registered question | Registered verdict | Recount verdict |
|---|---|---|---|
| RS1 | Exploit acquired under real signs | SUPPORTED | SUPPORTED |
| RS2 | SPR removes the exploit | SUPPORTED | SUPPORTED |
| RS3 | SPR keeps the edge | NOT SUPPORTED | NOT SUPPORTED |
| RS4 | TIR removes the exploit | SUPPORTED | SUPPORTED |
| RS5 | TIR keeps the edge | SUPPORTED | SUPPORTED |
| RS6 | RUIN5 does both | NOT SUPPORTED | NOT SUPPORTED |
| RS-D1 | TIR audit error within the registered tolerance | SUPPORTED | SUPPORTED |
| RS-D2 | Step-bracket and identified-step checks pass | SUPPORTED | SUPPORTED |

The three one-sided Mann-Whitney p-values are each approximately
1.35546e-7; the registered Holm sensitivity values are each 4.06638e-7.
Exploratory threshold sensitivity: TIR also meets Study T's stricter
90%/75% retention thresholds (actual ratio 0.997360, lower bound 0.754601).
This does not replace the registered 80%/60% rule.

### Ledger-free audit

- Recorded counters report 10,752,000 of 10,752,000 steps inside the bracket.
- TIR was recorded as inexact on zero of 6,106,409 identified steps.
- The maximum absolute error of TIR's estimated mean return across the
  48 ORACLE/SPR/TIR runs and both markets was **0.002372 percentage points**
  of the fixed allocation, versus the registered 1.0-point tolerance.
- Across all 80 runs, the largest such error was approximately 0.922 points
  (descriptive, not the RS-D1 decision set).

A separate recount (the folder name `independent_verification` is historical: the script was written with AI assistance after the registered summary existed, and was neither blinded nor by a human) recomputes episode means, error measures, all 16,000
episode brackets, probe measures, bootstrap decisions and CSV consistency
from the saved arrays. It sums and validates the step counters, but **does
not independently replay the trained policies step by step**: the runner
did not save policy weights or individual step traces.

## Registration integrity: a disclosed defect

A later inspection found a bug in the original log-prefix hasher:
`data.find(MARK)` matched the quoted marker in the log's introductory prose,
not its later standalone section heading.

- The registered log digest covers only its first **147 bytes**.
- The intended prefix through the standalone marker is **3,555 bytes**.
- Its intended-prefix SHA-256 is
  `c265e796477c5209b82e3c55d4f917d0abd6ac153ca7e14438856997fd44a72e`.
- The other **16 registered files**, including the protocol, training and
  analysis code, reused sources and market data, match their registered hashes.
- The run-log chronology was therefore **not fully protected by its recorded
  digest**. A match under the original `--verify` script is not sufficient.

The historical manifest, hasher, protocol and runner were not rewritten.
The [integrity snapshot](independent_verification/post_launch_integrity.json)
records this defect explicitly. It was taken after completion and inspection
of the registered summary; it is an audit checkpoint, not retroactive
preregistration or a pre-outcome commitment of the recount script.
Future registrations should match a unique complete marker line; that
behavior is regression-tested in the recount script.

## What this adds to the paper

This provides empirical support for TIR beyond the simulated-sign tests and
shows a useful separation between the two twin rewards: on this panel, both
prevent the measured gambling, but only TIR meets the edge-retention rule.
The evidence is consistent with a limitation of SPR under asymmetric signs;
the experiment does not by itself prove that this is the sole cause of the
SPR performance difference, and the head-to-head difference is not statistically
significant (`../../exploratory_claude_2026-11-05/rs_reanalysis.py`).

The claim made in the paper is:

> In a further internally registered stress test using historical DK2 spread
> signs and a lagged signal (16 seeds per arm), TIR removed maximum-leverage
> gambling at the specified low-equity probes and retained 99.7% of the
> ledger-trained mean booked edge (one-sided 95% lower bound 75.5%).
> SPR removed the same behavior but failed the registered edge-retention
> criterion. A 5% ruin penalty also failed the joint criterion.

Do not replace the existing limitations with an unqualified claim of
real-world validation:

1. The signal lag, zero-equity starts and retained market were selected
   following disclosed development work. DK1 was dropped after pilots, and
   test-panel statistics and a myopic test-panel evaluation informed parts
   of the design. New training seeds do not make this an untouched market
   holdout.
2. The bootstrap measures variation across training seeds, conditional on a
   single reused test panel. The sampled weeks may overlap; it is not an
   uncertainty estimate for future markets.
3. The zero floor, unrestricted empty-account trading, additive sign-reversed
   payoff and branch-invariant costs remain constructed. The protocol's
   phrase "no part ... holds by construction" is too broad: historical sign
   fairness is no longer imposed, but the replay/payoff structure still is.
4. A signal taken from a distant window is a mismatched-signal control, not
   proof of conditional independence or a mathematically no-edge market.
5. The separate margin-realism model was **not** used here. Initial margin,
   liquidation latency, execution costs and a finite insurance fund have
   not been jointly validated with these learned policies.
6. The registration was internal, not publicly timestamped, and has the
   run-log hashing defect described above.

The result strengthens the case for **TIR as a simulator-auditing and reward
correction method**. It does not establish universal superiority, market
profitability, or freedom from journal rejection risk.

## Reproduction and evidence

- [Registered decisions and numerical summary](VERDICTS_RS.md)
- [Registered per-run table](runs_rs.csv)
- [Recomputed decisions, exact statistics and result hashes](independent_verification/recount.json)
- [Recomputed per-run table](independent_verification/runs_recomputed.csv)
- [Repository isolation check](independent_verification/repository_isolation.json) (its baseline file is not shipped)
- [Run log](RUN_LOG_RS.md) and [progress log](run_full.log)
- [Recount script](verify_completed_rs.py) and [its tests](test_verify_completed_rs.py)

From this directory, validation commands that do not retrain:

```powershell
python -B portable_rs.py test_envs_rs.py
python -B portable_rs.py aggregate_rs.py --print
python -B portable_rs.py -m unittest -v test_verify_completed_rs
```

Observed validation: **309,221 registered environment checks** (including
40,320 random-policy twin steps), **15 recount-script tests**, and
synthetic agreement with the registered analysis on five decision scenarios.

The one-time recount, run in the original layout, was:

```powershell
python -B verify_completed_rs.py --snapshot
python -B verify_completed_rs.py
```

Its outputs already exist and are intentionally not overwritten on another
invocation. The recount imports neither the registered aggregator nor the
training environment. The original automatic aggregation was not rerun.
