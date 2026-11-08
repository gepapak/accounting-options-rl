# Engine twin check: protocol

**Status: exploratory, not a registered study.** No agent is trained: every run re-evaluates a frozen checkpoint of the engine audit. The rules below, the job list and the code were fixed and hashed (`PROTOCOL_HASHES.txt`) before the full evaluation.

**Seen first.** Three runs were looked at before this file was written: L3 and L7, MAPPO seed 7, DK1, under a first version of the tracer (kept in `pilot_2026-11-06/`; in horizon mode it wrongly moved the reversed payoff through the mark-to-market bucket, which is corrected), and the cap-on L7 run of the same seed and zone as a smoke test. The rules for E5 and E7 were written after seeing the L3 and L7 pilots, whose TIR and SPR errors they generalize.

## Question
The paper's bracket was checked on six hand states of the engine's cash writes. Does it hold on the engine's own evaluation paths, with its trained controllers, where the floor bound? What does a twin-based audit find without the ledger, compared with the ledger-based reconciliation and with a direct calculation?

## Runs
`JOBS.txt`, written by `make_jobs.py` from the audit's run table:
- the 54 runs in which the floor bound (L1 1, L3 20, L7 14, L7 without interpolation 5, without the flat fee 4, without the percentage payoff 6, without the sweeper 4);
- 39 comparison runs in which it did not, chosen for the protocols most likely to raise a false alarm (L0 2+2, L1 8, L3 anchor 2 and feasible 6, L7 anchor 2, feasible 6 and MAPPO 6, L7 leave-one-out MAPPO 5);
- 22 L7 runs with the percent payoff's cap really on (10 MAPPO seeds and the anchor, both zones).

The last group exists because the audit's L7 had the cap off. The audit removed `--disable_mtm_return_cap` from the command line, but evaluation restores the checkpoint's saved configuration, which has `mtm_apply_price_return_cap: false`, so the payoff was the uncapped percentage move. These 22 runs measure what depends on that.

## Method
`twin_trace.py` records every write to trading cash, with its nominal amount before any floor, without changing the engine's source. Each step is then replayed in the engine's order with its floors, on the real branch (which must reproduce the engine's cash) and on the twin branch (same pre-step state, action and own cash flows, gross P&L reversed). This is a one-step replay of the cash writes, not a second run of the engine's step. Definitions are in `analyze_engine_twin.py`. Booked P&L here includes the half-spread and the battery flow, which the audit's step ledger omitted.

## Gates (for a run to count)
- **G1.** The re-evaluated sleeve return equals the audit's table within 1e-6 points (all runs except the cap-on group).
- **G2.** The replay reproduces the engine's cash within 1e-3 DKK at every step, with no cash write of unknown kind.

## Rules
Over the 93 runs of the first two groups:
- **E1, the premise on the real path.** Reported minus booked is at least -1e-3 DKK at every step of every run.
- **E2, the bracket.** The booked P&L lies inside `[-r' - 2C, r]` at every step of every run, within 1e-3 DKK, and the booked return inside the summed bracket in every run.
- **E3, ledger-free reconciliation.** `|reported - sum of TIR| > 5` points flags at least 53 of the 54 floor-bound runs and none of the 39 comparison runs (the ledger's reconciliation: 53 and 0 in the audit).
- **E4, the width.** `width > 5` points flags at least 53 of the 54 floor-bound runs. Comparison runs are counted without a rule: a flag there shows credit available on the twin branch, not credit realized.
- **E5, the point estimates.** `|TIR error| < |SPR error|` in every floor-bound run with the volume payoff (27) and `|SPR error| < |TIR error|` in every floor-bound run with the percent payoff (27).
- **E6, the direct calculation.** With the payoff formula of the protocol in force, the computed gross differs from the booked gross by at most 0.001 points in every run without a liquidation. With the other protocol generation's formula it differs by more than 5 points in at least 50 of the 54 floor-bound runs.
- **E7, the cost-shift test.** At own-cost multipliers 0, 0.25, 0.5, 1, 2, 4, 8 and 16 the exact test flags at least one step in each of the 54 floor-bound runs, and at least 90% of the informative steps of those runs on which TIR errs by more than 0.01 points.

Reported without a rule: the bracket's width against the realized credit, the size of the TIR and SPR errors, the shift test in the comparison runs and the cap-on group's reported and booked means against the anchor.

## Reading
Every outcome is reported, whichever way it goes. The engine exposes its position and its gross P&L, so no outcome here can show that a twin is *needed* on this engine: the check asks whether the bracket survives realistic accounting. Technical corrections are allowed only before outcomes are read and are recorded in the run log.
