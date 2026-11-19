# Engine twin check: decisions

Exploratory. No agent trained; frozen checkpoints of the engine audit. Rules: PROTOCOL.md.

Runs: 93 (54 floor bound, 39 comparison), plus 22 with the percent payoff's cap on.

| Rule | Result | Decision |
|---|---|---|
| G1 re-evaluation | largest difference from the audit's table 1.137e-13 pp | met |
| G2 replay | largest parity error 0 DKK, 0 unknown writes | met |
| E1 forgiving on the real path | smallest credit -3.3e-06 DKK | supported |
| E2 bracket contains booked P&L | 0 steps outside; 93 of 93 runs contained | supported |
| E3 TIR gap flags | 53 of 54 bound, 0 of 39 comparison (ledger: 53, 0) | supported |
| E4 width flags | 53 of 54 bound (1 of 39 comparison, no rule) | supported |
| E5 estimators split by payoff | TIR closer in 24 of 27 volume-payoff runs, SPR closer in 27 of 27 percent-payoff runs | NOT SUPPORTED |
| E6 direct calculation | right formula: largest error 9.97e-06 pp over 86 runs without a liquidation; other generation's formula over 5 pp in 53 of 54 | supported |
| E7 cost-shift test | flags 53 of 54 bound runs and 57120 of 57173 steps where TIR errs by over 0.01 pp (0.999) | NOT SUPPORTED |
