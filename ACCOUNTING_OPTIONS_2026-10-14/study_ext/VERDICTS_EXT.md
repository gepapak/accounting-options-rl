# Study EXT verdicts

Runs: 70 of 62.

| arm | runs | X | mean position at probes | bust share | reported | booked | mean width | audit error: reading, TIR, SPR, reported |
|---|---|---|---|---|---|---|---|---|
| REP | 8 | 0.888 | 9.19 | 0.606 | +50.35 | -9.80 | 116.05 | 1.36e-13, 3.36e-13, 45.33, 60.16 |
| LEDGER | 8 | 0.031 | 1.82 | 0.016 | +1.13 | +0.13 | 2.62 | 4.55e-14, 6.34e-14, 1.31, 1.00 |
| SPR | 8 | 0.003 | 1.66 | 0.009 | +2.73 | +2.18 | 1.55 | 3.19e-14, 5.74e-14, 0.78, 0.54 |
| TIR | 8 | 0.000 | 1.72 | 0.005 | +0.18 | -0.14 | 1.30 | 3.53e-14, 5.06e-14, 0.65, 0.33 |
| TSR | 4 | 0.044 | 2.19 | 0.020 | +1.42 | +0.35 | 3.54 | 5.48e-14, 8.00e-14, 1.77, 1.07 |
| RUIN5 | 8 | 0.924 | 9.41 | 0.560 | +67.03 | +5.02 | 138.85 | 1.62e-13, 3.76e-13, 53.61, 62.01 |
| RUIN25 | 8 | 0.789 | 8.13 | 0.516 | +49.21 | +1.35 | 110.14 | 1.73e-13, 3.41e-13, 45.08, 47.86 |
| RUIN100 | 8 | 0.183 | 2.98 | 0.121 | +14.80 | +2.64 | 30.02 | 8.92e-14, 1.43e-13, 12.88, 12.16 |
| FINDER | 8 | 0.993 | 9.94 | 0.698 | +69.86 | -27.12 | 195.27 | 2.97e-13, 5.46e-13, 75.33, 96.98 |
| FINDER_DEF | 2 | 0.000 | 0.52 | 0.000 | +1.47 | +1.47 | 0.00 | 8.31e-15, 2.00e-14, 0.00, 0.00 |

Exploratory: median distance to the jump, multiples of the realized move, by position (leverage set): {'1': 49.114428857715126, '2': 25.821847431204937, '5': 10.84743236719283, '10': 5.455670982787149}; share within 16: {'1': 0.1100811123986095, '2': 0.329004329004329, '5': 0.615909090909091, '10': 0.8257918552036199}.

| ID | Result | Verdict |
|---|---|---|
| EA1 | X REP 0.888 > LEDGER 0.031: one-sided Mann-Whitney p = 0.0003 (need < 0.05) and at least LEDGER + 0.10 | SUPPORTED |
| EA2 | REP mean reported +50.35 minus mean booked -9.80 = 60.16 points (need >= 10) | SUPPORTED |
| EA3 | bust share REP 0.606 > LEDGER 0.016: p = 0.0004 (need < 0.05), difference at least 0.10 | SUPPORTED |
| EA4 | X SPR 0.003 < REP 0.888: p = 0.0003 (need < 0.05); at most LEDGER + 0.10 = 0.131 | SUPPORTED |
| EA5 | X TIR 0.000 < REP 0.888: p = 0.0002 (need < 0.05); at most LEDGER + 0.10 = 0.131 | SUPPORTED |
| EA6 | X TSR 0.044 < REP 0.888: p = 0.0040 (need < 0.05); at most LEDGER + 0.10 = 0.131 | SUPPORTED |
| EA7 | the penalty does not remove it: X RUIN5 0.924 > LEDGER 0.031: one-sided Mann-Whitney p = 0.0003 (need < 0.05) and at least LEDGER + 0.10 | SUPPORTED |
| EA8 | the penalty does not remove it: X RUIN25 0.789 > LEDGER 0.031: one-sided Mann-Whitney p = 0.0003 (need < 0.05) and at least LEDGER + 0.10 | SUPPORTED |
| EA9 | X RUIN100 0.183 < REP 0.888: p = 0.0004 (need < 0.05); at most LEDGER + 0.10 = 0.131 | NOT SUPPORTED |
| EA10 | FINDER X at least 0.5 in 8 of 8 runs (need >= 7); mean X 0.993 | SUPPORTED |
| EA11 | default position set: largest width over the test windows 1.02e-13 points in 2 runs (need <= 1e-6) | SUPPORTED |
| EA12 | bracket holds at every audited step in 70 of 70 runs (need all) | SUPPORTED |
| EA13 | largest error of the tail reading's audit in any window of any run 2.09e-11 points (need <= 0.01) | SUPPORTED |
| EA14 | REP runs: audit error SPR 45.33 points (need >= 5), TIR 3.36e-13 (need <= 0.01) | SUPPORTED |
| ES1 | tail reading exact on 1.0000 of the default set and 1.0000 of the leverage set (need >= 0.999 each) | SUPPORTED |
| ES2 | leverage set: 3505 jumps, 1.0000 at the true location, 1.0000 of a size equal to the current wealth within 5% (need >= 0.99 each) | SUPPORTED |
| ES3 | default set: width non-zero on 0 of 4000 states at t = +1, -1 (need 0) | SUPPORTED |
