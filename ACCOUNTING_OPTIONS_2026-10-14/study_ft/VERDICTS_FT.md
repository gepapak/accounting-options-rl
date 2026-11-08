# Study FT verdicts

Runs: 81.

| part | arm | runs | X | X low | mean leverage low | E funded | E zero | E deficit | leverage in deficit | placebo reported | placebo booked | audit error (edge): TSR, TIR, SPR, reported | audit error (deficit): TSR, TIR, SPR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | REP | 6 | 0.984 | 0.969 | 15.62 | -16.29 | -30.21 | -30.64 | 13.40 | -12.40 | -101.48 | 1.0e-13, 1.760, 5.386, 28.27 | 8.8e-14, 7.728, 11.634 |
| A | LEDGER | 6 | 0.042 | 0.000 | 3.72 | +20.76 | +20.60 | +20.60 | 3.71 | -7.50 | -9.93 | 1.2e-13, 0.001, 0.044, 0.01 | 1.0e-13, 10.719, 1.247 |
| A | SPR | 6 | 0.049 | 0.000 | 2.93 | +16.64 | +16.18 | +16.17 | 2.86 | -6.03 | -7.62 | 1.1e-13, 0.002, 0.020, 0.02 | 8.3e-14, 5.216, 0.896 |
| A | TIR | 6 | 0.097 | 0.000 | 3.10 | +17.49 | +17.50 | +17.50 | 3.11 | -6.36 | -8.19 | 1.1e-13, 0.002, 0.045, 0.02 | 8.7e-14, 12.318, 1.185 |
| A | TSR | 3 | 0.000 | 0.000 | 3.44 | +19.53 | +19.20 | +19.20 | 3.43 | -6.97 | -9.07 | 1.1e-13, 0.002, 0.035, 0.01 | 9.9e-14, 14.377, 1.056 |
| B | REP | 6 | 0.333 | 0.333 | 7.38 | +7.43 | +0.62 | -1.34 | 7.26 | -10.21 | -33.05 | 1.2e-13, 7.520, 7.975, 4.10 | 7.7e-14, 58.484, 58.887 |
| B | LEDGER | 6 | 0.000 | 0.000 | 3.69 | +20.23 | +20.40 | +20.41 | 3.69 | -7.99 | -9.76 | 1.3e-13, 0.005, 0.083, 0.00 | 1.1e-13, 30.769, 30.946 |
| B | SPR | 6 | 0.000 | 0.000 | 2.15 | +14.39 | +11.31 | +11.24 | 1.98 | -5.24 | -5.85 | 7.6e-14, 0.102, 0.116, 0.05 | 5.6e-14, 16.516, 16.516 |
| B | TIR | 6 | 0.000 | 0.000 | 2.40 | +14.82 | +12.24 | +11.79 | 2.10 | -6.30 | -7.15 | 1.0e-13, 0.085, 0.100, 0.05 | 8.4e-14, 17.720, 17.721 |
| B | TSR | 3 | 0.000 | 0.000 | 3.38 | +18.47 | +18.80 | +18.81 | 3.38 | -7.22 | -8.75 | 1.3e-13, 0.011, 0.082, 0.00 | 1.1e-13, 28.310, 28.452 |
| C | REP | 6 | 0.750 | 0.729 | 12.05 | -15.42 | -24.75 | -27.34 | 12.04 | -21.38 | -88.44 | 1.3e-13, 25.228, 26.192, 15.73 | 9.0e-14, 91.986, 93.148 |
| C | LEDGER | 6 | 0.000 | 0.000 | 3.75 | +16.32 | +20.76 | +20.76 | 3.75 | -9.33 | -11.14 | 1.2e-13, 0.028, 0.099, 0.01 | 9.7e-14, 31.256, 31.433 |
| C | SPR | 6 | 0.000 | 0.000 | 0.11 | +0.32 | +0.23 | +0.23 | 0.04 | -0.14 | -0.14 | 2.7e-14, 0.000, 0.000, 0.00 | 1.6e-14, 0.363, 0.363 |
| C | TIR | 6 | 0.000 | 0.000 | 0.04 | -0.03 | +0.05 | +0.05 | 0.01 | -0.26 | -0.26 | 5.0e-15, 0.000, 0.000, 0.00 | 3.7e-15, 0.087, 0.087 |
| C | TSR | 3 | 0.000 | 0.000 | 3.50 | +16.13 | +19.52 | +19.52 | 3.50 | -9.81 | -11.40 | 1.2e-13, 0.056, 0.113, 0.03 | 8.7e-14, 29.285, 29.427 |

Exploratory: Part B bracket, windows with every step inside: edge funded 78.8, deficit 0.0 of 100 for REP (the owner's claim is not forgiving, so failures are expected).

| ID | Result | Verdict |
|---|---|---|
| FA1 | bucket: exploit present: X REP 0.984 > LEDGER 0.042: p = 0.0015 (need < 0.05) and at least LEDGER + 0.10 | SUPPORTED |
| FA2 | bucket: removes: X SPR 0.049 < REP 0.984: p = 0.0015 (need < 0.05); at most LEDGER + 0.10 = 0.142 | SUPPORTED |
| FA3 | bucket: keeps: E SPR +16.64 / LEDGER +20.76 = 0.801 (need >= 0.80); one-sided 95% bound 0.647 (need >= 0.60) | SUPPORTED |
| FA4 | bucket: removes: X TIR 0.097 < REP 0.984: p = 0.0017 (need < 0.05); at most LEDGER + 0.10 = 0.142 | SUPPORTED |
| FA5 | bucket: keeps: E TIR +17.49 / LEDGER +20.76 = 0.843 (need >= 0.80); one-sided 95% bound 0.681 (need >= 0.60) | SUPPORTED |
| FA6 | bucket: removes: X TSR 0.000 < REP 0.984: p = 0.0071 (need < 0.05); at most LEDGER + 0.10 = 0.142 | SUPPORTED |
| FA7 | bucket: keeps: E TSR +19.53 / LEDGER +20.76 = 0.941 (need >= 0.80); one-sided 95% bound 0.775 (need >= 0.60) | SUPPORTED |
| FB1 | deficit: exploit present: X REP 0.333 > LEDGER 0.000: p = 0.0123 (need < 0.05) and at least LEDGER + 0.10 | SUPPORTED |
| FB2 | deficit: removes: X SPR 0.000 < REP 0.333: p = 0.0123 (need < 0.05); at most LEDGER + 0.10 = 0.100 | SUPPORTED |
| FB3 | deficit: removes: X TIR 0.000 < REP 0.333: p = 0.0123 (need < 0.05); at most LEDGER + 0.10 = 0.100 | SUPPORTED |
| FB4 | deficit: removes: X TSR 0.000 < REP 0.333: p = 0.0505 (need < 0.05); at most LEDGER + 0.10 = 0.100 | NOT SUPPORTED |
| FB5 | deficit, half of training funded: keeps the edge from a deficit start: Ed TSR +18.81 / LEDGER +20.41 = 0.922 (need >= 0.80); one-sided 95% bound 0.710 (need >= 0.60) | SUPPORTED |
| FB6 | deficit, half of training funded: keeps the edge from a deficit start: Ed SPR +11.24 / LEDGER +20.41 = 0.551 (need >= 0.80); one-sided 95% bound 0.401 (need >= 0.60) | NOT SUPPORTED |
| FB7 | deficit, half of training funded: keeps the edge from a deficit start: Ed TIR +11.79 / LEDGER +20.41 = 0.578 (need >= 0.80); one-sided 95% bound 0.390 (need >= 0.60) | NOT SUPPORTED |
| FD1 | deficit only: exploit present: X REP 0.750 > LEDGER 0.000: p = 0.0012 (need < 0.05) and at least LEDGER + 0.10 | SUPPORTED |
| FD2 | deficit only: removes: X SPR 0.000 < REP 0.750: p = 0.0012 (need < 0.05); at most LEDGER + 0.10 = 0.100 | SUPPORTED |
| FD3 | deficit only: removes: X TIR 0.000 < REP 0.750: p = 0.0012 (need < 0.05); at most LEDGER + 0.10 = 0.100 | SUPPORTED |
| FD4 | deficit only: removes: X TSR 0.000 < REP 0.750: p = 0.0104 (need < 0.05); at most LEDGER + 0.10 = 0.100 | SUPPORTED |
| FD5 | deficit only: the line reward keeps the edge: Ed TSR +19.52 / LEDGER +20.76 = 0.940 (need >= 0.80); one-sided 95% bound 0.774 (need >= 0.60) | SUPPORTED |
| FD6 | deficit only: a sign query cannot read the edge: Ed SPR +0.23 / LEDGER +20.76 = 0.011 (need < 0.50); one-sided 95% upper bound 0.026 (need < 0.80) | SUPPORTED |
| FD7 | deficit only: a sign query cannot read the edge: Ed TIR +0.05 / LEDGER +20.76 = 0.002 (need < 0.50); one-sided 95% upper bound 0.006 (need < 0.80) | SUPPORTED |
| FC1 | TSR's audit: largest error over all windows of all runs 1.02e-12 points (need <= 1e-6) | SUPPORTED |
| FC2 | bucket: the bracket holds at every audited step in 27 of 27 runs (need all) | SUPPORTED |
