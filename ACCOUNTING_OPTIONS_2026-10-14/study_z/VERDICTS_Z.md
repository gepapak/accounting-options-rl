# Study Z verdicts

Runs: 72 of 72.

| arm | runs | X | X both low | X one empty | hedge | park | E (booked edge, % K0) | E from zero | placebo reported | placebo booked | portfolio ruins / ep | account ruins / ep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REP | 8 | 0.672 | 0.938 | 0.406 | 0.000 | 0.297 | -18.30 | -33.81 | -12.27 | -179.60 | 22.07 | 61.10 |
| LEDGER | 8 | 0.004 | 0.000 | 0.008 | 0.000 | 0.008 | +33.82 | +33.56 | -7.84 | -15.78 | 3.13 | 8.95 |
| RUINP5 | 8 | 0.695 | 0.789 | 0.602 | 0.000 | 0.555 | -13.59 | -20.02 | -14.16 | -128.49 | 9.11 | 46.13 |
| RUINP25 | 8 | 0.297 | 0.242 | 0.352 | 0.000 | 0.336 | +11.35 | +11.26 | -13.66 | -47.44 | 1.19 | 18.02 |
| RUINA5 | 8 | 0.051 | 0.031 | 0.070 | 0.000 | 0.055 | +26.67 | +25.08 | -8.46 | -18.79 | 1.74 | 8.43 |
| SPR | 8 | 0.027 | 0.000 | 0.055 | 0.000 | 0.016 | +39.03 | +39.78 | -8.69 | -18.68 | 4.12 | 10.84 |
| TIR | 8 | 0.004 | 0.000 | 0.008 | 0.000 | 0.008 | +36.04 | +36.42 | -8.31 | -17.56 | 3.34 | 10.12 |
| P_REP | 8 | 0.984 | 0.984 | nan | 0.000 | nan | -21.53 | -42.93 | -8.80 | -179.41 | 29.91 | 29.91 |
| P_RUIN5 | 8 | 0.148 | 0.148 | nan | 0.000 | nan | +28.39 | +28.01 | -10.09 | -27.34 | 7.10 | 7.10 |

Exploratory: runs with X >= 0.25: 30; median placebo reported -12.36, booked -137.43; median |TIR - booked| 0.058, |SPR - booked| 1.913, |reported - booked| 129.863; median bracket width 255.59 points (placebo).

| ID | Result | Verdict |
|---|---|---|
| ZA1 | X REP 0.672 > LEDGER 0.004: one-sided Mann-Whitney p = 0.0003 (need < 0.05) and at least LEDGER + 0.10 | SUPPORTED |
| ZA2 | the portfolio ruin penalty does not remove it: X RUINP5 0.695 > LEDGER 0.004: one-sided Mann-Whitney p = 0.0012 (need < 0.05) and at least LEDGER + 0.10 | SUPPORTED |
| ZA3 | the portfolio ruin penalty does not remove it: X RUINP25 0.297 > LEDGER 0.004: one-sided Mann-Whitney p = 0.0003 (need < 0.05) and at least LEDGER + 0.10 | SUPPORTED |
| ZA4 | X RUINA5 0.051 < REP 0.672: p = 0.0004 (need < 0.05); RUINA5 <= LEDGER + 0.10 = 0.104 | SUPPORTED |
| ZA5 | X SPR 0.027 < REP 0.672: p = 0.0004 (need < 0.05); SPR <= LEDGER + 0.10 = 0.104 | SUPPORTED |
| ZA6 | SPR E +39.03 / LEDGER +33.82 = 1.154 (need >= 0.80); one-sided 95% bootstrap LB 1.027 (need >= 0.60) | SUPPORTED |
| ZA7 | X TIR 0.004 < REP 0.672: p = 0.0003 (need < 0.05); TIR <= LEDGER + 0.10 = 0.104 | SUPPORTED |
| ZA8 | TIR E +36.04 / LEDGER +33.82 = 1.066 (need >= 0.80); one-sided 95% bootstrap LB 0.927 (need >= 0.60) | SUPPORTED |
| ZA9 | pooled control, exploit present: X_sym P_REP 0.984 > LEDGER 0.000: one-sided Mann-Whitney p = 0.0001 (need < 0.05) and at least LEDGER + 0.10 | SUPPORTED |
| ZA10 | pooled control, the same penalty removes it: X_sym P_RUIN5 0.148 < P_REP 0.984: p = 0.0002 (need < 0.05); P_RUIN5 <= LEDGER + 0.10 = 0.100 | NOT SUPPORTED |
| ZB1 | twin bracket contains the booked P&L of every audited step and the booked return of every audited episode (both markets): 72 of 72 runs (need all) | SUPPORTED |
| ZB2 | portfolio ruins per account ruin (placebo test): RUINP5 and RUINP25 0.131 (16 runs) < REP 0.353 (8 runs): one-sided Mann-Whitney p = 0.0004 (need < 0.05) | SUPPORTED |
| ZB3 | the exploit's form: mean park share of the REP, RUINP5 and RUINP25 runs with X >= 0.25 (19 runs) 0.470 (need >= 0.50) | NOT SUPPORTED |
