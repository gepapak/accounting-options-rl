# Study X verdicts

Runs: 32 of 32.

| arm | runs | mean X (share of low-equity probes at leverage 16) | runs with X > 0 | mean E (booked % K0 / episode) | mean E from zero equity | placebo reported | placebo forgiven |
|---|---|---|---|---|---|---|---|
| PLAIN | 8 | 0.836 | 8 | -13.37 | -25.83 | -14.71 | 81.85 |
| ORACLE | 8 | 0.000 | 0 | +18.46 | +18.56 | -6.62 | 1.91 |
| SPR | 8 | 0.000 | 0 | +20.66 | +21.37 | -7.33 | 2.74 |
| TIR | 8 | 0.000 | 0 | +18.19 | +18.50 | -6.41 | 2.10 |

Exploratory: agents with X >= 0.25: 8; median |TIR - booked| (placebo) 0.054, median |reported - booked| (placebo) 93.747.

| ID | Result | Verdict |
|---|---|---|
| XA1 | X PLAIN 0.836 > ORACLE 0.000: one-sided Mann-Whitney p = 0.0002 (need < 0.05) | SUPPORTED |
| XA2 | X TIR 0.000 < PLAIN 0.836: p = 0.0002 (need < 0.05); TIR <= ORACLE + 0.10 = 0.100 | SUPPORTED |
| XA3 | TIR E +18.19 / ORACLE +18.46 = 0.986 (need >= 0.80); one-sided 95% bootstrap LB 0.817 (need >= 0.60) | SUPPORTED |
| XA4 | X SPR 0.000 < PLAIN 0.836: p = 0.0002 (need < 0.05); SPR <= ORACLE + 0.10 = 0.100 | SUPPORTED |
| XA5 | SPR E +20.66 / ORACLE +18.46 = 1.119 (need >= 0.80); one-sided 95% bootstrap LB 0.968 (need >= 0.60) | SUPPORTED |
| XB1 | ORACLE, SPR and TIR agents (24): max |TIR - booked| = 0.001 (edge market), 0.006 (placebo market) (need <= 1.0) | SUPPORTED |
