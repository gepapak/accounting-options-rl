# Study Y verdicts

Runs: 80 of 80.

| setting | arm | runs | mean X (share of low-equity probes at leverage 16) | runs with X > 0 | mean E (booked % K0 / episode) | mean E from zero equity | placebo reported | placebo forgiven |
|---|---|---|---|---|---|---|---|---|
| S_DK1+Z | ORACLE | 8 | 0.000 | 0 | +21.07 | +21.07 | -7.50 | 2.63 |
| S_DK1+Z | PLAIN | 8 | 0.312 | 5 | +5.09 | +2.66 | -18.60 | 31.56 |
| S_DK1+Z | SPR | 8 | 0.000 | 0 | +22.01 | +22.01 | -7.93 | 2.85 |
| S_DK1+Z | TIR | 8 | 0.000 | 0 | +21.07 | +21.07 | -7.50 | 2.63 |
| X_DK1+Z | ORACLE | 8 | 0.000 | 0 | +21.07 | +21.07 | -7.49 | 2.63 |
| X_DK1+Z | PLAIN | 8 | 0.383 | 5 | +2.62 | -2.81 | -17.66 | 38.40 |
| X_DK1+Z | SPR | 8 | 0.000 | 0 | +21.07 | +21.07 | -7.49 | 2.63 |
| X_DK1+Z | TIR | 8 | 0.000 | 0 | +22.01 | +22.01 | -7.92 | 2.85 |
| S_DK1 | ORACLE | 8 | 0.000 | 0 | +20.22 | +20.22 | -7.12 | 2.38 |
| S_DK1 | PLAIN | 8 | 0.000 | 0 | +20.22 | +20.22 | -7.12 | 2.38 |

Exploratory (S_DK1+Z): agents with X >= 0.25: 5; median |TIR - booked| (edge / placebo) 0.093 / 0.036; median |reported - booked| (placebo) 51.008.

Exploratory (X_DK1+Z): agents with X >= 0.25: 5; median |TIR - booked| (edge / placebo) 0.457 / 0.132; median |reported - booked| (placebo) 51.270.

| ID | Result | Verdict |
|---|---|---|
| YA1-S | single floor, zero starts: X PLAIN 0.312 > ORACLE 0.000: one-sided Mann-Whitney p = 0.0055 (need < 0.05) | SUPPORTED |
| YA2-S | single floor, zero starts: X TIR 0.000 < PLAIN 0.312: p = 0.0055 (need < 0.05); TIR <= ORACLE + 0.10 = 0.100 | SUPPORTED |
| YA3-S | single floor, zero starts: TIR E +21.07 / ORACLE +21.07 = 1.000 (need >= 0.80); one-sided 95% bootstrap LB 0.911 (need >= 0.60) | SUPPORTED |
| YA4-S | single floor, zero starts: X SPR 0.000 < PLAIN 0.312: p = 0.0055 (need < 0.05); SPR <= ORACLE + 0.10 = 0.100 | SUPPORTED |
| YA5-S | single floor, zero starts: SPR E +22.01 / ORACLE +21.07 = 1.044 (need >= 0.80); one-sided 95% bootstrap LB 1.000 (need >= 0.60) | SUPPORTED |
| YB1-S | single floor, zero starts: ORACLE, SPR and TIR agents (24): max |TIR - booked| = 0.0000 (edge market), 0.0009 (placebo market) (need <= 1.0) | SUPPORTED |
| YA1-X | two-write floor, zero starts: X PLAIN 0.383 > ORACLE 0.000: one-sided Mann-Whitney p = 0.0062 (need < 0.05) | SUPPORTED |
| YA2-X | two-write floor, zero starts: X TIR 0.000 < PLAIN 0.383: p = 0.0062 (need < 0.05); TIR <= ORACLE + 0.10 = 0.100 | SUPPORTED |
| YA3-X | two-write floor, zero starts: TIR E +22.01 / ORACLE +21.07 = 1.044 (need >= 0.80); one-sided 95% bootstrap LB 1.000 (need >= 0.60) | SUPPORTED |
| YA4-X | two-write floor, zero starts: X SPR 0.000 < PLAIN 0.383: p = 0.0062 (need < 0.05); SPR <= ORACLE + 0.10 = 0.100 | SUPPORTED |
| YA5-X | two-write floor, zero starts: SPR E +21.07 / ORACLE +21.07 = 1.000 (need >= 0.80); one-sided 95% bootstrap LB 0.911 (need >= 0.60) | SUPPORTED |
| YB1-X | two-write floor, zero starts: ORACLE, SPR and TIR agents (24): max |TIR - booked| = 0.0013 (edge market), 0.0025 (placebo market) (need <= 1.0) | SUPPORTED |
| YC1 | single floor, no zero starts: X PLAIN 0.000 > ORACLE 0.000: one-sided Mann-Whitney p = 1.0000 (need < 0.05); PLAIN runs with X > 0: 0 of 8 | NOT SUPPORTED |
