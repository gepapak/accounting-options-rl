# Study RS verdicts

Runs: 80 of 80.

## RS_DK2

| RS_DK2 arm | runs | X | X long | X short | E (booked % K0 / week) | E from zero equity | share of steps at leverage 16 | placebo booked | placebo reported | placebo forgiven |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAIN | 16 | 0.826 | 0.174 | 0.651 | -100.57 | -130.91 | 0.464 | -129.16 | -8.37 | 120.79 |
| ORACLE | 16 | 0.000 | 0.000 | 0.000 | +8.52 | +8.51 | 0.000 | -8.46 | -4.73 | 3.73 |
| SPR | 16 | 0.000 | 0.000 | 0.000 | +5.85 | +6.28 | 0.000 | -12.15 | -6.05 | 6.10 |
| TIR | 16 | 0.000 | 0.000 | 0.000 | +8.49 | +8.51 | 0.000 | -8.52 | -4.86 | 3.66 |
| RUIN5 | 16 | 0.094 | 0.000 | 0.094 | -9.48 | -16.87 | 0.067 | -23.59 | -5.07 | 18.52 |

## Ledger-free audit (100 test weeks per market and run)

| setting | arm | runs | TIR error edge (max) | TIR error placebo (max) | SPR error edge (max) | SPR error placebo (max) | reported error edge (max) | reported error placebo (max) | SPR bias edge (mean) |
|---|---|---|---|---|---|---|---|---|---|
| RS_DK2 | PLAIN | 16 | 0.881 | 0.922 | 75.887 | 60.262 | 181.57 | 154.08 | +35.789 |
| RS_DK2 | ORACLE | 16 | 0.001 | 0.001 | 1.239 | 2.890 | 2.88 | 6.35 | +0.214 |
| RS_DK2 | SPR | 16 | 0.002 | 0.002 | 5.655 | 6.119 | 12.60 | 13.34 | +1.241 |
| RS_DK2 | TIR | 16 | 0.002 | 0.001 | 0.943 | 2.641 | 2.57 | 6.29 | +0.118 |
| RS_DK2 | RUIN5 | 16 | 0.791 | 0.475 | 18.555 | 29.029 | 54.55 | 79.19 | +4.527 |

| ID | Result | Verdict |
|---|---|---|
| RS1 | X PLAIN 0.826 > ORACLE 0.000: one-sided Mann-Whitney p = 0.0000 (need < 0.05) | SUPPORTED |
| RS2 | X SPR 0.000 < PLAIN 0.826: p = 0.0000 (need < 0.05); SPR <= ORACLE + 0.10 = 0.100 | SUPPORTED |
| RS3 | SPR E +5.85 / ORACLE +8.52 = 0.687 (need >= 0.80); one-sided 95% bootstrap LB 0.370 (need >= 0.60) | NOT SUPPORTED |
| RS4 | X TIR 0.000 < PLAIN 0.826: p = 0.0000 (need < 0.05); TIR <= ORACLE + 0.10 = 0.100 | SUPPORTED |
| RS5 | TIR E +8.49 / ORACLE +8.52 = 0.997 (need >= 0.80); one-sided 95% bootstrap LB 0.755 (need >= 0.60) | SUPPORTED |
| RS6 | RUIN5 X 0.094 (need <= 0.100), E -9.48 (need >= 0.80 x ORACLE +8.52 = +6.81); ORACLE edge bootstrap LB +6.60 | NOT SUPPORTED |
| RS-D1 | largest TIR audit error 0.002 points over the 48 ORACLE, SPR and TIR runs and both markets (need <= 1.0); all 80 runs: 0.922 (descriptive) | SUPPORTED |
| RS-D2 | 10752000 of 10752000 audited steps inside the step bracket; TIR inexact on 0 of 6106409 identified steps (need: all inside, none inexact) | SUPPORTED |

Holm-adjusted Mann-Whitney p-values (sensitivity, not a decision rule): RS1 0.0000, RS2 0.0000, RS4 0.0000
