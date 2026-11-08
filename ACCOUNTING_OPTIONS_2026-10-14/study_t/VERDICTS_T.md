# Study T verdicts

Runs: Part A 90 of 90, Part B 80 of 80.

## Part A: Study S's edge-market test on a new market draw (DK1, PPO), with the occupancy-measure baseline

| S_DK1 arm | runs | mean X (share of low-equity probes at leverage 16) | mean E (booked % K0 / episode) | placebo reported | placebo forgiven |
|---|---|---|---|---|---|
| PLAIN | 10 | 0.600 | +1.98 | -8.68 | 47.29 |
| ORACLE | 10 | 0.025 | +20.46 | -7.18 | 3.89 |
| SPR | 10 | 0.000 | +19.40 | -6.77 | 2.24 |
| TIR | 10 | 0.025 | +21.34 | -7.78 | 3.17 |
| OM1 | 10 | 0.000 | +7.70 | -0.49 | 0.00 |
| OM0.1 | 10 | 0.000 | +7.70 | -0.49 | 0.00 |
| OM0.05 | 10 | 0.000 | +7.70 | -0.49 | 0.00 |
| OM0.02 | 10 | 0.287 | +0.94 | -3.23 | 22.85 |
| OM0.01 | 10 | 0.537 | -0.07 | -5.65 | 39.81 |

## Part B: T_DK2

| T_DK2 arm | runs | mean X (share of low-equity probes at leverage 16) | mean E (booked % K0 / episode) | placebo reported | placebo forgiven |
|---|---|---|---|---|---|
| PLAIN | 10 | 0.394 | +21.85 | -8.12 | 31.99 |
| ORACLE | 10 | 0.000 | +32.66 | -10.80 | 2.52 |
| SPR | 10 | 0.000 | +33.34 | -11.58 | 2.52 |
| TIR | 10 | 0.013 | +33.97 | -11.48 | 2.87 |

## Part B: T_DK1M

| T_DK1M arm | runs | mean X (share of low-equity probes at leverage 16) | mean E (booked % K0 / episode) | placebo reported | placebo forgiven |
|---|---|---|---|---|---|
| PLAIN | 10 | 0.388 | +6.72 | -5.43 | 21.32 |
| ORACLE | 10 | 0.000 | +11.63 | -5.94 | 1.66 |
| SPR | 10 | 0.000 | +13.45 | -6.92 | 2.05 |
| TIR | 10 | 0.000 | +13.25 | -6.68 | 2.16 |

| ID | Result | Verdict |
|---|---|---|
| TA1 | X PLAIN 0.600 > ORACLE 0.025: one-sided Mann-Whitney p = 0.0000 (need < 0.05) | SUPPORTED |
| TA2 | X SPR 0.000 < PLAIN 0.600: p = 0.0000 (need < 0.05); SPR <= ORACLE + 0.10 = 0.125 | SUPPORTED |
| TA3 | SPR E +19.40 / ORACLE +20.46 = 0.948 (need >= 0.80); one-sided 95% bootstrap LB 0.852 (need >= 0.60) | SUPPORTED |
| TA4 | X TIR 0.025 < PLAIN 0.600: p = 0.0000 (need < 0.05); TIR <= ORACLE + 0.10 = 0.125 | SUPPORTED |
| TA5 | TIR E +21.34 / ORACLE +20.46 = 1.043 (need >= 0.90); one-sided 95% bootstrap LB 0.983 (need >= 0.75) | SUPPORTED |
| TA6 | OM1: X 0.000, E +7.70 -> not both; OM0.1: X 0.000, E +7.70 -> not both; OM0.05: X 0.000, E +7.70 -> not both; OM0.02: X 0.287, E +0.94 -> not both; OM0.01: X 0.537, E -0.07 -> not both (both = X <= 0.125 and E >= +16.37) | SUPPORTED |
| TB-T_DK2-1 | X PLAIN 0.394 > ORACLE 0.000: one-sided Mann-Whitney p = 0.0029 (need < 0.05) | SUPPORTED |
| TB-T_DK2-2 | X SPR 0.000 < PLAIN 0.394: p = 0.0029 (need < 0.05); SPR <= ORACLE + 0.10 = 0.100 | SUPPORTED |
| TB-T_DK2-3 | SPR E +33.34 / ORACLE +32.66 = 1.021 (need >= 0.80); one-sided 95% bootstrap LB 0.930 (need >= 0.60) | SUPPORTED |
| TB-T_DK2-4 | X TIR 0.013 < PLAIN 0.394: p = 0.0072 (need < 0.05); TIR <= ORACLE + 0.10 = 0.100 | SUPPORTED |
| TB-T_DK2-5 | TIR E +33.97 / ORACLE +32.66 = 1.040 (need >= 0.90); one-sided 95% bootstrap LB 0.967 (need >= 0.75) | SUPPORTED |
| TB-T_DK1M-1 | X PLAIN 0.388 > ORACLE 0.000: one-sided Mann-Whitney p = 0.0004 (need < 0.05) | SUPPORTED |
| TB-T_DK1M-2 | X SPR 0.000 < PLAIN 0.388: p = 0.0004 (need < 0.05); SPR <= ORACLE + 0.10 = 0.100 | SUPPORTED |
| TB-T_DK1M-3 | SPR E +13.45 / ORACLE +11.63 = 1.156 (need >= 0.80); one-sided 95% bootstrap LB 1.000 (need >= 0.60) | SUPPORTED |
| TB-T_DK1M-4 | X TIR 0.000 < PLAIN 0.388: p = 0.0004 (need < 0.05); TIR <= ORACLE + 0.10 = 0.100 | SUPPORTED |
| TB-T_DK1M-5 | TIR E +13.25 / ORACLE +11.63 = 1.139 (need >= 0.90); one-sided 95% bootstrap LB 0.986 (need >= 0.75) | SUPPORTED |
