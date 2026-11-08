# Study V verdicts

Runs: Part A 72 of 72, Part B 16 of 16, Part C 56 of 56.

## Part A: the Study T edge-market test (DK1, PPO) with rewards that need no clone

| S_DK1 arm | runs | mean X (share of low-equity probes at leverage 16) | runs with X > 0 | mean E (booked % K0 / episode) | mean E from zero equity | placebo reported | placebo forgiven |
|---|---|---|---|---|---|---|---|
| PLAIN | 8 | 0.781 | 7 | -1.79 | -17.97 | -7.10 | 64.12 |
| ORACLE | 8 | 0.000 | 0 | +20.89 | +20.65 | -7.59 | 2.58 |
| PMR | 8 | 0.062 | 1 | +18.00 | +17.96 | -11.29 | 8.84 |
| RUIN1 | 8 | 0.336 | 4 | +11.74 | +4.51 | -5.88 | 23.14 |
| RUIN5 | 8 | 0.000 | 0 | +18.97 | +18.66 | -6.69 | 2.26 |
| RUIN25 | 8 | 0.000 | 0 | +17.63 | +14.95 | -6.18 | 1.50 |
| MV0.05 | 8 | 0.000 | 0 | +8.59 | +10.34 | -1.37 | 0.58 |
| MV0.2 | 8 | 0.000 | 0 | +4.48 | +4.27 | -0.24 | 0.00 |
| MV1 | 8 | 0.031 | 1 | +1.06 | +0.45 | -0.30 | 0.35 |

## Part B: T_DK2 (PMR against Study T's PLAIN and ORACLE runs in the same setting)

| T_DK2 arm | runs | mean X | runs with X > 0 | mean E |
|---|---|---|---|---|
| PLAIN (Study T) | 10 | 0.394 | 6 | +21.85 |
| ORACLE (Study T) | 10 | 0.000 | 0 | +32.66 |
| PMR | 8 | 0.000 | 0 | +35.12 |

## Part B: T_DK1M (PMR against Study T's PLAIN and ORACLE runs in the same setting)

| T_DK1M arm | runs | mean X | runs with X > 0 | mean E |
|---|---|---|---|---|
| PLAIN (Study T) | 10 | 0.388 | 8 | +6.72 |
| ORACLE (Study T) | 10 | 0.000 | 0 | +11.63 |
| PMR | 8 | 0.000 | 0 | +15.07 |

## Part C: the same test when half of the training episodes start at zero equity

| S_DK1+Z arm | runs | mean X (share of low-equity probes at leverage 16) | runs with X > 0 | mean E (booked % K0 / episode) | mean E from zero equity | placebo reported | placebo forgiven |
|---|---|---|---|---|---|---|---|
| PLAIN | 8 | 0.914 | 8 | -17.59 | -31.16 | -15.87 | 88.84 |
| ORACLE | 8 | 0.000 | 0 | +19.84 | +20.22 | -6.82 | 2.38 |
| TIR | 8 | 0.000 | 0 | +20.79 | +21.34 | -7.47 | 2.68 |
| PMR | 8 | 0.000 | 0 | +22.00 | +22.01 | -7.93 | 2.85 |
| RUIN1 | 8 | 0.641 | 8 | -2.35 | -15.54 | -9.73 | 57.43 |
| RUIN5 | 8 | 0.000 | 0 | +17.19 | +15.76 | -6.48 | 1.85 |
| RUIN25 | 8 | 0.000 | 0 | +4.18 | +0.03 | -2.52 | 0.00 |

## Part D: ledger-free audit of the trained agents (twin queries at evaluation time)

| estimate | max error, edge market | max error, placebo market | median error, placebo market, agents with X >= 0.25 |
|---|---|---|---|
| reported | 33.346 | 104.883 | 68.254 |
| spr | 5.772 | 2.171 | 1.146 |
| tir | 0.385 | 0.129 | 0.080 |
| pmr | 10.189 | 2.843 | 0.821 |

Agents audited: 128; agents with X >= 0.25: 28. Errors are |estimated - booked| of the 100-episode mean return, percent of K0 per episode.

| ID | Result | Verdict |
|---|---|---|
| VA1 | X PLAIN 0.781 > ORACLE 0.000: one-sided Mann-Whitney p = 0.0006 (need < 0.05) | SUPPORTED |
| VA2 | X PMR 0.062 < PLAIN 0.781: p = 0.0012 (need < 0.05); PMR <= ORACLE + 0.10 = 0.100 | SUPPORTED |
| VA3 | PMR E +18.00 / ORACLE +20.89 = 0.861 (need >= 0.80); one-sided 95% bootstrap LB 0.478 (need >= 0.60) | NOT SUPPORTED |
| VA4 | MV0.05: X 0.000, E +8.59 -> not both; MV0.2: X 0.000, E +4.48 -> not both; MV1: X 0.031, E +1.06 -> not both (both = X <= 0.100 and E >= +16.72) | SUPPORTED |
| VA5 | RUIN1: X 0.336, E +11.74 -> not both; RUIN5: X 0.000, E +18.97 -> both; RUIN25: X 0.000, E +17.63 -> both (both = X <= 0.100 and E >= +16.72) | NOT SUPPORTED |
| VB-T_DK2-1 | X PMR 0.000 < PLAIN 0.394: p = 0.0063 (need < 0.05); PMR <= ORACLE + 0.10 = 0.100 | SUPPORTED |
| VB-T_DK2-2 | PMR E +35.12 / ORACLE +32.66 = 1.075 (need >= 0.80); one-sided 95% bootstrap LB 1.023 (need >= 0.60) | SUPPORTED |
| VB-T_DK1M-1 | X PMR 0.000 < PLAIN 0.388: p = 0.0010 (need < 0.05); PMR <= ORACLE + 0.10 = 0.100 | SUPPORTED |
| VB-T_DK1M-2 | PMR E +15.07 / ORACLE +11.63 = 1.295 (need >= 0.80); one-sided 95% bootstrap LB 1.148 (need >= 0.60) | SUPPORTED |
| VC1 | X PLAIN 0.914 > ORACLE 0.000: one-sided Mann-Whitney p = 0.0001 (need < 0.05) | SUPPORTED |
| VC2 | X PMR 0.000 < PLAIN 0.914: p = 0.0001 (need < 0.05); PMR <= ORACLE + 0.10 = 0.100 | SUPPORTED |
| VC3 | PMR E +22.00 / ORACLE +19.84 = 1.109 (need >= 0.80); one-sided 95% bootstrap LB 1.022 (need >= 0.60) | SUPPORTED |
| VC4 | X TIR 0.000 < PLAIN 0.914: p = 0.0001 (need < 0.05); TIR <= ORACLE + 0.10 = 0.100 | SUPPORTED |
| VC5 | TIR E +20.79 / ORACLE +19.84 = 1.048 (need >= 0.80); one-sided 95% bootstrap LB 0.943 (need >= 0.60) | SUPPORTED |
| VC6 | RUIN1: X 0.641, E -2.35 -> not both; RUIN5: X 0.000, E +17.19 -> both; RUIN25: X 0.000, E +4.18 -> not both (both = X <= 0.100 and E >= +15.87) | NOT SUPPORTED |
| VD1 | max |TIR - booked| = 0.385 (edge market), 0.129 (placebo market) over 128 agents (need <= 1.0) | SUPPORTED |
| VD2 | 28 agents with X >= 0.25: median reported-return error 68.25 (need >= 5), median TIR/reported error ratio 0.0011 (need <= 0.05), placebo market | SUPPORTED |
