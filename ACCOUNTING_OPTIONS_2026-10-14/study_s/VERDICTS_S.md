# Study S verdicts

Runs: Part A 60 of 60, Part B 120 of 120.

## Part A: edge market (PPO, DK1, seeds 100-109)

| Arm | runs | mean X_A (share of low-equity probes at leverage 16) | mean E (booked % K0 / episode) | placebo reported | placebo forgiven |
|---|---|---|---|---|---|
| PLAIN | 10 | 0.850 | -6.52 | -14.09 | 88.15 |
| ORACLE | 10 | 0.056 | +18.05 | -6.30 | 9.41 |
| SPR | 10 | 0.000 | +22.58 | -6.70 | 3.98 |
| KL0.003 | 10 | 0.731 | -1.95 | -12.02 | 78.06 |
| KL0.03 | 10 | 0.169 | +7.77 | +0.15 | 4.57 |
| KL0.3 | 10 | 0.000 | +8.36 | -0.68 | 0.05 |

## Part B: registered fair-market environment (Study O), seeds 110-119, DK1 and DK2

| Learner | Arm | runs | runs at 16 at equity 0.02 | mean X_B = L(0.02) - L(2) |
|---|---|---|---|---|
| PPO | ZF | 20 | 20 | +12.15 |
| PPO | SPR | 20 | 7 | +2.73 |
| PPO | LEDGER | 20 | 7 | +1.39 |
| A2C-n64 | ZF | 20 | 18 | +7.76 |
| A2C-n64 | SPR | 20 | 5 | +0.72 |
| A2C-n64 | LEDGER | 20 | 5 | +3.00 |

| ID | Result | Verdict |
|---|---|---|
| SA1 | X_A PLAIN 0.850 > ORACLE 0.056: one-sided Mann-Whitney p = 0.0001 (need < 0.05) | SUPPORTED |
| SA2 | X_A SPR 0.000 < PLAIN: p = 0.0000 (need < 0.05); SPR <= ORACLE + 0.10 = 0.156 | SUPPORTED |
| SA3 | E SPR +22.58 / ORACLE +18.05 = 1.251 (need >= 0.80); one-sided 95% bootstrap LB 0.917 (need >= 0.60) | SUPPORTED |
| SA4 | KL0.003: X_A 0.731, E -1.95 -> not both; KL0.03: X_A 0.169, E +7.77 -> not both; KL0.3: X_A 0.000, E +8.36 -> not both (both = X_A <= 0.156 and E >= +14.44) | SUPPORTED |
| SB1 PPO | paired ZF - SPR (n = 20): mean +9.43 (need >= 4), one-sided Wilcoxon p = 0.0005 (need < 0.05) | SUPPORTED |
| SB2 PPO | paired SPR - LEDGER (n = 20): mean +1.34, one-sided 95% bootstrap UB +3.51 (need < 5) | SUPPORTED |
| SB1 A2C-n64 | paired ZF - SPR (n = 20): mean +7.04 (need >= 4), one-sided Wilcoxon p = 0.0020 (need < 0.05) | SUPPORTED |
| SB2 A2C-n64 | paired SPR - LEDGER (n = 20): mean -2.27, one-sided 95% bootstrap UB +0.33 (need < 5) | SUPPORTED |
