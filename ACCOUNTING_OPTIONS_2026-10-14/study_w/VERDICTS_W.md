# Study W verdicts

Runs: 48 of 48.

## T_DK2

Runs per strength: {'RUIN1': 8, 'RUIN5': 8, 'RUIN25': 8}. References (Study T): ORACLE 10 runs (X 0.000, E +32.66), PLAIN 10 runs (X 0.394, E +21.85).

| strength | runs | mean X | mean E (booked) | mean T (TIR audit) | mean R (reported) | both? |
|---|---|---|---|---|---|---|
| RUIN1 | 8 | 0.078 | +30.85 | +30.84 | +31.36 | True |
| RUIN5 | 8 | 0.000 | +33.27 | +33.27 | +33.33 | True |
| RUIN25 | 8 | 0.000 | +30.58 | +30.58 | +30.65 | True |

Audit-selected RUIN5 (E +33.27); booked-best RUIN5 (E +33.27); report-selected RUIN5 (E +33.27).

## T_DK1M

Runs per strength: {'RUIN1': 8, 'RUIN5': 8, 'RUIN25': 8}. References (Study T): ORACLE 10 runs (X 0.000, E +11.63), PLAIN 10 runs (X 0.388, E +6.72).

| strength | runs | mean X | mean E (booked) | mean T (TIR audit) | mean R (reported) | both? |
|---|---|---|---|---|---|---|
| RUIN1 | 8 | 0.000 | +11.22 | +11.22 | +11.29 | True |
| RUIN5 | 8 | 0.000 | +10.03 | +10.03 | +10.12 | True |
| RUIN25 | 8 | 0.000 | +7.32 | +7.32 | +7.39 | False |

Audit-selected RUIN1 (E +11.22); booked-best RUIN1 (E +11.22); report-selected RUIN1 (E +11.22).

| ID | Result | Verdict |
|---|---|---|
| W1-T_DK2 | audit-selected RUIN5 E +33.27 vs booked-best RUIN5 E +33.27: shortfall 0.00 (need <= 1.0) | SUPPORTED |
| W2-T_DK2 | audit-selected RUIN5: X 0.000 (need <= ORACLE 0.000 + 0.10), E +33.27 (need >= +26.13 = 0.80 x ORACLE +32.66) | SUPPORTED |
| W1-T_DK1M | audit-selected RUIN1 E +11.22 vs booked-best RUIN1 E +11.22: shortfall 0.00 (need <= 1.0) | SUPPORTED |
| W2-T_DK1M | audit-selected RUIN1: X 0.000 (need <= ORACLE 0.000 + 0.10), E +11.22 (need >= +9.31 = 0.80 x ORACLE +11.63) | SUPPORTED |
| W3 | max |TIR - booked| = 0.077 (edge market), 0.063 (placebo market) over 48 agents (need <= 1.0) | SUPPORTED |
| W4 | booked-best minus report-selected mean E: T_DK2 0.00 (RUIN5), T_DK1M 0.00 (RUIN1) (need >= 1.0 in at least one setting) | NOT SUPPORTED |
