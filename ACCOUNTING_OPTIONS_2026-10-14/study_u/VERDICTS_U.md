# Study U verdicts

Runs: 80 of 80.

## settlement lag (zombie window)

| arm | runs | mean X | mean booked (% of allocation) | mean reported |
|---|---|---|---|---|
| ZFG | 10 | 16.00 | -9.0 | -6.0 |
| ZBK | 10 | 1.18 | -13.6 | -13.6 |
| ZSPR | 10 | 1.45 | -11.5 | -11.1 |
| ZTIR | 10 | 1.21 | -10.7 | -10.7 |

## terminal limited liability

| arm | runs | mean X | mean booked (% of allocation) | mean reported |
|---|---|---|---|---|
| TFG | 10 | 10.84 | -0.7 | -0.6 |
| TBK | 10 | 1.59 | -0.2 | -0.2 |
| TSPR | 10 | 0.79 | -0.2 | -0.2 |
| TTIR | 10 | 0.38 | -0.1 | -0.1 |

| ID | Result | Verdict |
|---|---|---|
| U1 Z | X(ZFG) - X(ZBK) = +14.82 (need >= 8), one-sided 95% bootstrap lower bound +14.26 (need > 4) | SUPPORTED |
| U2 Z | X(ZFG) - X(ZSPR) = +14.55 (need >= 4), lower bound +13.82 (need > 0); X(ZSPR) - X(ZBK) = +0.27 (need < 4) | SUPPORTED |
| U3 Z | X(ZFG) - X(ZTIR) = +14.79 (need >= 4), lower bound +13.41 (need > 0); X(ZTIR) - X(ZBK) = +0.04 (need < 4) | SUPPORTED |
| U1 T | X(TFG) - X(TBK) = +9.25 (need >= 8), one-sided 95% bootstrap lower bound +6.38 (need > 4) | SUPPORTED |
| U2 T | X(TFG) - X(TSPR) = +10.05 (need >= 4), lower bound +7.55 (need > 0); X(TSPR) - X(TBK) = -0.80 (need < 4) | SUPPORTED |
| U3 T | X(TFG) - X(TTIR) = +10.46 (need >= 4), lower bound +7.99 (need > 0); X(TTIR) - X(TBK) = -1.21 (need < 4) | SUPPORTED |
