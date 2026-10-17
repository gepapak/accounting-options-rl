# Study O verdicts

runs: 80 ({0.0: 10, 0.25: 10, 0.5: 10, 0.7: 10, 0.8: 10, 0.9: 10, 0.95: 10, 1.0: 10}); c* = 0.513%; control alpha = 1

| alpha | index (%) | DP region (alloc.) | mean L(0.02) | gap G | runs at 16 | prediction | outcome |
|---|---|---|---|---|---|---|---|
| 0 | 1.805 | 1.75 | 16.00 | +14.40 | 10/10 | learned | learned |
| 0.25 | 1.345 | 1.65 | 16.00 | +14.40 | 10/10 | learned | learned |
| 0.5 | 0.886 | 1.52 | 13.22 | +11.62 | 8/10 | learned | learned |
| 0.7 | 0.518 | 1.35 | 9.62 | +8.03 | 5/10 | learned (transition) | learned |
| 0.8 | 0.335 | 1.21 | 8.10 | +6.50 | 4/10 | not learned | learned |
| 0.9 | 0.154 | 0.98 | 1.50 | -0.10 | 0/10 | not learned | not learned |
| 0.95 | 0.064 | 0.75 | 3.02 | +1.42 | 1/10 | not learned | not learned |
| 1 (control) | 0.000 | 0.00 | 1.60 | | 0/10 | | |

| ID | Result | Verdict |
|---|---|---|
| O1 | confident arms classified correctly: 5/6 (need >= 5) | SUPPORTED |
| O2 | run-level Spearman(index, L(0.02)) = 0.746, one-sided permutation p = 0.0001 (need >= 0.4 and < 0.01) | SUPPORTED |
| O3 | alpha 0.9: DP region 0.98 (need >= 0.5 in both zones: True); gap -0.10 (need < 4); runs at 16: 0/10 (need < 5) | SUPPORTED |
| O4 | alpha 0 vs control: gap +14.40 (need >= 4), one-sided 95% bootstrap lower bound +13.50 (need > 0) | SUPPORTED |

## Exploratory
- logistic P(L(0.02) = 16) on log10(index): c50 = 0.429% (bootstrap 95% interval [0.302, 0.573]%), slope 5.20; registered c* = 0.513%
- per arm: mean learned region T vs DP region; reported vs booked return (% of K0, test draws):
  - alpha 0: T 1.12 (DP 1.75); reported +54.6, booked +4.4
  - alpha 0.25: T 0.57 (DP 1.65); reported +28.3, booked -0.3
  - alpha 0.5: T 0.30 (DP 1.52); reported +23.7, booked -1.0
  - alpha 0.7: T 0.28 (DP 1.35); reported +33.9, booked -0.2
  - alpha 0.8: T 0.11 (DP 1.21); reported +5.8, booked -4.6
  - alpha 0.9: T 0.00 (DP 0.98); reported +5.1, booked +3.9
  - alpha 0.95: T 0.05 (DP 0.75); reported +3.1, booked -5.7
  - alpha 1: T 0.00 (DP 0.00); reported +12.8, booked +11.7
