# Study Q verdicts

runs: 100 ({('A2C-n64', 0.0): 10, ('A2C-n64', 0.5): 10, ('A2C-n64', 0.8): 10, ('A2C-n64', 0.9): 10, ('A2C-n64', 1.0): 10, ('DQN-default', 0.0): 10, ('DQN-default', 0.5): 10, ('DQN-default', 0.8): 10, ('DQN-default', 0.9): 10, ('DQN-default', 1.0): 10})

## A2C-n64

| alpha | index (%) | runs at 16 | mean L(0.02) | gap over control | mean L(2) |
|---|---|---|---|---|---|
| 0 | 1.805 | 10/10 | 16.00 | +12.82 | 7.35 |
| 0.5 | 0.886 | 4/10 | 7.35 | +4.17 | 1.00 |
| 0.8 | 0.335 | 4/10 | 7.75 | +4.58 | 4.95 |
| 0.9 | 0.154 | 2/10 | 4.30 | +1.12 | 3.00 |
| 1 | 0.000 | 1/10 | 3.17 | +0.00 | 2.62 |

exploratory: logistic c50 = 0.538% (slope 3.06); Study O (PPO) c50 = 0.429%

## DQN-default

| alpha | index (%) | runs at 16 | mean L(0.02) | gap over control | mean L(2) |
|---|---|---|---|---|---|
| 0 | 1.805 | 9/10 | 14.43 | +5.20 | 9.00 |
| 0.5 | 0.886 | 8/10 | 12.93 | +3.70 | 8.80 |
| 0.8 | 0.335 | 5/10 | 9.40 | +0.18 | 9.32 |
| 0.9 | 0.154 | 6/10 | 10.53 | +1.30 | 9.03 |
| 1 | 0.000 | 5/10 | 9.22 | +0.00 | 11.72 |

exploratory: logistic c50 = 0.153% (slope 1.73); Study O (PPO) c50 = 0.429%

| ID | Result | Verdict |
|---|---|---|
| Q1 A2C-n64 | run-level Spearman(index, L(0.02)) = 0.500, one-sided permutation p = 0.0003 (need > 0 and p < 0.05) | SUPPORTED |
| Q2 A2C-n64 | alpha 0 vs control: gap +12.82 (need >= 4), one-sided 95% bootstrap lower bound +10.12 (need > 0) | SUPPORTED |
| Q3 A2C-n64 | alpha 0.9 vs control: gap +1.12 (need < 4) | SUPPORTED |
| Q1 DQN-default | run-level Spearman(index, L(0.02)) = 0.309, one-sided permutation p = 0.0149 (need > 0 and p < 0.05) | SUPPORTED |
| Q2 DQN-default | alpha 0 vs control: gap +5.20 (need >= 4), one-sided 95% bootstrap lower bound +0.83 (need > 0) | SUPPORTED |
| Q3 DQN-default | alpha 0.9 vs control: gap +1.30 (need < 4) | SUPPORTED |
