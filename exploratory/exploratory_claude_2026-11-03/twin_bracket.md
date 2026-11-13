# Twin bracket on stored audits: 304 agents, 60,800 episodes; violations 0; negative widths 0; one-sided 95% lower bound below the booked mean in 608 of 608

| study | floor | learner | agents | agent-markets | episodes | violations | median width | max width | median booked - LB95 | max TIR error | TIR outside bracket (episodes) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V | single | PPO | gamblers | 56 | 5600 | 0 | 70.1844 | 482.18 | 56.3006 | 0.3845 | 0 |
| V | single | PPO | non-gamblers | 232 | 23200 | 0 | 0.1700 | 94.31 | 3.0120 | 0.0147 | 0 |
| W | single | PPO | gamblers | 2 | 200 | 0 | 31.6110 | 294.03 | 27.1208 | 0.0771 | 0 |
| W | single | PPO | non-gamblers | 94 | 9400 | 0 | 0.9208 | 77.90 | 4.2346 | 0.0102 | 0 |
| X | two writes | PPO | gamblers | 16 | 1600 | 0 | 87.1774 | 484.16 | 60.6927 | 1.5658 | 0 |
| X | two writes | PPO | non-gamblers | 48 | 4800 | 0 | 0.8746 | 38.26 | 3.0120 | 0.0065 | 0 |
| Y | single | A2C | gamblers | 10 | 1000 | 0 | 58.4310 | 283.45 | 41.5684 | 0.1041 | 0 |
| Y | single | A2C | non-gamblers | 86 | 8600 | 0 | 0.9337 | 38.09 | 3.0120 | 0.0009 | 0 |
| Y | two writes | A2C | gamblers | 10 | 1000 | 0 | 77.1897 | 480.42 | 54.9273 | 1.3935 | 0 |
| Y | two writes | A2C | non-gamblers | 54 | 5400 | 0 | 1.1545 | 38.26 | 3.0529 | 0.0025 | 0 |

Engine hand cases (own cost = fee + funding):

| case | r | r' | C | bracket | booked | covered | TIR | TIR exact |
|---|---|---|---|---|---|---|---|---|
| NO_COST | 1 | -1 | 0 | [1, 1] | 1 | True | 1 | True |
| FEE_FLOOR_NATIVE_TIR_COUNTEREXAMPLE | 1 | 0 | 10 | [-20, 1] | -9 | True | 1 | False |
| LOSING_REAL_BRANCH | -5 | 3 | 1 | [-5, -5] | -5 | True | -5 | True |
| POST_FUNDING_BOTH_CENSORED | 0 | 0 | 10 | [-20, 0] | -9 | True | -10 | False |
| PRE_FEE_POST_FUNDING | 0 | 0 | 12 | [-24, 0] | -11 | True | -12 | False |
| TWO_DELIVERIES | 2 | 0 | 10 | [-20, 2] | -8 | True | 2 | False |
