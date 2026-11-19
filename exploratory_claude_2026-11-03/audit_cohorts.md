| cohort | floor | learner | agents | gamblers | max TIR error edge / placebo | non-gamblers: max edge / placebo | gamblers: median reported error (placebo) |
|---|---|---|---|---|---|---|---|
| V, Parts A, C | single | PPO | 128 | 28 | 0.3845 / 0.1286 | 0.0019 / 0.0147 | 68.25 |
| V, Part B | single | PPO | 16 | 0 | 0.0000 / 0.0009 | 0.0000 / 0.0009 | nan |
| W | single | PPO | 48 | 1 | 0.0771 / 0.0634 | 0.0102 / 0.0046 | 23.94 |
| X | two writes | PPO | 32 | 8 | 1.5658 / 0.1889 | 0.0013 / 0.0065 | 93.75 |
| Y, zero starts | single | A2C | 32 | 5 | 0.1041 / 0.0375 | 0.0000 / 0.0009 | 51.01 |
| Y, no zero starts | single | A2C | 16 | 0 | 0.0000 / 0.0009 | 0.0000 / 0.0009 | nan |
| Y, zero starts | two writes | A2C | 32 | 5 | 1.3935 / 0.2506 | 0.0013 / 0.0025 | 51.27 |

Single-floor cohorts ['V_AC', 'V_B', 'W', 'Y_S', 'Y_N']: 240 agents, max TIR error 0.3845 (edge) / 0.1286 (placebo).
