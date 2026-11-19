# Study GC verdicts (SMOKE)

- runs: crypto ORIG 1 (expected 10), crypto FLPATCH 1 (expected 10), crypto LINE 1 (expected 10), forex ORIG 1 (expected 5), forex FLPATCH 1 (expected 5)
- GC1 (crypto) x_ORIG(0.1) - x_FLPATCH(0.1) = 0.006 [0.006, 0.006] (need lower bound > 0) -> SUPPORTED
- GC2 (crypto) mean x over e<=0.25, ORIG - FLPATCH = 0.013 [0.013, 0.013] (need lower bound > 0) -> SUPPORTED
- GC3 (crypto) ORIG reported - booked = 0.00 pp [0.00, 0.00] (need lower bound > 0) -> NOT SUPPORTED
- GC4 (crypto) mean x over e<=0.25: ORIG 0.013, LINE 0.055, one-sided Mann-Whitney p = 1.0000 (need < 0.05), and LINE at most FLPATCH + 0.10 = 0.100 -> NOT SUPPORTED
- GC5 (crypto) gap of ORIG - gap of LINE = 0.00 pp [0.00, 0.00] (need lower bound > 0); LINE gap 0.00 pp -> NOT SUPPORTED
- GC6 (forex) x_ORIG(0.1) - x_FLPATCH(0.1) = 0.000 (need absolute value at most 0.10), ORIG reported - booked = 0.00 pp (need at most 0.50) -> SUPPORTED
- GC7 (both) largest error of the reading against the booked step profit, every test step of every agent and every LINE training step: 1.179e-09 USD (need at most 0.001) -> SUPPORTED
- descriptive: training steps on which the floor wrote, mean per run: crypto ORIG 99, crypto LINE 98, forex ORIG 8; x(0.1): crypto LINE 0.000 against ORIG 0.006

```
                  x0.02  x0.05   x0.1  x0.25  x0.5   x1   x2   xlow  reported_return_pct  booked_return_pct  gap  floored_share  wiped_share  orders_per_episode  max_leverage  train_floor_steps
market condition                                                                                                                                                                                 
crypto FLPATCH    0.000  0.000  0.000  0.000   0.0  0.0  0.0  0.000                  0.0                0.0  0.0            0.0          0.0                 0.0           0.0                0.0
       LINE       0.156  0.062  0.000  0.000   0.0  0.0  0.0  0.055                  0.0                0.0  0.0            0.0          0.0                 0.0           0.0               98.0
       ORIG       0.031  0.012  0.006  0.002   0.0  0.0  0.0  0.013                  0.0                0.0  0.0            0.0          0.0                 0.0           0.0               99.0
forex  FLPATCH    0.000  0.000  0.000  0.000   0.0  0.0  0.0  0.000                  0.0                0.0  0.0            0.0          0.0                 0.0           0.0                0.0
       ORIG       0.000  0.000  0.000  0.000   0.0  0.0  0.0  0.000                  0.0                0.0  0.0            0.0          0.0                 0.0           0.0                8.0
```
