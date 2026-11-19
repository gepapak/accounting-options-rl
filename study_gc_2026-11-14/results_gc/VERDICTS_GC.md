# Study GC verdicts

- runs: crypto ORIG 10 (expected 10), crypto FLPATCH 10 (expected 10), crypto LINE 10 (expected 10), forex ORIG 5 (expected 5), forex FLPATCH 5 (expected 5)
- GC1 (crypto) x_ORIG(0.1) - x_FLPATCH(0.1) = 0.296 [0.050, 0.514] (need lower bound > 0) -> SUPPORTED
- GC2 (crypto) mean x over e<=0.25, ORIG - FLPATCH = 0.200 [0.095, 0.289] (need lower bound > 0) -> SUPPORTED
- GC3 (crypto) ORIG reported - booked = 3.36 pp [1.13, 6.26] (need lower bound > 0) -> SUPPORTED
- GC4 (crypto) mean x over e<=0.25: ORIG 0.278, LINE 0.054, one-sided Mann-Whitney p = 0.0009 (need < 0.05), and LINE at most FLPATCH + 0.10 = 0.178 -> SUPPORTED
- GC5 (crypto) gap of ORIG - gap of LINE = 2.86 pp [0.50, 5.82] (need lower bound > 0); LINE gap 0.50 pp -> SUPPORTED
- GC6 (forex) x_ORIG(0.1) - x_FLPATCH(0.1) = 0.000 (need absolute value at most 0.10), ORIG reported - booked = 0.00 pp (need at most 0.50) -> SUPPORTED
- GC7 (both) largest error of the reading against the booked step profit, every test step of every agent and every LINE training step: 4.398e+00 USD (need at most 0.001) -> NOT SUPPORTED
- descriptive: training steps on which the floor wrote, mean per run: crypto ORIG 6258, crypto LINE 5588, forex ORIG 408; x(0.1): crypto LINE 0.062 against ORIG 0.427

```
                  x0.02  x0.05   x0.1  x0.25   x0.5     x1     x2   xlow  reported_return_pct  booked_return_pct    gap  floored_share  wiped_share  orders_per_episode  max_leverage  train_floor_steps
market condition                                                                                                                                                                                        
crypto FLPATCH    0.025  0.102  0.131  0.054  0.014  0.006  0.003  0.078                0.166             -0.482  0.648          0.065        0.065               0.625         7.752                0.0
       LINE       0.006  0.125  0.062  0.024  0.008  0.005  0.000  0.054               -0.009             -0.512  0.503          0.059        0.059               7.637         5.022             5588.3
       ORIG       0.072  0.409  0.427  0.203  0.069  0.018  0.010  0.278                1.917             -1.445  3.362          0.222        0.222              21.276        22.192             6258.1
forex  FLPATCH    0.000  0.000  0.000  0.000  0.000  0.000  0.004  0.000                0.000              0.000  0.000          0.000        0.000               0.000         0.000                0.0
       ORIG       0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000                0.000              0.000  0.000          0.000        0.000               0.000         0.000              408.4
```
