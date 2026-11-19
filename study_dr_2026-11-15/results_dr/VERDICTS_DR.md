# Study DR verdicts

- runs: V2 ORIG 6 (expected 6), V4 ORIG 10 (expected 10), V8 ORIG 10 (expected 10), V16 ORIG 10 (expected 10), L30 ORIG 6 (expected 6), L500 ORIG 10 (expected 10), V16 FLPATCH 6 (expected 6), L500 FLPATCH 6 (expected 6)
- DR1 (volatility randomization switches the floor on) low-equity exposure, floor-trained minus V16 floor-deleted controls; V8: 0.128 - 0.045 = 0.082 [-0.021, 0.160], learners 8/10; V16: 0.246 - 0.045 = 0.201 [0.077, 0.311], learners 8/10 (need both lower bounds > 0) -> NOT SUPPORTED
- DR2 (leverage 500 switches the floor on at nominal volatility) L500: 0.041 - 0.014 = 0.027 [-0.028, 0.085], learners 2/10 (need lower bound > 0) -> NOT SUPPORTED
- DR3 (dormant where the published edge is not positive) V2: learners 3/6, mean low-equity exposure 0.146; L30: learners 0/6, mean low-equity exposure 0.000 (need at most 1 in each) -> NOT SUPPORTED
- DR4 (dose-response) Jonckheere-Terpstra over V2 < V4 < V8 < V16 on low-equity exposure: J = 303.0 of 480 pairs, one-sided permutation p = 0.03474 (need < 0.05); means V2 0.146, V4 0.115, V8 0.128, V16 0.246 -> SUPPORTED

## Descriptive

- V2 ORIG: n 6, learners 3, low-equity exposure 0.146 (max 0.411), x(1) 0.010; training episodes reaching the floor 0.120 (before training, initial policy 0.091); published edge C -1.77%; nominal test reported -0.02 booked -0.02; stressed test reported +0.47 booked +0.35 wiped 0.024
- V4 ORIG: n 10, learners 4, low-equity exposure 0.115 (max 0.358), x(1) 0.014; training episodes reaching the floor 0.274 (before training, initial policy 0.223); published edge C +2.06%; nominal test reported -0.13 booked -0.13; stressed test reported -0.32 booked -0.59 wiped 0.048
- V8 ORIG: n 10, learners 8, low-equity exposure 0.128 (max 0.230), x(1) 0.017; training episodes reaching the floor 0.401 (before training, initial policy 0.364); published edge C +12.53%; nominal test reported -0.04 booked -0.04; stressed test reported +0.55 booked +0.41 wiped 0.028
- V16 ORIG: n 10, learners 8, low-equity exposure 0.246 (max 0.430), x(1) 0.000; training episodes reaching the floor 0.474 (before training, initial policy 0.468); published edge C +34.42%; nominal test reported +0.00 booked +0.00; stressed test reported +0.00 booked +0.00 wiped 0.000
- V16 FLPATCH: n 6, learners 1, low-equity exposure 0.045 (max 0.271), x(1) 0.000; training episodes reaching the floor 0.000 (before training, initial policy 0.468); published edge C +34.42%; nominal test reported +0.00 booked +0.00; stressed test reported +0.00 booked +0.00 wiped 0.000
- L30 ORIG: n 6, learners 0, low-equity exposure 0.000 (max 0.000), x(1) 0.031; training episodes reaching the floor 0.000 (before training, initial policy 0.000); published edge C -0.74%; nominal test reported -0.02 booked -0.02; stressed test reported +0.12 booked +0.07 wiped 0.012
- L500 ORIG: n 10, learners 2, low-equity exposure 0.041 (max 0.218), x(1) 0.002; training episodes reaching the floor 0.267 (before training, initial policy 0.317); published edge C +13.94%; nominal test reported -0.00 booked -0.00; stressed test reported +0.12 booked +0.05 wiped 0.010
- L500 FLPATCH: n 6, learners 0, low-equity exposure 0.014 (max 0.085), x(1) 0.001; training episodes reaching the floor 0.000 (before training, initial policy 0.317); published edge C +13.94%; nominal test reported -0.01 booked -0.01; stressed test reported +0.00 booked +0.00 wiped 0.000
- floor-deleted controls with low-equity exposure >= 0.1: 1 of 12

```
                x0.02  x0.05   x0.1  x0.25   x0.5     x1     x2   xlow  learner  train_floored_share  nom_reported_return_pct  nom_booked_return_pct  k4_reported_return_pct  k4_booked_return_pct  k4_wiped_share
arm  condition                                                                                                                                                                                                    
L30  ORIG       0.000  0.000  0.000  0.000  0.051  0.031  0.008  0.000    0.000                0.000                   -0.016                 -0.016                   0.120                 0.066           0.012
L500 FLPATCH    0.032  0.014  0.008  0.004  0.003  0.001  0.001  0.014    0.000                0.000                   -0.008                 -0.008                   0.005                 0.005           0.000
     ORIG       0.055  0.069  0.033  0.007  0.000  0.002  0.002  0.041    0.200                0.267                   -0.003                 -0.003                   0.116                 0.052           0.010
V16  FLPATCH    0.104  0.042  0.023  0.012  0.000  0.000  0.005  0.045    0.167                0.000                    0.000                  0.000                   0.000                 0.000           0.000
     ORIG       0.100  0.288  0.443  0.155  0.074  0.000  0.000  0.246    0.800                0.474                    0.000                  0.000                   0.000                 0.000           0.000
V2   ORIG       0.000  0.167  0.311  0.105  0.017  0.010  0.005  0.146    0.500                0.120                   -0.020                 -0.020                   0.470                 0.349           0.024
V4   ORIG       0.050  0.160  0.172  0.080  0.050  0.014  0.000  0.115    0.400                0.274                   -0.131                 -0.131                  -0.322                -0.587           0.048
V8   ORIG       0.000  0.040  0.191  0.279  0.055  0.017  0.004  0.128    0.800                0.401                   -0.042                 -0.042                   0.551                 0.405           0.028
```
