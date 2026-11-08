# Study RE verdicts

- runs: 50 (expected 50)

 scale  reach_R  n_pairs  mean_D  pairs_D_positive  pairs_identical_probe  pairs_identical_params  ZF_mean_X  FL_mean_X  ZF_mean_L002  FL_mean_L002  ZF_floor_episodes_min  ZF_floor_episodes_max  FL_closed_episodes_min  FL_closed_episodes_max  ZF_reported_minus_ledger_pct
 1.000  855.823        5  13.150                 4                      0                       0     15.450      2.300        16.000         4.900                    609                    737                     281                    1144                        16.740
 0.300  442.829        5  15.550                 5                      0                       0     14.900     -0.650        16.000         0.500                    280                    372                      65                     246                         0.635
 0.100  148.502        5   6.150                 3                      1                       0      9.150      3.000         9.800         3.650                     10                    102                       5                     103                         0.000
 0.030    5.729        5  -3.150                 0                      4                       2     -0.200      2.950         3.500         6.650                      0                      4                       0                       6                         0.000
 0.010    0.000        5   0.000                 0                      5                       5     -0.550     -0.550         0.700         0.700                      0                      0                       0                       0                         0.000

- RE1 (reach is necessary): c=0.01 identical probes in 5/5 pairs, identical parameters in 5/5, ZF floor-reaching training episodes max 0.0 (need all pairs identical on the probe grid and 0 floor episodes) -> SUPPORTED
- RE2 (the exploit where the reach is large): c=1 mean D = 13.15, D > 0 in 4/5 pairs (need mean >= 4 and >= 4 of 5) -> SUPPORTED
- RE3 (the exploit follows the reach): mean D at c=1, 0.3, 0.1 = 13.15, 15.55, 6.15; at c=0.03, 0.01 = -3.15, 0.00 (need every high >= every low + 2) -> SUPPORTED
- Descriptive (floor-reaching ZF training episodes against the predicted reach R at maximal leverage): c=1: R=855.8, observed 609.0-737.0; c=0.3: R=442.8, observed 280.0-372.0; c=0.1: R=148.5, observed 10.0-102.0; c=0.03: R=5.7, observed 0.0-4.0; c=0.01: R=0.0, observed 0.0-0.0
