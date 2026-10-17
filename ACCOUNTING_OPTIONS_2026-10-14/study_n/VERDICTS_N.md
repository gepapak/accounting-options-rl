# Study N verdicts

runs per arm: {'ZFG': 10, 'ZBK': 10, 'TFG': 10, 'TBK': 10}

| ID | Decision rule | Result | Verdict |
|---|---|---|---|
| N1 | mean zombie-probe |lev| ZFG - ZBK >= 8, lower bound > 4 | 11.91 [8.94, 14.08] | SUPPORTED |
| N2 | ZFG |lev| = 16 at (n=-0.1, k=48) in >= 8 of 10 runs | 10/10 | SUPPORTED |
| N3 | solvent |lev| at e=0.1: ZFG - ZBK, lower bound > 0 | 11.68 [7.03, 15.12] | SUPPORTED |
| N4 | ZBK mean zombie-probe |lev| <= 1 in >= 8 of 10 runs | 3/10 | NOT SUPPORTED |
| T1 | mean probe |lev| (L in {0.02,0.1,0.25}, tau <= 96) TFG - TBK >= 8, lower bound > 4 | 9.12 [4.58, 13.11] | SUPPORTED |
| T2 | TFG |lev| = 16 at (L=0.1, tau=16) in >= 8 of 10 runs | 8/10 | SUPPORTED |
| T3 | insolvent-before-horizon |lev| (L in {-0.25,-0.1}) TFG - TBK, lower bound > 0 | 8.42 [3.75, 12.37] | SUPPORTED |
| T4 | TFG |lev| at (L=0.1, tau=16) > at (L=0.1, tau=192) in >= 8 of 10 runs | 3/10 | NOT SUPPORTED |
| X1 | reported - booked > 0 in >= 8 of 10 runs, in ZFG and in TFG | ZFG 5/10, TFG 1/10 | NOT SUPPORTED |

## Exploratory

- ZFG: zombie-probe |lev| 16.00; solvent e=0.1 14.43; eval zombie-step max-leverage share 0.23; forgiven 4.1% of K0; reported -13.7% vs booked -17.8%
- ZBK: zombie-probe |lev| 4.09; solvent e=0.1 2.75; eval zombie-step max-leverage share 0.04; forgiven 0.0% of K0; reported -19.9% vs booked -19.9%
- TFG: probe |lev| 11.43; insolvent probes 11.17; reported -0.10% vs booked -0.10% per 2-day horizon
- TBK: probe |lev| 2.31; insolvent probes 2.76; reported -0.05% vs booked -0.05% per 2-day horizon
