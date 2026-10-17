# Study R verdicts

One internally hash-stamped composite primary decision; all 60 planned agents included.

| ID | Decision | Verdict |
|---|---|---|
| R1 | G(.4,.8)=4.500 >=4; G(1.5,.8)=4.475 <4; interaction=0.025 >=4; exact seed sign p=0.81250 <=.05 | NOT SUPPORTED |

## Every arm (descriptive except the R1 components)

| Fee | alpha | index (%) | mean L(.02) | gap over matched control | max16 runs |
|---|---|---|---|---|---|
| 0.4 | 0 | 1.157656 | 16.000 | 12.900 | 10/10 |
| 0.4 | 0.8 | 0.156848 | 7.600 | 4.500 | 4/10 |
| 0.4 | 1 | 0.000000 | 3.100 | 0.000 | 1/10 |
| 1.5 | 0 | 0.891680 | 16.000 | 14.875 | 10/10 |
| 1.5 | 0.8 | 0.026688 | 5.600 | 4.475 | 3/10 |
| 1.5 | 1 | 0.000000 | 1.125 | 0.000 | 0/10 |

Paired seed interactions: {'60': -1.5, '61': 0.125, '62': 9.0, '63': -7.5, '64': 0.0}

The sign test targets the probability of a positive seed-cluster interaction under independent seed clusters, not a population mean or generalization to markets. Zeros count as nonpositive.
The G cutoffs are operational sample-cohort classifications, not equivalence confidence intervals. All prior retrospective failures and calibration uncertainty remain relevant.
