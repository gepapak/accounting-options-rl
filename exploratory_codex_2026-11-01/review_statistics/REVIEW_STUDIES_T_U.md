# Independent saved-result addendum: Studies T and U

All **16 T** and **6 U** registered verdicts independently reproduce from raw JSON. Exact rosters: T170, U80; every checked CSV scalar equals the raw result exactly (maximum discrepancy0). All U20-draw booked/reported means also recompute exactly. No result-affecting implementation or aggregation bug was found. No registered module, environment, learner, model or market data reader was imported/executed; no old file was edited.

Numerical record: `RECOMPUTATION_T_U.json` SHA256 `107e816977ef743719df9a3b80b03f973eca88bf876b3bda6aa0d6ab32bc3633`. Source: `recompute_studies_t_u.py`, using only the independently written rank arithmetic in `recompute_study_v.py`. Each study independently starts generator20261031; all bootstrap calls preserve source order, sample sizes, 10,000 resamples and fifth-percentile lower bounds. The additional U analysis below is explicitly post hoc sensitivity, not a replacement of registered decisions.

## Study T: all16 supported

| ID | Independent deciding statistic | Registered threshold |
|---|---|---|
| TA1 | p=.0000236570363020 | p<.05 |
| TA2 | SPR X=0; p=.0000164496840293; oracle X=.025 | p<.05; X≤oracle+.10 |
| TA3 | SPR/oracle=.947989439860; LB=.852420352423 | ratio≥.80; LB≥.60 |
| TA4 | TIR X=.025; p=.0000236570363020 | p<.05; X≤oracle+.10 |
| TA5 | TIR/oracle=1.04270347043; LB=.982876052490 | ratio≥.90; LB≥.75 |
| TA6 | none of five OM coefficients meets both | X≤oracle+.10 and E≥.80oracle |
| TB-DK2-1 | p=.00291488223111 | p<.05 |
| TB-DK2-2 | SPR X=0; p=.00291488223111 | p<.05; X≤oracle+.10 |
| TB-DK2-3 | SPR/oracle=1.02081442319; LB=.929947815886 | ratio≥.80; LB≥.60 |
| TB-DK2-4 | TIR X=.0125; p=.00723142852163 | p<.05; X≤oracle+.10 |
| TB-DK2-5 | TIR/oracle=1.03998910655; LB=.966900380181 | ratio≥.90; LB≥.75 |
| TB-DK1M-1 | p=.000354026403854 | p<.05 |
| TB-DK1M-2 | SPR X=0; p=.000354026403854 | p<.05; X≤oracle+.10 |
| TB-DK1M-3 | SPR/oracle=1.15619919818; LB=1.00008836909 | ratio≥.80; LB≥.60 |
| TB-DK1M-4 | TIR X=0; p=.000354026403854 | p<.05; X≤oracle+.10 |
| TB-DK1M-5 | TIR/oracle=1.13907197160; LB=.985530122864 | ratio≥.90; LB≥.75 |

The exact source IDs include `TB-T_DK2-*` and `TB-T_DK1M-*`; shortened table labels above refer to the same tests. Removal tests are one-sided in the correct direction. SPR and TIR intentionally have different edge-retention thresholds. Exploit-present gating and TA6's conjunction match the registration and code (`study_t/aggregate_t.py`, functions `core`, `removes`, `edge_ratio`, and the TA6 loop). No bootstrap oracle mean was nonpositive.

TA6 is a result for **this reference, this implementation and five tested coefficients**, not a general impossibility for occupancy regularization. OM1/.1/.05 all have X=0 and E=7.7020226953, below the required .8×20.4639557052=16.3711645641. OM.02 has X=.2875/E=.9445854134, and OM.01 X=.5375/E=-.0733026731.

## Study U: all6 supported, dependence sensitivity retains the conclusion

| ID | Mean FG−comparator | Registered LB | Same-seed paired-block LB | Independent-arm seed-block LB | Twin−BK |
|---|---:|---:|---:|---:|---:|
| U1 Z | 14.825000 | 14.262500 | 14.400000 | 14.400000 | n/a |
| U2 Z | 14.550000 | 13.825000 | 13.980000 | 14.000000 | .275000 |
| U3 Z | 14.789583 | 13.412500 | 13.333333 | 13.333333 | .035417 |
| U1 T | 9.255000 | 6.380000 | 7.363333 | 6.795000 | n/a |
| U2 T | 10.051667 | 7.551667 | 7.376333 | 7.806667 | -.796667 |
| U3 T | 10.465000 | 7.994833 | 8.200000 | 8.185000 | -1.210000 |

U1 requires point≥8 and LB>4. U2/U3 require point≥4, LB>0 and twin−BK<4. The saved aggregation implements exactly those inequalities and gates U2/U3 on U1 (`study_u/aggregate_u.py:77–88`). The “near BK” condition is one-sided, not equivalence or a two-sided distance requirement; the negative terminal differences correctly pass as registered.

The registered bootstrap samples ten model/zone rows independently within each arm (`aggregate_u.py:44–47`). Those rows arise from only five nominal learner seeds reused across two fixed zones (`run_u.py:31–32`, `:145`). To examine that dependence without changing any verdict, I averaged the two zones within each seed and retained both zones in every resample. The paired sensitivity exhaustively enumerates all5^5=3,125 resamples of matched seed differences. The second sensitivity independently resamples five seed blocks per arm with10,000 draws, seed20261031, retaining the original six-contrast order. All six numerical decision criteria still pass under either procedure. Neither analysis creates new market replications or a reliable large-sample theorem from five clusters; paired versus independent resampling is an assumption choice, not proof of the true covariance structure.

**Important outcome boundary:** U's registered X is greedy probe leverage, not booked-profit improvement or absence of realized forgiveness. In the zombie setting FG's mean booked return is -9.00340955%, SPR -11.46547481%, TIR -10.68030516%, and BK -13.60169084%. Thus the supported probe result does not show a utility improvement over FG. Reported minus booked return remains .3184909375 points for ZSPR and .0173728125 for ZTIR. A claim that these runs eliminate every realized credit would be false; the demonstrated result is attenuation of the specified insolvent/terminal leverage tendency toward the booked-control level. Terminal SPR/TIR means also remain negative in the fair market, although TIR's loss is smaller. None of these descriptive returns alters the registered X verdicts.

## Bounded ORPO implementation review

The local baseline actually uses state-action occupancy discrimination, not only per-state action KL. `orpo_t.py:72–82` obtains fresh reference rollouts; `:84–99` applies the stated current-positive/reference-negative logistic loss; `:107–117` uses current state-action samples and exp(d)−1 with the disclosed trimmed normalization; `:118–123` adds the penalty and recomputes returns/advantages. Source choices such as discriminator architecture, d clipping and normalization are disclosed. This bounded review does not replace the parent's comparison with the upstream implementation or establish its theorem's premises for this approximate neural discriminator.

“Safe reference” needs an explicit definition. `orpo_t.py:33–38` samples the signal-following leverage1 action with probability.7 and each other action with probability.0375. Hence **7.5% of requests are maximal leverage±16**, even though the greedy/modal reference has only leverage1. The actual reference rollouts used by the discriminator and sigma estimate are stochastic (`:76`, `:109–111`). A positive booked return for the greedy reference does not establish positive booked performance or accounting safety for that stochastic reference. The parent has separately computed approximately-9.4% for the stochastic reference versus+7.7% for the greedy rule; those source/data calculations were not repeated here. I independently confirm that these are different policies and cannot be interchanged in the comparator description. “Full-support low-leverage-mode reference” is more precise than an unqualified safe policy.

Large ORPO coefficients converging to the greedy reference's modest return are compatible with that reference choice and do not demonstrate an inherent inability of occupancy regularization to preserve an edge with a stronger reference. A weak coefficient's failure and a strong coefficient's imitation are still legitimate registered observations. No evidence of a result-affecting coding bug was found; criticism of reference quality or discriminator approximation is a baseline/design limitation, not a reason to modify registered results or silently substitute a new baseline.

No fixes, new experiments, training calls, source-environment calls or Git operations were performed. Timing, registered-prefix hashing, publication claims and the upstream ORPO comparison remain assigned to the parent review.
