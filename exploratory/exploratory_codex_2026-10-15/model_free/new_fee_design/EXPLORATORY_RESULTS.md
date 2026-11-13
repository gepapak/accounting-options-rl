# Exploratory new-fee design result: two-fee bracket unavailable

The fixed design-only computation completed on 15 October 2026 in **34.66 seconds**, counting **61,620,932 logical branch transitions**. It evaluated all 32 fee/zone/alpha conditions in four shared rollout groups, using 24 batches × 256 independent antithetic pairs per group. It trained no agents and read no learner outcomes or DP outputs. All previous analysis artifacts were preserved.

Frozen c=.366768%; high eligibility requires the combined 95% MC-plus-tail **lower bound at least .550152% in both DK1 and DK2**. Low eligibility requires the combined **upper bound at most .244512% in both zones**. Alpha1 is control only. The pre-execution rule chooses the largest eligible high treatment alpha and smallest eligible low treatment alpha.

| Fee (EUR/MWh) | High treatment | Low treatment | Result |
|---|---|---|---|
| .4 | alpha.25 | unavailable | Fixed bracket fails |
| 1.5 | alpha.25 | alpha.9 | This fee alone brackets |

At fee .4, alpha.95's combined bands are [0,.251736]% in DK1 and [0,.904163]% in DK2, so it does not meet the low criterion. No other treatment alpha meets it either. DK2's alpha1 control has point index .271490% and combined band [0,.872013]%; control uncertainty/reference-policy mismatch also merits caution. These are simulated diagnostic contrasts, not evidence of a true exploit under full booking.

At fee1.5, the selected treatments have:

| Treatment | Zone | Point index (%) | Combined lower (%) | Combined upper (%) |
|---|---|---:|---:|---:|
| alpha.25, high | DK1 | 1.327251 | .912563 | 1.810687 |
| alpha.25, high | DK2 | 1.165092 | .735034 | 1.668333 |
| alpha.9, low | DK1 | .033141 | 0 | .195902 |
| alpha.9, low | DK2 | 0 | 0 | .078068 |

Both fee1.5 high bounds exceed .550152%; both low bounds are below .244512%. Alpha0 also qualifies high and alpha.95 qualifies low; the predetermined nearest-alpha rule selects .25 and .9. Alpha1 would be the matched control if a future study were separately registered.

The requested **two-fee, 60-run study cannot follow this fixed bracketing rule**. No additional fees, alternative references, larger sampling budget or weaker criteria were searched after this failure. No Study R files were created and no training was launched. A one-fee 30-run study would be a different proposed design requiring a separate decision and registration; this analysis does not silently substitute it.

Uncertainty is conditional on the empirical TRAIN sample. Intervals describe Monte Carlo error plus a conservative discount tail; they do not include the earlier calibrated threshold's uncertainty, future-agent variability, or unseen markets. Passing at fee1.5 is a design-feasibility result, not a learned-policy outcome. The primary model-free analysis already found imperfect near-transition calibration, so this should not be advertised as solved generality.

Full results, fixed choices and source/code hashes are in `outputs/exploratory_new_fee_indices.csv`, `outputs/exploratory_new_fee_summary.json`, the checkpoints and `exploratory_new_fee_design.log`. Run the unchanged `exploratory_new_fee_design.py` to reproduce/resume these settings. The default helper remains fixed and no DP optimization is used.
