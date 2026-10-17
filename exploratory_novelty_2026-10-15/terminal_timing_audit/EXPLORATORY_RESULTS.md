# Terminal timing mismatch: quantified materiality

Exploratory audit, 15 October 2026; no new learning and no changes to frozen
Study N or running Study R. The issue is one extra discount factor on the
terminal forgiveness bonus in the registered DP, relative to the environment.
The script changes only the boundary value from D to D/.99.

## Isolation check

Recomputing the original 300-bin DP with sparse interpolation exactly matches
all **108 registered TFG first-action choices**. Its largest absolute
optimal-versus-flat gap discrepancy from the stored frozen JSON is
**4.55e−11 EUR in DK1** and **5.01e−11 EUR in DK2**. This isolates the timing
change from data, binning, grid, clipping and tie-breaking changes. The booked
control has zero value and a flat optimum analytically; all 108 stored TBK
probes match it.

## Corrected timing

- **0/108 TFG registered first actions change.**
- **0/108 TBK registered first actions change.**
- The largest absolute change in a registered optimal-versus-flat action gap
  is **0.634571455 EUR**, at DK2, equity .02, tau 1: **59.937671336 →
  60.572242791 EUR**.
- Among nonzero registered gaps, the largest relative change is **14.50%**
  at DK1, equity .5, tau 96. This is a small absolute change,
  **0.119457988 → 0.136784540 EUR** (increase 0.017326552 EUR), with the
  same maximal-leverage action. Relative changes should therefore not be
  summarized as uniformly around one percent.
- Over all 1,801 ledger points and 192 horizons in both zones,
  **780/691,584 state-times (0.1128%)** change their selected first action.
  The zone counts are 410 in DK1 and 370 in DK2. The largest full-grid gap
  change is 1.557577104 EUR. Every registered tau-192 TFG probe remains flat.

Thus the mismatch is real and should be disclosed. Its correction preserves
the registered-probe policy comparisons used in the paper, while slightly
changing quantitative gaps and some off-probe policy boundaries. No claim
is made that the complete optimal policy is unchanged, or that all numerical
gaps can be reused without qualification.

All five action values, both optimal-minus-flat and maximal-leverage-minus-
flat contrasts, and their corrections are retained for every probe in the
JSON/CSV. The original learned-agent outcomes and supported/unsupported
decisions are unchanged. The frozen DP is not repaired in place.

The numerical run took **10.49 seconds** with one numerical-library thread.
Source and script hashes, as well as the full state-time change counts by
horizon, are recorded alongside the results. Existing independent R workers
were not interrupted or modified.
