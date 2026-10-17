# Sample-trajectory manuscript integration drafts

16 October 2026. Drafts only. No main manuscript, bibliography, frozen theory artifact, checkpoint, experiment plan or result was changed. No experiment or build was run. Native comparison results were not read. All result insertions are explicit `PENDING` markers.

## Integration route and page budget

Replace the approximately three-page main `sections/predictable_audit.tex` with `predictable_audit_sample_only_draft.tex`, retaining `sec:adaptiveaudit`. This replacement is intentionally compact and has no extra main results table yet. Actual pagination is unverified: root should build after inserting final verified results and keep the main manuscript at or below 50 pages. A short table may need to replace prose, rather than add to it.

Append `supp_sample_only_audit_draft.tex` to the supplement. Keep the existing exact-oracle S19 theory, repair theorem, held-out benchmark, fixed-policy comparison, Pareto results and negative findings. The replacement main section refers to `app:predictableaudit` and the new `app:sampleonlyaudit`; the supplement also uses existing `app:predictablebenchmark` and `prop:kernelrepair`.

If removing the current main comparison table, move it intact to the existing predictable benchmark supplement and retain `tab:predictablecomparison`, after checking that it is not already duplicated there. Its evidence remains important: selected 36/72 versus 24/72 and neutral 33/72 versus 24/72 describe the exact empirical-kernel method, with no additional positive state; Pareto confirms 3/16 for both methods and raises predictable variance in 4/16. Do not attribute these results to the new sample-trajectory method.

Merge the four entries in `verified_references.bib` only after checking current main keys for duplicates. The 2018 regression paper is the closest required addition. The 2024 MDP and 2016 DR entries are requested context; the neural paper belongs mainly in the supplement. Do not cite the original 2018 arXiv title and the published neural paper as two works: they are versions of the same arXiv record.

## Closest prior art and candid novelty judgment

**The same-model two-sign coefficient is established prior art.** Belomestny et al. (2018), Proposition 3.3, Equation (3.8), gives the branch difference for one Rademacher innovation; Section 4.2, Equation (4.3), keeps a fitted control conditionally centered on independent test paths. [Primary full text](https://arxiv.org/pdf/1510.03141), [author-institution bibliographic record](https://www.uni-due.de/mathematik/ag_belomestny/publikationsliste).

Belomestny and Schoenmakers (2024), Equation (5.6), extends learned centered innovation controls to general MDP settings. [Published primary record](https://epubs.siam.org/doi/10.1137/22M1526010), [accepted author manuscript](https://www.wias-berlin.de/people/schoenma/Belomestny_SchoenmakersAccept.pdf).

Sabate Vidales, Šiška and Szpruch also use neural martingale controls without requiring an exact neural approximation. [Primary paper](https://arxiv.org/pdf/1810.05094), [publisher PDF](https://www.tandfonline.com/doi/pdf/10.1080/1350486X.2022.2030773). The publisher labels volume 28(4) as 2021, although online publication was 28 March 2022; the author institution's export instead uses 2022. The draft follows the publisher's issue year and records the online date in its note.

Jiang and Li's sequential DR estimator is relevant to model-error discussion, but its off-policy weighting setup is different. [Primary proceedings record and paper](https://proceedings.mlr.press/v48/jiang16.html). Calling the present branch control a new DR estimator would misstate both contributions.

The defensible increment is an engineering construction for **ordered financial-component audits of frozen learned controllers**: local pairing at actual cash under explicitly instrumented signs; attribution through interleaved nonfinancial events; a predictable absolute-amplitude budget that supplies a deterministic negative endpoint for any finite fitted predictor; and independent capped confirmation with disclosed abstention. The budget and one-sided cap are elementary predictable-control and concentration devices, not enough by themselves to support a new general AI algorithm claim. No priority claim for their combination is established by this search.

The substantive empirical question is whether trained continuation models add confirmation power beyond local pairing and simple cost-aware residuals in the authentic simulator. If they tie or lose, the paper must retain that result and claim the valid financial audit/application rather than a demonstrated need for neural learning. A positive certificate for an actual learned policy is useful but alone does not establish estimator superiority. New RL acquisition evidence is not produced by these audit regressions.

## Technical integration cautions

- The financial event target excludes nonfinancial cash flows **and** their floor injections. It is not NAV, training reward or an automatically zero-flat advantage.
- Fresh conditional sign symmetry after all actor/auditor information is known simulator access. Sampling an unknown magnitude law does not make the method arbitrary black-box or assumption free.
- Both queries use the same frozen predictor and common actual presign state. Synthetic branch post-cash features are permitted; a true next-state value mixed with a model branch average is not centered in general.
- Frozen learned actor weights do not imply frozen native causal statistics. Preserve the engine's rolling observation normalization and controller buffers; only the supervised audit scalers are fitted on FIT and fixed thereafter.
- The current law's financial costs require a deterministic executed envelope; unknown finite moments are insufficient. The old oracle method's unbounded-cost guarantee and uniform partial-charge repair do not transfer.
- Fixed-budget confirmation is at the whole-episode level. An upper cap is applied **after** subtracting the learned control. FIT/TUNE selection cannot use CONFIRM endpoints.
- The terminal Tanaka sum already contains the potential difference from the initial ledger. Do not subtract initial debt twice. At a native floor it is the existing realized residual, not another estimator.
- Charge views zero/one reuse physical histories; full charging is not a newly trained-policy result. A strict/guard-restored profile can still receive queued maturities and financial injections.
- Warm-up claims are inherited risks. Their later forgiveness cannot all be credited to actions taken after the thin-cash intervention.

## Required result replacement checklist

Resolve all `PENDING` markers only after full completion and independent verification. Report every case, all six methods, charge zero/one, strict/vulnerable breakdown, cost bounds and abstentions. Include paired differences from the strongest simple baseline and disclose model-prediction overhead, native runtime, support widths, selected budgets/caps and any technical replay. Describe the 40 cases as 20 saved policies crossed with two profiles, not 40 independent trained policies. The 32/32/256 split, 384/96 steps, $10^{-4}F$ cash intervention and 520-limit allocation are frozen design facts, not results.

The new supplement draft contains a full proof and access/implementation assumptions. It is not evidence that the root-owned runner implements every assumption; independent source/arithmetic verification remains necessary. The numerical allowance is engineering padding, not a formal floating-point theorem.
