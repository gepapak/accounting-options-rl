# V2 public arithmetic pipeline — source-only preparation

Status: implemented and checked on synthetic constant records only. No v2 native endpoint, actor, model or frozen experiment has been accessed or created by these tools. Original run001 bundles remain unchanged. This document does not authorize a native launch or public release.

Owned new scripts:

- `reproduce_public_joint_fault.py`: Python standard library only; `verify_payload`, `verify_artifacts`, `verify_directory` and a standalone CLI.
- `public_export_joint_fault.py`: default held; `--execute` requires complete private verification and provenance gates, then exclusively creates `results/run_001/` under this v2 directory.
- `check_public_joint_fault_synthetic.py`: closed-form zero-variance toy arithmetic plus tampering/privacy/missing-gate tests; it never imports native/numerical/model modules.

## Final pre-statistical schema

The schema is `sample_only_joint_fault_public_v2`. All20cases/policies and6400paths are retained (32FIT+32TUNE+256CONFIRM per case). IDs are policy0..19, with original vulnerable lineage `source_run001_case_id=2*policy_index+1`; original paths are not reused.

The family has340limits: 20cases ×(10alpha0 methods +6alpha1 methods +1cost). There are320 frozen charged choices/limits. Alpha0 has the original six methods plus `cash_capped_raw`, `cash_ridge_control`, `cash_mlp_control` and `cash_local_control`; alpha1 has only the original six. All use delta=.05/340. RAW/cost use320same-law paths; capped/selected methods use256CONFIRM paths.

Public paths whitelist the declared new-component G/C/realized/paired summaries, direct excluded-old summaries/counts, all ordered maturity coins and, on CONFIRM, five vectors: `delta_ridge`, `delta_mlp`, `delta_cash_ridge`, `delta_cash_mlp`, `delta_cash_local`. Cash learned/local controls subtract their budgeted martingale from RAW rather than PAIR. Their support is `[-K-B,min(cap,Gbar+B)]`; cash RAW uses `[-K,min(cap,Gbar)]`. The numerical allowance deliberately retains original G+C+B access. Four independently fitted regressors are identified separately through `models/` and `models_raw/` manifests; the local term is analytic and adds no fit.

## Export gates and outputs

Before export, require original-style completed phase artifacts, final completion/result, SOURCE_FREEZE and SELECTION_FREEZE, all320 frozen choices, both model manifests and independent private `PASS` with `bounds_checked=340` and nuisance prediction rechecking. Current/snapshot v2 sources, exact original source dependencies, selected model manifests, phase artifact bytes, result and completion hashes are checked. The source freeze must include both public scripts and `validation/verify_joint_fault_results.py`. Confirmed selection fields are `model_manifest_sha256`, `raw_model_manifest_sha256`, `selection_sha256`, `fit_input_sha256` and `raw_fit_input_sha256`.

Output whitelist: `cases_and_bounds.json`, `selection.json`, `results.json`, `provenance.json`, `EXPORT_ARITHMETIC_CHECK.json`, `paths.jsonl.gz`, `reproduce_public_joint_fault.py`, `README.md`, then the sole `MANIFEST.json` completion seal. The exporter reopens/hashes/reproduces persisted bytes before sealing and rechecks private/source bindings. Failures preserve partial output without a completion manifest; outputs are never overwritten. Public JSON excludes absolute private paths and whitelists numeric computation metadata. It copies no observations, raw panels, cash traces, checkpoints, pickles or fitted models, and assigns no redistribution rights.

Later, after coordinator release and full gates:

    python public_export_joint_fault.py --run-id run_001 --execute
    python results/run_001/reproduce_public_joint_fault.py results/run_001 --write-report

Reproduction is conditional arithmetic on exported summaries, source supports and declared sign law. It is not independent native-engine replay, a clean-install/full-pipeline test, anonymous-packet certification, simulator-authenticity proof, whole-fund guarantee, acquisition result or AI-method superiority claim. A separate340-limit95% family does not imply paper-wide95% coverage.

## Checks completed without scientific endpoints

`validation/PUBLIC_ARITHMETIC_SYNTHETIC_CHECK_20261016T130733Z.json` records PASS on6400 constant toy paths and all340 arithmetic limits against independently constructed closed-form zero-variance bounds. It checks the budget saturation/cancellation identities, final10/6 method map, RAW-versus-PAIR/cash base distinctions, and rejects nine tampering/privacy/missing-gate cases. The missing-gate check uses an isolated temporary directory and forbids any outcome read. No NumPy, torch, sklearn or native adapter was imported. These are software tests, not new scientific confidence events or a native result. Real completed-record export/replay remains unexecuted. This final check follows a header-only correction from PASS280 to PASS340; the earlier synthetic report is preserved.
