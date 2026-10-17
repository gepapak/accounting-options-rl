# Proposed final private v2 review addendum — not built

This is a packaging plan, not authorization to execute a replay or build another
archive while the primary experiment runs. Root's post-run release is required.
No frozen source, scientific phase artifact, input, result, existing manifest or
v1 bundle is altered. Final addendum hashes are collected only after all intended
static/software corrections and complete numerical verification are finished.

## Base and transport contract

Base: `review_bundle_private/native_review_complete_v1.zip`,155files,
53,564,447bytes, SHA-256
`d96cf47f5d6746cf760608293bfbe5e62a5720f2b98c30702a89debf5f6b8299`.
Its exact34native modules,20paired investor/battery checkpoints,20configs,
10runtime CSVs and5optionalANN files are retained byte-for-byte. The original
experiment and previous source-only/complete v1 packets remain intact.

Create a NEW `review_bundle_private/native_review_v2_addendum/` directory and
new ZIP, plus an exclusive completion manifest. Every new archive member uses
the distinct top-level prefix `native_review_v2_addendum/`, including updated
READMEs/reviews under its `documents/` subtree. The addendum can be unpacked
beside a fresh copy of v1 with zero base-path collisions.
If a combined full v2 tree/archive is desired, create it separately from a verified
v1 copy plus this addendum, never by modifying preserved v1. No market-input or
RL-weight reserialization is needed.

The new addendum README must state which complete verified results/replay were
available when built, list every file's category/bytes/SHA, bind both primary and
auxiliary source freezes and original v1 SHA, and state all omitted private
material. New scripts are labeled as post-freeze verification/packaging tools,
not retroactively claimed to be part of the primary or auxiliary source freeze.

## Added source/document list

Copy final exact bytes (no rewritten originals) of:

- New unfrozen `prepare_private_review_v2_addendum.py`,
  `PRIVATE_V2_ADDENDUM_BUILDER.md`, its literal/static checker and retained PASS
  record. Do not execute the builder until all completion/replay gates and root's
  post-run release hold. Its default is held and performs no packaging/endpoint
  or native work. Manifest categories distinguish all copied sources and records.

- Frozen auxiliary definition:
  `EXPLORATORY_NEW_CONTRACT_PLAN.md`, `NEW_CONTRACT_INDEPENDENT_REVIEW.md`,
  `analyze_new_contract_component.py`, `NEW_CONTRACT_SOURCE_FREEZE_run_001.json`,
  and the3unchanged files under `new_contract_source_snapshot_run_001/`.
- Independent arithmetic verification:
  `validation/verify_native_results.py`,
  `validation/verify_new_contract_component.py`,
  `validation/verify_sample_core.py`, their applicable complete literal/static
  test records and `validation/README.md`. Do not substitute verifier source
  hashes for the original experiment/source-freeze hashes.
- Public arithmetic exports/reproducers:
  `public_export.py`, `reproduce_public_native_audit.py`,
  `public_export_new_contract.py`, `reproduce_public_new_contract.py`.
  Keep the primary520 and auxiliary240 public result trees separate.
- Packet-only software replay and dependency/provenance helpers:
  `replay_private_bundle.py`, `CLEAN_BUNDLE_REPLAY_PLAN.md`,
  `prematerialize_native_forecast_views.py`,
  `validate_native_support.py`, `validate_runtime_provenance.py`,
  `audit_native_state_sources.py`, `validation/NATIVE_STATE_STATIC_INVENTORY.json`,
  `validation/NATIVE_EPISODE_STATE_REVIEW.md`,
  `validation/audit_native_chronology.py`,
  `validation/NATIVE_CHRONOLOGY_STATIC.json`,
  `validation/NATIVE_CHRONOLOGY_REVIEW.md`,
  `POST_RUN_REVIEW_CHECKLIST.md` and this addendum plan.
- Final `INTERPRETATION_NOTE.md`,
  `validation/NO_NEW_CLAIMS_REFERENCE_COROLLARY_REVIEW.md`,
  latest software-compute JSON/Markdown inventory (explicit as-of timestamp),
  `post_completion_native_tables.py` and its complete generated reports/fragments.
  Keep the conditional corollary's unproved native-reference executability clear.
- Separate post-gate descriptive control layer:
  `post_completion_control_diagnostics.py`,
  `POST_COMPLETION_CONTROL_DIAGNOSTIC_PLAN.md`, its retained literal/software
  check records, and exactly
  `control_diagnostics/<run>/{RESULTS.json,case_diagnostics.csv,README.md,COMPLETION.json}`
  after its own source/plan/private/public/report-bound completion gate passes.
  Its240 fixed target/case/model records use the original256CONFIRM paths.
  Uncapped paired/control and same-learned-cap comparisons are descriptive:
  no new CI/family, query, fit, retuning or superiority claim. Preserve the
  original eighteen-file report and its whitelist. Package this layer under
  `descriptive_control_diagnostics/`, separately from the original report.
  Include `native_validation/UNRESTRICTED_NEGATIVE_TAIL_BOUNDARY.md` as an
  established-theory scope note, not a new audit theorem. The previous builder
  source/SHA is retained under `validation/source_versions/`.
- Updated packaging/access notes: final `README.md`,
  `PUBLIC_REPRODUCIBILITY.md`, `REPRODUCTION_ACCESS_PLAN.md`,
  `public_reproducibility_manifest.json`,
  `build_public_reproducibility_manifest.py`,
  `prepare_private_review_bundle.py` and
  `prepare_complete_private_review_bundle.py` as historical builders. Clearly
  distinguish superseded source-only-stage notes from complete-packet contents.
- New unfrozen portable preparation wrapper `prepare_packet_forecast_cache.py`,
  `PORTABLE_CACHE_PREPARATION.md` and its static/literal software-check record.
  Its execution remains held until post-run release. It accepts explicit packet,
  auditor, engine, source-freeze, input-manifest, output/cache and provenance
  roots; no run identifier is hardcoded. It hashes sources/models/runtime inputs,
  loads only the exact forecast reader and adapter key function, and atomically
  prepares/verifies all8 views with zero actors/environments/audit draws/steps.
  Include any later released execution report only after it exists. The original
  historical preparation helper remains unchanged and binds `run_001`. The new
  helper supports OUTPUT==AUDIT for the existing compatible mirrored packet
  layout, writing only its new runtime-cache subtree; the unchanged primary
  runner can infer packet paths in that layout. External OUTPUT requires an
  explicit in-memory HERE wrapper. Neither mode has executed a full replication.
  Include `validation/check_portable_cache_wrapper.py` and its final PASS record;
  retain the earlier attempt snapshots/failure note as engineering provenance.

## Completed evidence to add only after it exists

- Primary PASS520 report(s), auxiliary PASS240 report, their exact verifier
  source hashes and the complete public-arithmetic reproduction reports.
- The completed primary and auxiliary public eight-file result bundles, including
  their completion manifests and numeric path-summary/choice/support records.
  These contain no actor observation rows, raw market panels, weights, fitted
  nuisance-model pickles or absolute private paths.
- The released packet-only2-path replay's source/plan snapshot, PASS/FAIL report,
  loaded-module/file-access provenance and numerical-array comparison summary.
  A PASS must explicitly retain the existing-dependency-environment limitation.
- The two exact prior `warm_attempt_2` software reference NPZ traces and their
  original validation report, **private only**, so replay comparison can be
  supplied via an explicit packet-local `--reference-dir` instead of depending
  on the original repository tree. They are software fixtures, not extra
  statistical observations. These recorded investor observations are prohibited
  from the public arithmetic exports but can remain in this local private packet.
- The deterministic forecast-cache-preparation and saved-trace calendar-support
  PASS reports, with their bound helper/source snapshot hashes and0query scope.
- Original managed-startup/transport note and any actual technical-failure/replay
  notes, as provenance; never describe a zero-query detached startup as an
  independent run or erase preserved failed attempts.

Do not copy the ongoing private scientific execution directories, fitted audit
critic/model bundles, full private actor-feature datasets, or new scientific
endpoint samples merely for convenience. Hashes and final public arithmetic
records suffice for this addendum's intended verification layer. Any later
expanded private archive requires an explicit scope decision and a fresh manifest.

## Verification and release boundary

The passed two-path packet replay is attempt4, with the exact explicitly named
`replay_private_bundle_attempt4.py` source/snapshot. Include its PASS record,
the zero-native eight-view packet cache preparation PASS, and the consolidated
`native_validation/PACKET_PORTABILITY_VALIDATION.md`. Preserve earlier failed
guard attempts and count their actual native work. This packaging addendum is
separate from the unexecuted `native_joint_fault_v2` scientific proposal; it
does not include that proposal's future models, phases or endpoints.

Before sealing, compare every copied byte/hash to its source, verify preserved
v1 integrity and no collision with its frozen auditor/native/model/input files,
stream the ZIP and rehash every decompressed member. Write the final completion
manifest only after those checks. Retain partial/error artifacts rather than
rewriting another attempt. Record exact compressed/uncompressed sizes and work;
use low compression or store already-compressed ZIP/Keras artifacts.

This remains a LOCAL PRIVATE review packet. No Git push, upload or public weight
release is performed. Native Energinet/ERA5/author-derived provenance and explicit
artifact-license/attribution tasks remain in the release notes. Original configs
and metadata retain user/project paths for exact byte identity and need a separate
anonymized derivative before anonymous delivery. Neither packaging nor two-path
replay is a clean dependency-install validation, a full new40-case replication,
ordinary-market profitability evidence or a stronger AI-novelty result by itself.
