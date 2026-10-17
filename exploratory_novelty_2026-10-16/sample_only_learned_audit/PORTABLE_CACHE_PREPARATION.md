# Portable forecast-cache preparation

Post-run software validation is complete: all eight packet views passed in
`native_validation/portable_cache_packet_attempt_1`, with zero native steps or
actors. The exact two historical packet fixtures subsequently passed in
`native_validation/clean_bundle_replay_attempt_4`. See
`native_validation/PACKET_PORTABILITY_VALIDATION.md` for failed attempts,
resource-guard exceptions, complete work counts and the existing-dependency
scope. No full statistical replication or clean installation was performed.

`prepare_packet_forecast_cache.py` is a new, unfrozen engineering wrapper. The
historical `prematerialize_native_forecast_views.py` and all frozen numerical
sources remain byte-for-byte intact. The original prepared workflow required a
release before loading native modules or preparing packet views. Execute only after the coordinator releases
post-run work. Its default invocation reports the hold without native imports.

The wrapper accepts explicit `--packet-root`, `--audit-root`, `--engine-root`,
`--source-freeze`, `--input-manifest`, `--output-root`, `--cache-root` and
`--provenance-output`. For an unpacked complete v1 packet at arbitrary `PACKET`,
the roots are:

- packet: `PACKET`;
- engine: `PACKET/Prototype5`;
- auditor: `PACKET/github/accounting-options-rl/exploratory_novelty_2026-10-16/sample_only_learned_audit`;
- freeze: `PACKET/provenance/SOURCE_FREEZE.json`;
- input manifest: `PACKET/BUNDLE_MANIFEST.json`;
- output: either the exact auditor root for the compatible packet layout, or a
  separate writable runtime directory for an external replay wrapper;
- cache: `OUTPUT/runtime_cache/forecast_views`;
- provenance: a fresh, exclusive directory for this preparation attempt.

All paths must be supplied with `--execute`. The complete packet is private;
do not upload it or mistake this example for a public-data release. Original
metadata paths remain in its byte-exact artifacts.

Before loading the frozen forecast reader, the helper checks the packet-bound
source freeze, all34 native sources, all11 frozen auditor sources, all60 paired
checkpoint/config files, and all10 exact runtime CSV sizes/hashes. Model files
are hashed only; they are not deserialized. No environment, model, actor,
forecaster network or audit RNG is instantiated. The only native engine source
module loaded is the frozen `forecast_prior.py` reader; the frozen adapter is
loaded for its exact cache-key function with in-memory `ENGINE`/`HERE` roots.
It never calls `NativeRuntime`, `NativeScenario` or `NativeEpisode`.

The reader loads the existing original/v2 precomputed forecast CSVs. For each
market source it reproduces `np.linspace(0, source_rows-480, 4, dtype=int)`,
uses the native float32 feature order/defaults, and serializes each480-row view
with the same pandas `DataFrame(...).to_csv(index=False)` operation. The key
includes the explicit packet forecast-source path, start row and480-row horizon,
matching the unchanged adapter. Cache keys consequently change when a packet
moves; source data and forecasts do not.

With `--output-root` exactly equal to `--audit-root`, the helper permits only
new cache writes under `AUDIT/runtime_cache/forecast_views`. Provenance must
still be a fresh directory outside both frozen source trees. Frozen source
hashes are checked before and after. This compatible packet mode lets the
unchanged adapter infer `PAPER1`/`ENGINE` from the existing packet layout and use
the prepared views at its ordinary `HERE`; a later fresh run identifier can use
the frozen runner without an extra Windows-spawn path wrapper. No such full
rerun has been executed. Other output directories nested inside the auditor
tree are rejected.

Each target is either verified against the complete expected bytes or published
from an exclusively created partial file via atomic, non-replacing publication
(Windows rename; POSIX hard link).
Existing partial or wrong targets are preserved and cause failure. A filesystem
that cannot perform its native non-replacing publication fails rather than fall
back to an overwrite. The provenance records each view's hash, source hash,
action, exact roots, dependency versions, CPU/wall work and zero native-query
counts. All source/model/input hashes are checked again before PASS. Attempt
directories are exclusive; failures and partial files remain reviewable.

During preparation, normal imports are restricted to installed Python roots;
native modules are loaded only from the explicit packet files. An I/O audit
guard permits packet/dependency reads and cache/provenance writes, and rejects
reads from an original Prototype5 tree outside the packet. No original-native
fallback or write is used. This is a source/input-preparation guard for the
Python operations this helper performs, not an operating-system sandbox.

An independent full run requires the compatible packet layout above, or an
explicit external execution wrapper that overrides frozen path roots in memory.
For external-output mode, that wrapper must set `HERE=OUTPUT` for these cache
files to be used. The unchanged primary runner is compatible with the mirrored
packet layout; it does not automatically discover an arbitrary external output
directory. No clean dependency installation, full40-case replication or public
weight release is claimed.

The existing held `replay_private_bundle.py` separately replays exactly two
historical software fixtures after release. Its runtime path must likewise
match a cache preparation output if the helper is used beforehand. These are
software checks, not additional scientific observations.
