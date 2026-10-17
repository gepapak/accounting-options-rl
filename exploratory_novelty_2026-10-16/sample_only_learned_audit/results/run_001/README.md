# Public arithmetic reproduction of the completed native audit

Run with Python 3.10 or newer, using only its standard library:

    python reproduce_public_native_audit.py .

Optionally add `--write-report` to exclusively create `REPRODUCTION.json`. The
program first checks every public artifact SHA256, then independently recomputes
all 520 one-sided empirical-Bernstein limits and decisions. No NumPy, sklearn,
native simulator, raw market panel, actor checkpoint or fitted model is required.

`paths.jsonl.gz` contains all 12,800 whole-episode summaries from the three fixed
phases and 40 cases. RAW pools 320 same-law paths per case; selected/capped methods
confirm on 256 paths, after 32 FIT and 32 TUNE paths. Cost limits pool 320 paths.
Both charge views share native histories. The 480 charged and 40 cost limits use
the separate family allocation delta=.05/520; other experiment families are not
included. Sequential absolute-amplitude controls are reconstructed from ordered
coin/predictor-difference vectors, without a fitted model. Deterministic scalar
supports, numerical allowances, choices and every original confidence number are
in the accompanying JSON files. Booked harm uses the larger of the cost lower
limit and the source-proved unavoidable access-fee lower bound.

This reproduces confidence arithmetic conditional on the exported records and
the declared instrumented fair-sign law. It does not independently validate the
native engine, actor interfaces, source-derived supports or sign intervention.
The result concerns the forward financial-event component for twenty frozen
policies under eight fixed contexts and two guard profiles, not native reward,
complete fund/lifecycle return, future markets or retrained full-booking policies.
The strict profile is guard-restored and may retain prepaid claims; inactivity
or zero forgiveness is not assumed. All abstentions and full-booking views remain.

No actor observations, native raw panels, absolute private paths, weights,
pickles or private feature rows are included. `provenance.json` supplies source,
checkpoint, data/context and model SHA256 identities without supplying access or
redistribution rights. Full engine reproduction remains a separate dependency
and access requirement. Original internal results are preserved unchanged.
