# Native audit launch and diagnostic transport

The scientific source/design freeze for run_001 is
6152949392bd8de0f494bc21ffb5fd7d9be4a16a6c8866f4eb5595254da523d4.
The first hidden shell background launch returned PID2984 but produced empty
redirected logs and no Python LAUNCH marker or phase data. It exited before the
experiment started. The same unchanged frozen program was then launched through
the managed execution session61413. Its LAUNCH.json records actual PID26404 and
start2026-10-16T09:43:18.249161+00:00. There was no design change, replayed
statistical trajectory or completed observation from the first shell attempt.

Whole-phase rows, compact events, branch features, hashes and completion records
are durably saved under the private execution/run_001/case_XX directories.
Managed stdout chunks are retained beside them as diagnostics. Tool transport
can truncate repetitive warning output; all reported numerical values and query
counts come from complete phase artifacts. Runtime progress contains episode
counts; native margin warnings can also contain descriptive account values.

Before any feasible-action case reached forecast preparation, an independent
helper validated or atomically created the eight deterministic480-row forecast
cache views. The frozen source uses an if-absent CSV write, which could otherwise
race across workers. Pre-materializing identical source-derived bytes avoids
that race without changing a model, input, event map, stream or numerical rule.
The helper, source hashes and all view hashes are retained under
native_validation/forecast_cache_prep_attempt_1. It used99.58wall/54.72CPU seconds,
zero actor calls, zero native transitions and zero random draws. This setup work
is additional to the frozen worker's phase/preparation counters.

The exact source/checkpoint/input preparation is recorded separately from
statistical path work. Earlier deterministic software checks, including the
retained fingerprinting failure and its same-path replay, are excluded from all
FIT/TUNE/CONFIRM estimates. No registered study artifact has been changed.
