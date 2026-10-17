# Retained sample-only software-validation compute

2026-10-16T10:46:03.171773+00:00

Read-only inventory; no statistical financial endpoint or running episode record was opened.

| Retained attempt | Completed warm | Completed audit | Learned actor calls | CPU s | Wall s |
|---|---:|---:|---:|---:|---:|
| native_validation/attempt_1/validation_report.json | 0 | 192 | 384 | 35.203125 | 44.924607 |
| native_validation/warm_attempt_1/validation_report.json | 768 | 96 | 1728 | 33.125000 | 37.043354 |
| native_validation/warm_attempt_2/validation_report.json | 768 | 192 | 1920 | 33.953125 | 37.792922 |
| native_validation/calendar_support_attempt_1/validation_report.json | 768 | 0 | 1536 | 83.734375 | 112.031876 |
| native_validation/forecast_cache_prep_attempt_1/preparation_report.json | 0 | 0 | 0 | 54.718750 | 99.578961 |
| native_validation/runtime_guard_20261016T094704200903Z/provenance_report.json | 0 | 0 | 0 | unrecorded | unrecorded |
| native_validation/runtime_guard_20261016T094811542972Z/provenance_report.json | 0 | 0 | 0 | unrecorded | unrecorded |
| native_validation/runtime_guard_20261016T095042753498Z/provenance_report.json | 0 | 0 | 0 | unrecorded | unrecorded |

Retained completed software work: 2304 warm + 480 audit = 2784 native transitions; 5568 learned actor calls.

The failed first warm attempt completed one384+96 path, then the second384-step warm loop before fingerprint failure. The latter count follows the retained traceback and source control flow; there were zero second-case audit steps. Its per-case time is not independently recorded.

Calendar support validation reused saved audit traces: count768fresh warm steps and ZERO fresh audit steps. Forecast-cache materialization and metadata/arithmetic checks add no native transitions, though some measured CPU/wall is nonzero.

The timing subtotal in JSON sums only records containing timers; it is neither a complete compute ledger nor total elapsed time. Missing timers and unlogged/overwritten arithmetic invocations remain unknown. The planned primary1474560native transitions are separate. No historical learner training is included.

Packet replay: HELD, NOT EXECUTED. Planned768warm+192audit=960native steps and1920actor calls are excluded from executed totals.

Every input report and this inventory source are hashed in the public-safe JSON. No observations, raw panels, private absolute paths, weights or financial profits are copied.
