# Study R run log

## Before hashing

15 October 2026: new design after the renewed request for deeper research and
novelty. The earlier failed Monte Carlo surrogate/two-fee design remains intact.
The new exact empirical one-step diagnostic and known retrospective failures
were disclosed before selecting the same-dose cross-cost test. No full R
treatment/control training occurred before registration.

Software validation and a single 8192-step TRAIN-only booked-control smoke are
performed before freeze; their artifacts are excluded from all full R results.
The protocol, exact predictions, analysis, environment and runner are reviewed
before hashing. Original protected studies are read-only.

The first software-smoke attempt exposed SB3 replacing the decimal-containing
model filename suffix when an extension was omitted. It trained only the stated
8192-step booked control and wrote model artifacts, but no completed job summary.
The save path was corrected to explicitly end in `.zip`, and the same short
control/seed is retried before freeze. Failed-attempt artifacts are retained
under `development_smoke/` and are excluded. No treatment or full-run outcome
informed this software fix or changed the protocol.

Before freeze, the 12096-transition validation passed with zero reward difference;
synthetic supported/failed primary decisions and rejection of partial cohorts
passed. Runtime versions are in `validation_r.json`. The retried booked-control
software smoke passed all completion/artifact/counter checks in 24.6 seconds;
its greedy low-equity leverage was 16 after only 8192 steps. This undertrained
development outcome is disclosed, excluded and was not used to alter the design.

## After hashing

### Detached full launch

UTC 2026-10-15T14:43:51.3175218Z: registration verified before full training; PID 22804, 6 workers, all 60 planned jobs, stdout run_full_r.log, stderr run_full_r.err. The unchanged runner automatically aggregates only after all jobs complete. Initial runtime estimate: several hours, to be refined from completed jobs. No full-run outcome inspected before launch.

### Detached launch outside sandbox after process-lifetime failure

UTC 2026-10-15T14:45:50.1685591Z: initial sandbox PID 22804 exited without any stdout, stderr or completed full job. Registered inputs were reverified unchanged. Relaunched outside the sandbox, PID 22496, 6 workers; new logs run_attempt2_r.log and run_attempt2_r.err preserve the initial empty logs. No training outcomes informed this retry. All 60 planned jobs and the automatic completion analysis remain unchanged.

### First completed wave: runtime-only handoff

UTC 2026-10-15T15:23:46+00:00: 6 of 60 complete job summaries; observed runtime range1629–1647seconds, error log empty. Extrapolated finish approximately21:20–22:00 Copenhagen time15October2026, provisional. Only completion counts and runtime fields were used; no partial hypothesis analysis or design change. Parent PID22496 and6workers continue.

### Completion and unchanged registered aggregation

UTC 2026-10-15T17:56:07+00:00: all 60 jobs complete; registered aggregator executed unchanged. R1 NOT SUPPORTED, seed sign p=0.81250. No selective stopping or reruns based on outcomes.
