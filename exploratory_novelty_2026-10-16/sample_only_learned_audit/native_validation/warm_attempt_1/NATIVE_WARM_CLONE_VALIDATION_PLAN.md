# Additional native warm-clone software validation

Fixed before execution, 16 October 2026, after the original two cold-path checks.
This is a separately authorized software check and supplies no new statistical
endpoint. Exactly two profiles are checked: plain MAPPO seed7 / strict, and
feasible-action MAPPO seed7 / vulnerable. Each uses the original context at
start row0, 384 native original-sign burn-in steps, and96 audit steps. Vulnerable
sets only margin enable=False and loss-exit threshold=+infinity (recorded as
`disabled_inf`); zero would activate the most aggressive loss exit. All native
halt flags/history are retained. Both start the audit at .1 times the ORIGINAL
initial trading allocation while preserving the original fixed notional sizing.

The burn-in environment has no audit hooks. Deepcopy covers every economic,
controller, price/forecast/normalizer and RNG buffer. Only the non-economic
UnifiedMemoryManager (with an unpicklable thread lock) is replaced by a fresh
native memory manager of the same capacity. A recursive all-buffer digest,
excluding that approved resource alone, must match immediately before and after
the first clone; the parent snapshot must remain unchanged after the audit.

No endpoint experiment, policy training, monetary tuning, cap selection or
confidence bound is permitted. Deterministic audit signs are all+1 for plain7
and alternating by relative audit time for feasible7. Both terminal queues must
be empty under the native final-window refusal to open unmaturing contracts.
Record native query counts, import/warm-up/clone/per-step CPU/wall timing and
worker resident memory, separately from all future estimator budgets. Earlier
software snapshots remain immutable; this is a new software-validation record.
