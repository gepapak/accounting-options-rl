# Final exploratory full-clock extension

Written after both earlier empirical benchmarks found zero new certified
opportunities beyond the all-leverage isolated-action reference, and before
this extension's endpoint calculations. This is the last empirical extension
in this task. Prior scripts, outputs, uncertainty failures and negative search
comparisons remain untouched. It is not a new preregistration or an acquisition
experiment.

The substantive limitation of the short-window extension is that no active
controller could trade for more than 16 periods. Replace its duration set with
the **entire frozen N clock grid {1,4,16,48,96,192}**, retaining all 14 positive
ledger states, both Danish TRAIN kernels, all five comparator methods and the
same shared latest-placement compiler. No state or method is removed after a
failure. One-action retains T1; all feedback methods receive every duration.

Keep at most 262,144 SEARCH active transitions per state/method, allocated by
equal paths across controller-duration candidates with lengths accounted for.
Keep 32,768 independent confirmation paths per selected controller, all prior
policy grammar/cap grid, SEARCH-predicted confirmation score and exact
all-allowed-leverage isolated-action reference. Longer controller libraries
therefore have fewer search paths per candidate; this budget limitation will be
reported, not concealed by more sampling.

Use fresh deterministic phase seeds fixed now: SEARCH 2101000+state_id,
CONFIRMATION 2201000+state_id, random controller library
2301000+10*state_id+method_id. Source-validation seed is 2401000. Phase
independence remains mandatory; shared CRN within phases is allowed. No Study R,
DP or trained-agent output is read.

This is a separate 420-event confidence family, delta=.05/(14*5*6), covering the
same six one-sided bounds on all selected policies. No 95% simultaneous-coverage
claim is made across this and earlier exploratory families. Calendar/alpha
scaling remains algebraic for each fixed policy, with incompatible H<T recorded
as abstention. No horizon-selection based on confirmation and no extra trials
are manufactured from calendar rows.

Replay actual frozen environments for T4 and T192 late placements at H192.
Record all candidates, selected policies, exact references, raw/residual/clipped
bound comparisons, misses, controls and compute in separate `full_clock_*`
files. Implement a per-rollout process-CPU guard, allowing time to save all
completed rows and a partial status before 110 CPU seconds. Include actual
vectorized transition work, even if a current method is unfinished. No extra
confirmation paths, reranking or follow-up state extension will be launched to
obtain success. If there is still no certified isolated-action miss, report that
result and stop empirical extensions.
