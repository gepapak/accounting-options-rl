# Outcome-blind native runtime and power planning

16 October 2026. These estimates use ONLY timings supplied by the engine owner,
without financial aggregates. This validation task ran no episode, checkpoint
inference, fitting or random draw. Root owns the final frozen execution plan.

Current retained `native_validation/warm_attempt_2` software checks use plain7
strict and feasible7 vulnerable, with 384 native warm steps and 96 audit steps.
Full slots/controller/normalizer/RNG-buffer hashes agree across deepcopy, the
cached parent remains unchanged after audit, and both terminal queues are empty.
These establish software compatibility rather than a financial finding or a
guarantee for every policy/context. Warm preparation takes 2.1123/2.4570 seconds,
clone plus buffer hashing .0491/.0541 seconds, and audit steps including actor
inference .005672/.006212 seconds. Worker RSS is about 567/575 MB, with observed
peaks 592/602 MB. Current MAX3 process workers supersedes earlier illustrative
six-worker estimates. Native runtime is process-local and changes cwd: processes,
not threads, are required. Scheduling must not change streams or sample counts.

The current outcome-blind budget is FIT32, TUNE32, CONFIRM256 for all forty
policy/profile cases: 12,800 audit episodes and 1,228,800 audit transitions. Six
methods and two ledger views share these native paths. To avoid holding forty
cases in memory, each case is prepared once for development and again for
confirmation. Eight 384-step warm contexts therefore add 245,760 native
transitions, giving 1,474,560 total native steps. Auxiliary fitting, support
initialization and branch prediction arithmetic are additional measured work.

Using the slower measured case gives about .65 seconds per 96-step clone/episode
plus about 1,600 seconds for 640 warm snapshots: about 2.76 serial hours, or 55.2
minutes at ideal three-worker scaling. A conservative .010 seconds per audit step
plus .06 seconds per clone gives about 4.07 serial hours, or 81.4 ideal
three-worker minutes, including that warmup allowance. Imports, preparation,
contention, serialization and fitting add elapsed time. These are estimates,
not promised performance.

Earlier cold traces reported about .27 seconds construction and .00469/.00754
seconds per audit step, with 32.435 seconds initial imports. Their conservative
1.26 seconds per episode and six-worker figures are superseded by this warm-clone
design and MAX3. The earlier 262,144 transitions PER CASE placeholder was
withdrawn before fitting/confirmation endpoints. RAW pools all 320 episodes only
because all three phases draw the same fixed law. Capped/learned methods confirm
on 256 independent episodes. No job is authorized by this planning note.

The family allocation is delta=.05/520. Its fixed-n empirical-Bernstein
radius is

    sqrt(2*s^2*log(20800)/n) + 7*R*log(20800)/(3*(n-1)),

with R the true deterministic support width and s^2 the ddof1 path variance.
The unavoidable additive term, before variance, is approximately .3683R at n64,
.1827R at n128, .0910R at n256 and .0454R at n512. Thus a small budget can yield
only abstentions even for positive true means. Model fitting is not a substitute
for confirmation power or valid supports. A descriptive wrong-model variance
increase is a possible result, and policy/case/sample sizes cannot change after
seeing it.

Retain all forty cases and disclose the range/variance terms. The strict profile
is guard-restored, not analytically inactive: prepaid warm claims may mature
before margin checks. Do not increase n after a failed confirmation. Root owns
the final budget/guard/process count. The proposed numerical allowance is
max(1e-6,1e-10*(1+Gbar+Cbar+Bctrl)*96), applied to supports and lower endpoints.
It is an engineering floating-point allowance, not a formal rounding proof.
