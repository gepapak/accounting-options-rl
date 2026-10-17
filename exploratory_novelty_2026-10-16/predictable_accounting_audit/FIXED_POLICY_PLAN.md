# Exploratory development-case confirmation comparison

16 October 2026, fixed before these fresh draws. This is explicitly a known,
selected development case, not another held-out market validation or optimizer
discovery-rate experiment. All previous results, including simple delayed
open-loop success and the completed held-out benchmark, are known.

Use the same DK1 TRAIN law, initial10000 EUR, fee1, gamma.99 and outer terminal
horizon96 as the earlier selected case. Freeze all nine distinct policies from
the matched-baseline family: the original two-stage controller, constant
16/4/1/.25 for96 steps and final16 exposure for1/4/16/48 steps. No new controller
or duration is selected. Compiler scaling is given identically to all methods.

Compare raw, upper-capped raw, capped realized-cost residual, capped mean-cost
residual and uncapped predictable drift on the SAME paths of each frozen policy.
SEARCH8192 chooses each capped estimator's cap from the already fixed grid
{32,128,512,2048,8192,32768,131072}. CONFIRM32768 independent whole paths per
policy is the original benchmark-sized confirmation budget, one sixteenth of
the earlier matched family. Each method has equal full-clock/path budgets;
active durations and extra conditional-oracle arithmetic are disclosed.

Shared global96-path disturbance arrays are allowed; no marginal is selected
using confirmation data. SEARCH seed5130000, CONFIRM5140000. Family error .05
over45 lower bounds (nine policies times five representations), so each bound
has delta .05/45=.01/9, the same per-policy lower-tail error as the preceding
large-budget nine-policy experiment. This new family is separately allocated;
do not combine the several families into a paper-wide confidence claim.

Reuse unchanged predictable_audit.py and the completed held-out runner's
certificate functions. Local active episodes compile to the outer deadline by
gamma^(96-duration). Cap choices are in outer-clock reward units. Every earlier
output and failed result remains untouched. Preserve all policies, caps, bounds,
variances, source/input hashes, actual transitions, queries, runtime and failures.
No sample-size increase, case substitution, cap retuning or follow-up controller
search after confirmation. No RL training or R inputs.
