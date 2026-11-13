# Exploratory latest-placement extension

Written after the complete initial benchmark's negative adaptive-discovery
result and before these extension endpoints are calculated. This is a disclosed
method correction/extension, not an independent confirmation of the initial
search algorithm. All initial files and failures remain available.

## Compiler guarantee and scope

Freeze an active T-step controller, initial ledger and its local state/clock
semantics. Under a stationary IID active-step kernel, no information or cost
while flat, and no early closure, couple the same active disturbances for two
placements within global deadline H. Active ledger paths and final deficit are
identical. Delaying start by s multiplies discounted active costs by gamma^s;
the deadline forgiveness discount stays fixed. The accounting residual improves
pathwise by (1-gamma^s)*discountedLocalCost. Its charged expectation also weakly
improves, because predictable gross has zero mean. This does not compare
different T/controllers or arbitrary policies that use information while idle.

The latest placement, starting H-T, obeys raw charged and booked return
R_H=gamma^(H-T)*R_T pathwise. Thus an independent local certificate scales to
all H>=T and the same alpha interval without new horizon/alpha inference.
Outside compatible horizons the method abstains. Earlier raw returns need not
be pathwise dominated. This compiler is shared by every comparator.

## Fixed full positive-ledger grid and equal active-call budget

Use both Danish TRAIN kernels and every positive registered N ledger fraction
{.02,.05,.1,.25,.5,1,2}: 14 state tasks, with no DP or agent inputs. Each of the
same five methods may choose local active duration T in {1,2,4,8,16}. The
one-action method has only T1; all other policy grammars are exactly the initial
benchmark's, including mandatory initial16, local clock gates and observed
ledger clipping. Degenerate T1 copies are retained rather than outcome-filtered.

Each state/method gets at most 262,144 actual active simulator transitions in
SEARCH. Divide equally in path count across all of its controller-duration
candidates, accounting for their different lengths: n=floor(budget/sum T).
One-action therefore uses 262,144 one-step search paths; myopic5 candidates,
barrier20, structured160 and random160 get the same active-call ceiling. Every
selected state/method policy gets 32,768 fresh confirmation paths of its own T.
All conditions and methods are retained, including abstentions. Shared paths
within phases are allowed; whole paths are IID and phases independent.

## Search correction and independent inference

The initial search evaluated a lower-bound heuristic at its own small search
sample size, often selecting cap32 even when this erased a genuine witness.
Following the already established pre/post-cap-selection approach of Thomas et
al. (AAAI2015, Remark5), use SEARCH mean/variance to predict the bound at the
fixed future confirmation n=32,768. This rule applies equally to every method
and is fixed here before extension calculations. Policy and residual cap are
chosen jointly from all controller-duration/cap candidates; raw cap is chosen
separately on SEARCH for that same selected policy. No confirmation-based
retuning, sample increase, optional stopping or success-subset reporting.

The six one-sided empirical-Bernstein events per selected state/method remain
raw, residual, clipped raw, clipped residual lower bounds, forgiveness lower and
cost upper. Delta=.05/(14*5*6) allocates all 420 events. Deterministic algebraic
scaling supplies every compatible H in {1,4,16,48,96,192}; these repeated calendar
rows are not new trials. Raw and both accounting-residual bounds use the SAME
selected policy and confirmation paths, with the source-checked reusable kernel.
Exact all-allowed-leverage isolated-action values remain the stronger baseline.

For each calendar placement, also retain the paired-sample EARLY and LATE means
derived from the same local rollout aggregates, and the nonnegative residual
improvement (1-gamma^(H-T))*meanCost. These are descriptive means, not extra
confidence events. An exact expected-improvement lower bound uses the mandatory
first action's positive known expected fee. No additional simulations are needed.

Report physical active transition counts separately from logical idle/calendar
slots represented by compilation. No claim of an independent outer-horizon
sample or black-box generality. Maintain a 110-second process-CPU guard and one
numeric thread. Write only extension-prefixed outputs in this directory.
