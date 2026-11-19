# Native identification status: completed kernels, held policy screen

The concrete gain is a checked transfer counterexample and a conservative
ordered-cash inverse. It is not yet evidence of a new AI algorithm or a
nontrivial advantage on a learned policy's native path. The cheaper saved-cohort
audit should decide whether an additional native screen has value.

## Completed original-source hand gate

FROZEN_HAND_PROTOCOL_V1.json SHA256 is
`521d5f0e1514814af58f62a147ae00bd0775738aef86006cba410acc921d4024`.
The root authorized exactly that gate. Its preserved
`runs/hand_v1/RESULTS.json` reports **54 attempted/returned top calls and 462
attempted/returned helper calls**, 18 hand proxies, **zero full Env instances,
resets, Env.step calls, learned-model calls, fits or market reads**. No later
native calls have been made by this workstream.

The six declared hand cases all completed with source parity and containment.
They used unchanged original public-engine methods. A separate AST receipt
finds the 17 tested methods identical in public and Prototype5 sources. This
does not establish whole-engine or frozen-policy trajectory parity.

| Fixed hand case | Real/shadow ending cash | Canonical cash-increment TIR | True booked | Conservative booked interval |
|---|---:|---:|---:|---:|
| No cost | 101 / 99 | 1 | 1 | [0.75,1.25] |
| Fee before floor | 1 / 0 | 1 | -9 | [-9.25,-8.75] |
| Losing real branch | 0 / 8 | -5 | -5 | [-5.25,-4.75] |
| Post-funding both censored | 0 / 0 | -10 | -9 | [-20.25,0.25] |
| Pre-fee plus post-funding | 0 / 0 | -12 | -11 | [-14.25,-9.75] |
| Two deliveries | 2 / 0 | 2 | -8 | [-8.25,-7.75] |

The decisive fee example has starting cash 0, own opening cost 10, then
settlement gross +1 in the real branch and -1 in the shadow. The native fee
write floors cash first; settlement then adds/floors. Reported increments are
(1,0), so applying the canonical one-final-floor TIR formula gives +1 while
booked G-C is -9. The ordered inverse contains -9 and excludes +1. This is a
counterexample to transferring the canonical observation model to these native
cash writes. It does not invalidate the canonical theorem under its stated
one-floor assumption. A real-only cost-aware inverse also solves this example;
the correction alone is not evidence that a twin query adds value.

The source binding is explicit: public `engine/environment.py:4329` floors the
opening-cost write, `:4360` floors access fees, and `:4543` independently floors
the aggregate settlement write. MWh payoff subtraction is `:4402`; accepted
MWh use and ordered gross accumulation are `:4513` and `:4519`. Funding is
another floor at `:2725`. The scalar investor reward is subsequently shaped
(`:5230`, `:5418`), so this inverse targets financial cash/own booked receipts,
not that shaped scalar. Extreme funding rates in some hand states test source
semantics only; they establish no realistic learned-policy reachability.

## Exact characterization and corrected comparator

`KNOWN_SUM_IDENTIFICATION_V2.md` proves the exact real-valued result. An ordered
known suffix reduces to f(z)=max(z+A,T). Given a known paired gross sum K,
intersect the inverse of f(z) with that of f(K-z). If either branch exceeds T,
the real gross is unique. If both equal T, its sharp set is
`[K-(T-A),T-A]`, intersected with the same public support. Strictly positive
width is unavoidable ambiguity; equality at K=2(T-A) is a singleton.

At an isolated settlement event, A is observed pre-cash, T=0. A K=0 pair is
the existing local zero-cost TIR baseline. With a general known K, both
branches can remain censored. Real-only cash already identifies any real
uncensored branch and must get the same own quantities, entry prices and
public support. The paired method can claim only its additional narrowing.

For public, fixed delivery centers R_d, linear MWh settlement reflection gives
K=2 sum_i q_i(R_d-P_i). This permits different accepted entries for a delivery,
signed quantities and net-zero portfolios. `known_sum_v2.py` supplies a public
rounding-discrepancy bound and conservative cash inverse. `known_sum_v3.py`
improves the baseline support by aggregating quantities per common delivery;
it preserves all V2 files and outputs. This matters particularly when net Q=0:
the ideal gross can already be known from the entries without any price query.

The fresh `KNOWN_SUM_PURE_CHECKS_V3.json` preserves the 96-case grid, tightens
24 support intervals, and still finds 26 strict paired improvements over the
equally informed real-only comparator. Twelve exact both-floor set checks
also pass. The V2 count was relative to a looser per-contract support; it is
not silently relabeled. These are own arithmetic cases, not frequencies from
native policies. The intervals are conservative binary64 enclosures under the
declared operand bound, not sharp discrete IEEE identified sets or statistical
confidence statements. All-censored ambiguity remains explicit.

## Minimal runnable policy adapter is prepared and HELD

`policy_event_observer_v1.py:40` is the allowlisted estimator interface.
The original actor receives its original observation throughout. A read-only
CPython call/return profiler (`:137`) captures each actual due-settlement call
after actions/costs have committed. It captures only prices already paid by the
original settler; a trusted shadow service uses them privately. The estimator
receives pre/post cash, reflected-shadow post-cash, accepted own fills, public
centers and public support. It receives no S, G, booked reward or moment oracle.
This is logical function-level separation, not OS process isolation.

For each due event, the unchanged source settler runs on a minimal proxy once
unflipped and once reflected (`:116`, `:167`). Exact unflipped cash, queue,
gross and per-asset parity must pass before interpretation. The actual Env is
never cloned, stepped or modified by the shadow. No downstream funding/margin
response is asserted invariant: this is explicitly an event-local query.
Actual observed own step costs translate the recovered real G into actual
financial G-C. They are returned nominal source receipts, not cumulative-counter
differences or gold-derived cash credits. Known no-due steps have G=0.

`run_policy_event_gate_v1.py` can execute this bounded screen only after a new
exact root authorization. `--prepare` has run successfully using source and
metadata only. **--run has not run.** The frozen protocol SHA256 is
`58ee42dc9192661e34d6eda542998c6fa3c1035f12ba731a15bdb24a0185adf2`.
The old hand authorization cannot satisfy its distinct authorization status.

The proposed screen is plain PPO seed4102, deterministic, already-used2022
context15, no training. A fresh reference and a fresh observed episode require
2016 original native steps, 336 selected-PPO predictions, at most 166 due
events/332 additional original settler calls, and at most 10000 extra original
helper calls. Two isolated seed7 native streams must produce exact matching
native-row digests, actor traces, financial totals and RNG receipts before any
audit outcome is interpreted. Failure is preserved and stops the gate.

Mandatory references are real-only cost accounting with grouped-delivery
support; the directly exposed source ledger; paid-price reconstruction; and
the same own-contract reconstruction from the existing release-limited price
reader (`policy_event_observer_v1.py:253`). The latter permits at most 166
additional released-row queries after the episode. Its delivery+7 release
rule is a declared simulator convention, explicitly not a real-world publishing
claim (`native_joint_fault_fullcash_gate_001/run_gate_v3.py:84`). Source maturity
is delivery+6, hence a one-native-step latency difference in this pinned case.
The immediate source ledger is already available in the native interface.

The observer's six own stand-in checks pass, including source-shaped shadow
parity, a rejected forbidden gold field, real-only retention when reflection
is unsupported, a release-price baseline and a both-censored interval.
`POLICY_EVENT_OBSERVER_PURE_CHECKS_V2.json` records zero external original-body
calls, native Env instances, model calls or market reads. Stand-ins do not
establish actual native runtime parity; that remains a held gate.

## Contribution boundary

The existing engine exposes settlement G/S, and the reused PPO was trained and
observed with ledger features. Calling this audit interface restricted does not
turn the policy into a ledger-free learner or establish natural monetary reward
censorship in that engine. A real-only inverse may already recover every event
on a solvent saved path, leaving zero empirical value for an extra query.
Source hand success cannot substitute for measuring that baseline.

This work supports a more accurate transfer statement: canonical TIR can fail
under an intermediate cost floor; source-order accounting with explicit
receipts repairs the observation model and retains nonidentification where
floors destroy information. General known-sum reflection and RN enclosures
make a concrete adapter feasible. It does not establish new AI verification,
automatic source-to-certificate synthesis, learned-policy improvement, broad
generalization or an EAAI-level AI method contribution by itself.
