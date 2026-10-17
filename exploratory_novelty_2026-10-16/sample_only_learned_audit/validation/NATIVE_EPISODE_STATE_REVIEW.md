# Independent source audit of episode state and randomness

16 October 2026. Read-only review of the exact34 native Python files bound by
the primary source freeze, plus the installed Stable-Baselines3 prediction path.
No native import, simulation, actor inference, training or financial endpoint
analysis is performed by this review. Frozen experiment sources are unchanged.

Result: **no concrete hidden-global-state or RNG coupling defect found in the
executed audit path**. This is an executed-path source review, not a formal
verification of every imported numerical library. Its evidence is separate from
the existing warm-state recursive-clone and parent-immutability checks.

## Explicit randomness

`environment.py:623` initializes `self._rng` from the configured native seed;
`:1846` resets that instance generator when a seed is supplied. The unseeded
fallback at`:1849` is entered only if seed construction raises; this audit
supplies valid integer checkpoint seeds. No draws from `_rng` occur elsewhere
in this environment source. The generator state is an environment field, and
the recursive clone fingerprint encodes its bit-generator state.

`environment.py:2344` through the native step path contains no global NumPy,
Python-random or Torch sampling call. Market-condition updates at`:6244` are
deterministic functions of instance performance history and the declared context.
The fresh audit signs and uniform context selection instead come from the
explicit independent per-episode SeedSequence in `native_phase_worker.py`.

Other RNG hits in the34 files are outside this prediction path: native
`main.py`/`evaluation.py` CLI seed setup, forecast-model training seed setup,
ablation generators, the training rollout-buffer permutation at
`policy.py:289`, and the unused `metacontroller.DummyGymEnv` sampling methods.
The audit imports these definitions where necessary but does not invoke their
CLI/training/ablation/dummy-environment execution routines.

## Shared frozen models

The runtime shares loaded actor/battery objects between episodes, so deterministic
prediction also requires review. `native_engine_adapter.py` explicitly requests
`deterministic=True` and puts saved policies in evaluation mode.

The custom Beta actor's `policy.py:241` prediction path reconstructs the action
distribution from the current observation. `BetaDistribution.mode` at`:114`
returns the current distribution mean; `sample`/`rsample` at`:110` is bypassed.
Its alpha/beta/distribution tensors are overwritten by the current network
output, not an accumulated recurrent policy state. The saved actor architecture
is the declared feed-forward MLP, with its native observation/action contract.
No `learn`, optimizer step or rollout-buffer update is invoked.

Installed Stable-Baselines3 2.7.0 `common/policies.py:351` forces evaluation mode
on prediction and`:367` uses `no_grad`; `common/distributions.py:87` chooses
`mode()` when deterministic. Its DQN `dqn.py:250` conditions epsilon-greedy RNG
on `not deterministic`, so the matching frozen battery's deterministic call
short-circuits that random branch. Model construction/loading can initialize
temporary network parameters before restoring saved weights; those setup draws
do not determine the frozen prediction outputs or carry episode outcomes.

## Economic histories and online calibration

Risk and meta rule actions dispatch through
`environment.py:3027` to instance methods. The risk action at`:2950` reads
instance drawdown, risk snapshots and market volatility. The meta action at
`:2968` maintains `_meta_rule_signal_ema` and its target allocation on `self`.
There is no module singleton or hidden policy-training call in this path.

`environment.py:6255` updates `self.enhanced_risk_controller`. That object owns
its configuration, bounded price/generation/portfolio/cash-flow/stress deques,
thresholds and last risk metrics (`risk.py:82–135`,`:398`,`:491–523`). These are
deep-copied as part of the environment rather than recreated from a partial state.
Reward, physical/battery state, normalizer, queues and execution/evidence histories
likewise belong to the cloned environment/configuration graph.

The feasible actor uses precomputed forecast features, not a live globally
updated ANN forecaster. Its online evidence calibration is
`self._forecast_prior_processor` (`environment.py:1959`,`:8506`).
`ConformalForecastPrior.__slots__` (`forecast_prior.py:144`) includes the raw
feature array, hit/error deques, calibrated timestamps, price dictionary, EMA
and last-signal fields. They are instance buffers, and slot-based buffers are
explicitly included in the passed recursive clone fingerprint. The full raw
cached input is read-only in this computation; calibration updates live on
the episode's processor. No forecast `fit`/`learn` or shared model update occurs.

## Module/class state and defaults

The AST inventory `NATIVE_STATE_STATIC_INVENTORY.json` lists all direct
module/class mutable declarations and RNG-like calls. Runtime-relevant lists
and sets are feature-column labels, supported-fee/contract constants, exported
names or environment-render metadata. No economic history singleton is found.
The only explicit `global` declarations are logger configuration
(`logger.py:63`) and lazy TensorFlow-module initialization (`utils.py:34`).
An additional AST scan finds no mutable list/dict/set function default and no
aliased imported random/cache primitive. Risk/forecast modules use instance
histories rather than a module-level memoization cache.

## Approved memory-manager replacement

`UnifiedMemoryManager` (`utils.py:364`) owns a lock, process-RSS history,
cleanup counters/statistics and optional tracked-resource weak references.
Within the native environment it is constructed at`:954` and referenced only
by the destructor's forced cleanup at`:9035`; there is no step/risk/forecast
decision read and no registration of an economic cache, policy or buffer.

The fresh clone manager therefore does not remove economic state. Light cleanup
can clear only registered caches (none registered by this environment), while
heavier cleanup collects garbage, clears unused CUDA allocation caches or a
TensorFlow session. The audited saved policies run on CPU, and forecasts are
already-cached arrays rather than live TensorFlow forecasters. These cleanups
affect resources and time, not economic state or frozen model parameters.
The original warmed environment manager is retained; only the copied environment
receives a fresh same-capacity resource manager/lock.

The review therefore supports treating a complete audit episode as the fixed
context/warm-state deterministic native transform of its independent context
choice and hidden sign vector, conditional on the recorded frozen artifacts.
It makes no claim that the ordinary uninstructed native market has a fair-sign
law or that arbitrary other engine configurations satisfy this state audit.
