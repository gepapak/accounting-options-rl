# Native chronology and information-set review

This is a source/input-metadata review, not an episode or endpoint analysis.
`NATIVE_CHRONOLOGY_STATIC.json` records exact input/source hashes, CSV timestamp
extents and ANN metadata; `audit_native_chronology.py` reads these using only the
standard library. No native module, actor, environment or audit RNG was invoked.
Frozen plans and numerical sources were not changed.

## History chronology and price normalization

The native loader `evaluation.py:1549` selects newest numbered history files
until the requested trailing history length is available, sorts timestamps,
deduplicates and trims to365×144=52,560 rows. For the frozen inputs, history019
has26,208 rows from2024-01-01 00:00 through2024-06-30 23:50; history020 has26,496
rows from2024-07-01 00:00 through2024-12-31 23:50. The365-day trim consequently
starts2024-01-02, ending2024-12-31. It includes no2025 context row.

`evaluation.py:1598–1606` computes historical price running moments and physical
p95 scales from that history. `native_engine_adapter.py:282–292` then advances
the price running state through `frame.iloc[:start_row]`, which excludes the
selected window itself. `environment.py:114–153` computes each normalized price
using the preceding state and only then updates moments with that row. The
native constructor precomputes the entire480-row normalized array and stores
the final state in config (`environment.py:816–832`); this fact should not be
described as an online-only implementation, although the array's row formulas
use prior moments. The policy's price/momentum features use the corresponding
current/past indexed entries, rather than the final config moments.

Both original/v2 market inputs contain39,312 rows, from2025-01-01 through
2025-09-30. Their four source windows have identical timestamps:

| Start row | Native warm-up begins | Audited96 rows begin | Audited rows end |
|---:|---|---|---|
|0|2025-01-01 00:00|2025-01-03 16:00|2025-01-04 07:50|
|12,944|2025-03-31 21:20|2025-04-03 13:20|2025-04-04 05:10|
|25,888|2025-06-29 18:40|2025-07-02 10:40|2025-07-03 02:30|
|38,832|2025-09-27 16:00|2025-09-30 08:00|2025-09-30 23:50|

The384 warm-up transitions are native policy/environment transitions on the
preceding64 hours, with full original cash and no audit fair-sign intervention.
They preserve native buffers, halt flags and pending claims. They do not by
themselves prove all native features were available in an ordinary deployment.

## Forecast dates and declared information set

The frozen episode20 ANN metadata records training from2023-01-01 to2024-05-26
15:00, validation ending2024-09-13 06:00 and test ending2024-12-31 23:00. Those
dates precede all selected2025 contexts. The metadata declares a24-hour
settlement-price lookback and the information set
`settlement_history_strictly_before_origin_plus_current_day_ahead_entry`.

The actual source is consistent with that declared row ordering:
`forecast_price_experts.py:518–531` pads with a stored training-price mean and
uses `values[start:t]` (excluding the current target row);
`forecast_price_experts.py:542–549` appends current entry prices when declared.
`scripts/precompute_hourly_forecast_cache_v2.py:221–268` expands each hourly
origin prediction by holding it constant within that hour, refusing missing
hourly origins; its metadata at~354 onward records origin alignment and
`same_delivery_causal_v3`. `forecast_prior.py` updates calibration only on
strictly matured observations according to its native decision clock.

This verifies source ordering and recorded fit dates. It does not independently
verify the real market publication latency, revisions or timestamp conventions
of every settlement/day-ahead input, and does not certify a deployment data feed.

## Full-scenario context prevents a broad causal-interface claim

The native data-generation scripts explicitly compute a full-scenario risk
summary `std(revenue)/mean(revenue)` and repeat it as the CSV `risk` column:
`dataset generation/datagenerationforevaluationyear2025.py:257–259` and
`datagenerationforevaluationyear2025_unseendata_v2.py:194–196`.
The subsequent ffill/completion scripts recalculate it using their complete
output (`scripts/build_ffill_training_protocol.py:108–115`;
`scripts/complete_evaluation_window.py:171–177`). Each frozen2025 input has
exactly one unique risk value across all39,312 rows. It therefore incorporates
information beyond an early selected context.

This is economically reachable native state: `environment.py:838` loads
`_riskS`; `environment.py:6250` assigns it to market stress;
`environment.py:6284–6293` passes stress to the risk controller and stores
resulting snapshots. `risk.py:521–523` retains stress; `risk.py:316–320` uses
it in regulatory risk, which enters overall risk at `risk.py:369–390`.
The native rule risk cap and observations depend on those snapshots
(`environment.py:2960–2963`, `:6489–6530`). Preserving the original policy
interface preserves this engineered context feature; it is not evidence of a
leak-free historical trading evaluation.

Physical initialization also requires precise wording. The native constructor
fills the first rolling physical standard deviations using full-window std
(`environment.py:849–863`). These `_wind_std/_solar_std/_hydro_std` arrays have
no consumer in the reviewed native source path beyond assignment, so their
presence alone does not establish an actor-observation leak. More directly,
`_resolve_persistent_scale` (`environment.py:154–184`) returns the preexisting
historical scale for this environment but updates the config's next scale using
the full window's p95. The current `self.wind_scale/solar_scale/hydro_scale/
load_scale` values returned by this function therefore remain history-based
when the primed state is present. The complete cloned state nonetheless retains
native full-window metadata. No broad all-feature real-time guarantee follows.

## Safe scope and inferential distinction

Use wording such as: "384 native warm-up transitions preserve the actor's
original interface and state. Price normalization is primed on2024 history and
advanced through each preceding price prefix; the forecast construction uses
prior settlement windows. Native full-scenario engineered context/risk summaries
are retained, so this is a conditional simulator stress audit rather than a
leak-free chronological market backtest."

The fresh audit coin is drawn independently and hidden until after the actor's
action. Conditioning on the complete fixed native context—including its
engineered summary—does not reveal that new coin. Thus the chronology caveat is
distinct from the conditional financial-sleeve no-edge and sign-control
identities. It limits market/deployment claims; it does not by itself invalidate
those identities. This source review alone neither proves the executed trace
contracts nor establishes any financial endpoint; the separate frozen run and
independent arithmetic verification supply their own evidence.
