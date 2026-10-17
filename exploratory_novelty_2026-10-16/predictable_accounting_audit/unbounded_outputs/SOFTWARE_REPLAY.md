# Identical-stream replay after an output serialization failure

The first execution on 16 October 2026 reached final JSON serialization and
failed with `TypeError: Object of type bool_ is not JSON serializable`. It wrote
no results.json, table or numerical endpoint to the tool output. The pre-draw
script hash was
`f3c8c75ae75a6617b1ef2b45ebc8d2c870f66970eff566a4c5696e8467d91ab7`.

The only source correction converts the scalar h0 result from NumPy float64 to
native float at construction. This preserves its double-precision numerical
value and makes the derived repair comparison a native boolean for JSON.
No formula, policy, law, cap grid, scoring rule, seed, path count, range,
confidence allocation, guard or planned interpretation changes. The written
plan remains byte-exact with SHA256
`293e540c6724a85c7342b6b613bd6a14cf714eaa10f981c1953b23bdb0c03e14`.

The replay uses the identical predetermined streams and all fixed budgets.
There is no endpoint-informed adaptation, replacement sample or extension.
The final result records the corrected source hash; its runtime is the replay's
runtime and excludes the failed serialization attempt, whose traceback is
retained in the conversation record.

The complete attempt1 source is preserved as
`../run_unbounded_attempt1_snapshot.py`; its byte hash matches the original
pre-draw hash above. It was reconstructed by reverting only the one scalar-cast
line, and the match was verified before writing the snapshot. The corrected
replay source hash is
`99c90368d0a1c8e8d935036b0377415b43419b950d5c38fade316b724f8fb280`.
