Revision experiment L — 10 October2026
20 planned agents: DK1/DK2, seeds0–4, injection charges0/1, matched reflection and continued trading, fresh training signs, target1M steps, final rollout1,007,616.
This is prospective work during revision, separate from the original internally registered Studies B–K and their46 hypotheses.

Files:
run_continuation.py: exact frozen training/evaluation runner. Its scientific contents and SHA256 are unchanged; it contains the original author's path.
run_portable.py: path adapter, leaving the frozen runner unmodified. Set RCT_ROOT to the companion repository when necessary; otherwise it detects the installed repository.
aggregate_continuation.py: strict aggregation of all20 agents, bootstrap over5 seed clusters retaining both zones; checks frozen source/protocol/panel hashes and accounting identities.
continuation_protocol.txt: frozen design, endpoints and interpretive limits.
continuation_experiment/prospective_manifest.json: source/protocol/panel hashes, freeze timestamp, package versions.
continuation_experiment/jobs/: every model checkpoint, normalization state and100-draw result file. Smoke results are excluded.
continuation_experiment/summary.json, agents.csv, fig_continuation.pdf: derived outputs.
run_ahead.py: original scheduling helper for the last planned seed, using separate temporary outputs. Scheduling affects neither the planned cohort nor its scientific settings.

Reproduce the released aggregation from this folder:
python aggregate_continuation.py
Validate planned training jobs or fill missing results:
python run_portable.py --workers 4
To retrain, preserve the released jobs in a separate directory first, then run the adapter. Existing completed jobs are skipped. The train-only smoke check is optional: python run_portable.py --smoke --workers 2.

The original manifest records original absolute paths. Aggregation checks those when present, otherwise resolves the same filenames in the released source folder and companion panel directory, with the same content hashes. The frozen runner is never rewritten by the path adapter.

The primary interval is descriptive, conditions on two fixed panels and shared evaluation signs, and uses five seed clusters.100 sign draws are not100 independent trained agents. Normalization uses the same procedure with treatment-specific realized statistics. Reflected cash growth is not booked trading profit. A positive finite-sample booked mean is possible under the no-edge expected-cost identity. Full charging repairs the raw monetary objective; it does not guarantee that finite-training PPO is flat.
