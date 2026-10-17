Revision audit M: independent model-selection and certification

This folder evaluates all 20 frozen checkpoints from revision experiment L without retraining. The original 100-draw reported winners were fixed before M1 and M2. M1 uses 1,000 independent sign draws per zone. M2 uses 2,000 further independent draws per zone, with a frozen certificate plan. Policies share signs within each zone. Original registered studies are unchanged.

Both original winners remain first by M2 reported mean (+108.8% DK1, +104.3% DK2). One joint 95% guarantee across all 20 policies establishes positive expected reported profit and worst expected booked profit within each ten-agent zone set. Expected booked intervals are [-18.5,-16.7]% and [-10.8,-9.8]%. This depends on the fixed panels, independent fair signs, frozen policies and fee model. It does not certify real-market performance or the largest population reported mean.

Run from the companion code repository using its existing Python environment (NumPy, SciPy, PyTorch, Stable-Baselines3, Gymnasium, pandas, Matplotlib):
python revision_experiments/2026-10-10-selection/summarize_frontier.py
python revision_experiments/2026-10-10-selection/verify_frontier.py

The frozen evaluators can reproduce missing jobs; completed results are cached:
python revision_experiments/2026-10-10-selection/audit_frontier.py --workers 4
python revision_experiments/2026-10-10-selection/certify_frontier.py --workers 4

RCT_ROOT can point to a different companion repository on Windows. All checkpoint and panel hashes must match the manifest; do not overwrite or edit frozen sources/manifests to pass checks. The evidence archive requires the companion repo and the L checkpoint archive. This archive does not duplicate the 20 L checkpoints or original market panels.

Manifests retain the original freeze times and source hashes. M2's base evaluator prose mentions 1,000 draws; its explicit numeric field, wrapper and certificate plan specify the actual 2,000. This discrepancy is disclosed in the supplementary material rather than changing the frozen record. M2's procedure was fixed after M1 and before M2; neither revision audit is externally registered or part of the original 46-hypothesis scorecard.

Expected booked estimates use minus executed costs under the known no-edge identity. Empirical Bernstein cost intervals and upper-truncated Hoeffding reported-profit lower bounds are classical. Their joint accounting-based ranking application is described in the main paper. Empirical charge breakpoints are descriptive estimates and have no population confidence guarantee.

Checks: every cached policy evaluator agrees with its exported network on 10,000 probes. Four policies also reproduce three complete M2 paths using only direct network evaluation. Exact enumeration verifies selection probability 1-2^(-n), n=1,...,10. Two hundred random finite-policy sets verify affine winner intervals and monotonicity.
