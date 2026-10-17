# Two-path clean-bundle software replay (prepared; execution held)

Root must explicitly release execution after the full native statistical run.
No native import, actor inference or rollout is performed while preparing this
script. This is software equivalence validation, not additional statistical
evidence, a selected incentive condition or a new confidence-bound experiment.

Use `replay_private_bundle.py --execute` only after release. `--bundle` accepts
any absolute location containing the unpacked complete packet's `Prototype5`
and `github/accounting-options-rl/...` subtrees. `--reference-dir` explicitly
identifies the original saved `warm_attempt_2` software traces; these are read
only as golden comparison arrays before the native runtime is imported. No
source, market input, model or environment is loaded from the original tree.
`--output-dir` must be a fresh exclusive local validation directory.

The exact two historical paths are retained:

1. Plain MAPPO seed7, strict guards, original context start0; native384-step
   original-sign warm-up, cash fraction0.1,96 audit steps, all audit signs+1.
2. Feasible-action MAPPO seed7, vulnerable guards, same original context/window,
   warm-up/cash/horizon; signs alternate+1,-1, beginning+1.

The new wrapper loads the unchanged frozen adapter from the packet. It patches
only that in-memory module's ENGINE/HERE/CODE/PAPER1 path roots to the supplied
packet and fresh output runtime directory; no file is edited. Native module
resolution is restricted to the packet, root module names must not already be
loaded, and all34 native source files are hash-checked before use. Every loaded
native module's `__file__` must resolve within the packet's native root.

After golden traces are loaded, a Python audit hook forbids file reads outside
the packet, the fresh output runtime, Python's installed runtime/dependencies
and the Windows system tree. It forbids writes outside the fresh output runtime.
The original Prototype5 tree is explicitly denied even if nested in another
allowed prefix. Temporary/cache/log roots are redirected into the new output.
Read attempts are recorded by relative packet path; no user-home data or original
market/checkpoint fallback is permitted. A blocked access is a retained software
failure, not permission to relax the guard silently.

Each path must match the original event/path/transfer arrays, including full
actor observations and actions. Ignore only the measured per-step timing array.
Integer/string arrays compare exactly; finite monetary/physical numerical arrays
use absolute1e-6 and relative1e-10 tolerance; actor observation/action arrays use
absolute/relative1e-7. Save new traces, numerical discrepancies, source/model/
input hashes, loaded-module origins and file-access provenance. Both queues must
end empty and native/config/warm-parent immutability gates must pass. Retain a
failure and all partial artifacts if any gate fails; do not change dynamics,
cash, signs, policy selection or tolerances in response to outcomes.

This first packet replay uses the existing installed dependency environment.
Even a PASS establishes packet-only source/input/policy equivalence for these
two paths, not a separately validated fresh package installation or every policy
and context. Original absolute paths inside copied configs remain metadata;
the guard tests that they are not used as data/model fallbacks.
