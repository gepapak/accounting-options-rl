# Evaluation software attempts

Attempt1 used run_heldout_benchmark.py and retained heldout_outputs/frozen_inputs.json.
The first case/repetition completed its simulation but JSON serialization failed
on a NumPy boolean before any endpoint was printed or persisted. No sample size,
case, policy, cap, criterion, source data or random seed was changed. The original
script and its frozen manifest remain byte-exact.

Attempt2 is a separate source file and output directory. Its only execution
changes are the output-directory name and a JSON encoder converting NumPy scalar
values to their corresponding Python scalars. It repeats the complete fixed
evaluation with the same random streams, including the first case. The protocol
and numerical kernel are unchanged. This is a software retry, not a new statistical
family or outcome-dependent extension.
