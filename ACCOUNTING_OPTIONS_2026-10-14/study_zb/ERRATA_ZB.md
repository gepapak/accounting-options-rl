# Study Z, Part B: errata to the hashed protocol

Written 4 November 2026, after the registered analysis had run. The hashed files are unchanged; this note records where their text and the
registered analysis differ.

1. **ZIB1 and ZIB2 say "For all 56 runs".** They were written when Part B had seven arms (56 runs). The FINDER arm (8 runs) was added before
   hashing (3 November, 18:22-18:23, see `RUN_LOG_ZB.md`), and the design paragraph and the run count were changed to 64, but these two lines
   were not. The hashed analysis script (`aggregate_zb.py`) tests all 64 runs. The registered analysis, run once at 01:11, reports ZIB1
   supported (64 of 64 runs) and ZIB2 not supported (largest TIR audit error 1.446 points of the allocation per week). Read as written, over the
   56 runs of the seven non-finder arms, ZIB2 would be supported: the largest error among them is 0.289 points. Every run above the bound is a
   FINDER run; all eight finder seeds converged to the same policy (maximal leverage in isolated margin at every probe), which drains the
   account about 51 times per edge-market week, so many of its steps are taken by an account that cannot pay its own cost, where TIR is not
   exact (THEORY_ZB.md, part 1). The manuscript reports the analysis as run (ZIB2 not supported) and states this discrepancy.
