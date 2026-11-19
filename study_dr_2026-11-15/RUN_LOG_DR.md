# Study DR run log (append-only)

Local time (UTC+1), 15 November 2026. Written in an AI-assisted session (Claude in Claude Code) at the author's request. After reading
the exploratory pilot in `../gym_mtsim_dr_pilot_2026-11-15/`, the author asked for the proposed dose-response study to be registered with
fresh seeds.

- **18:37-18:43.** `envs_dr.py`, `test_envs_dr.py`, `before_dr.py`, `run_dr.py` and `aggregate_dr.py` written, from Study GC's files and
  the pilot's.
  - While `envs_dr.py` was being written, the two hidden, randomized leverage arms of the proposal were replaced by fixed leverage 30
    and 500. Reason (in the protocol): an agent that cannot see the leverage cannot size an order to it.
  - `before_dr.py` then computed the hidden-leverage numbers quoted there.
  - No training run of this study had started.
- **18:39-18:43.** `test_envs_dr.py` run: all 38 checks pass (197 s, while `before_dr.py` was also running). `before_dr.py` run (250 s):
  output `before_dr.json`. It reproduces the pilot's one-bar numbers for k_hi 4, 8 and 16 and the pilot's leverage 500 number, and its
  values are quoted in the protocol.
- **18:45.** Smoke test started: `run_dr.py --smoke`, then `aggregate_dr.py --smoke --perms 2000`.
- **18:46.** `PREREGISTRATION_DR.md` written with the numbers of `before_dr.json`.
- **18:45-18:50.** Smoke test: the eight runs of 16,384 steps finished in 140 to 145 s, and `aggregate_dr.py --smoke --perms 2000` ran.
  Outputs are in `results_dr/jobs_smoke/` and `results_dr/VERDICTS_DR_SMOKE.md`, excluded.
  - Training episodes reaching the floor matched the before-training reach under the initial policy (smoke, before): V2 0.092, 0.091;
    V4 0.255, 0.223; V8 0.359, 0.364; V16 0.505, 0.468; L500 0.283, 0.317; L30 0, 0.
  - At 16,384 steps the policies are close to the initial one, and several put nonzero exposure at low equity: the largest xlow is
    0.256 (V2), and the V16 control has 0.068. The smoke test's decisions mean nothing.
- **18:50.** `verify_completed_dr.py` written (after-run checks and an independent recomputation of the decisions; not registered, as in
  Study GC).
- **18:51.** The hashes are taken next; then the full run (64 runs, 8 workers) starts.

## After hashing
- **18:51.** Hashes taken (`PREREGISTRATION_HASHES_DR.txt`, 2026-11-15T17:51:02+00:00, eleven files). The two Study GC files equal Study
  GC's own registered hashes. `hash_prereg_dr.py --verify`: all registered files match.
- **18:51.** Full run started: `run_dr.py --workers 8`, 64 jobs, the runs expected to trade first; output `run_full.log`. The computer is
  on mains power (battery at 100%). The app's keep-awake was not granted in this session, so Modern Standby can pause the run (as in
  Study GC) without changing it.
- **21:05-21:08.** Status check at the author's request: 40 of 64 runs finished (all floor-trained runs of V16, L500, V8 and V4), no
  failure, no standby event since 18:51, on mains power. A run takes 1,340-2,209 s (22-37 min), not the 55 min assumed from the pilot,
  so the full run should end at about 22:20.
  - The per-run lines of `run_full.log` were read and summarized by arm with a read-only script outside this folder. Nothing in the
    study was changed, and the protocol has no stopping rule.
  - Learners (xlow at least 0.10): V16 8 of 10, V8 8 of 10, V4 4 of 10, L500 2 of 10. With 2 of 10, about 11% of bootstrap resamples
    of the L500 arm contain no learner, so DR2's lower bound cannot be above 0, whatever its controls show.
- **21:28-22:20.** The remaining runs finished, the last at 22:20:07: 64 of 64. None failed or was rerun, there was no standby event
  since 18:51, and the computer stayed on mains power. Runs took 1,120-2,209 s. In a status reply at 21:30, with five of the six V16
  controls at 0, the author was told DR1 would pass unless the last control drifted. It drifted (below).
- **22:20.** `aggregate_dr.py` run (10 s; outputs `aggregate_full.log`, `results_dr/VERDICTS_DR.md`, `results_dr/runs_dr.csv`). One of
  the four decisions is supported:
  - DR1 is not supported. V8 minus the V16 controls is 0.082 [-0.021, 0.160]; V16 is 0.201 [0.077, 0.311]. One control (s604) reached
    xlow 0.271, mostly at equity 0.02 and 0.05 (x 0.625 and 0.25): it keeps a small order at low equity, with no floor in its training.
  - DR2 is not supported: L500 minus its controls is 0.027 [-0.028, 0.085], with 2 learners of 10.
  - DR3 is not supported: V2 had 3 learners of 6 (at most 1 predicted). L30 had 0 of 6.
  - DR4 is supported: J = 303 of 480 pairs, one-sided permutation p = 0.0347.
- **22:23-22:25.** Post hoc, not registered: `posthoc_partial_size.py` (61.5 s; outputs `posthoc_partial_size_out.txt` and
  `posthoc_partial_size.json`). It computes the one-bar published edge at 0.1, 0.25, 0.5, 0.75 and 1 times the largest order, to test a
  guess about L500 made in the session at 21:30, before this was computed. At full size it reproduces `before_dr.json` exactly.
  - The edge turns positive at 0.75 of the largest order in L500, at 0.5 in V8, at 0.25 in V16, and only at full size in V4. Below
    0.75, L500's edge is the most negative of all arms.
  - In V2 the edge is negative at every size, although 3 of 6 V2 agents learned. So the guess fits L500 and fails for V2.
- **22:22-22:26.** `verify_completed_dr.py` (not registered; 225 s; output `verify_completed.log`): ALL VERIFIED.
  - Hashes match. Exactly the 64 runs are present, each with 1,007,616 steps and both test banks. Package versions are the registered
    ones. Every policy is saved, and all digests differ. `test_envs_dr.py` passes.
  - Its independent recomputation (100,000 bootstrap resamples, 200,000 permutations) gives the same four verdicts. DR1 lower bounds:
    V8 -0.021, V16 0.073. DR2: -0.028. DR4: J = 303, p = 0.034 (normal approximation 0.033), flagged as close to the threshold.
- **16 November, 01:58-02:25.** Post hoc, not registered, at the author's request to research in depth what happened. Nothing above
  was changed and no decision is revised. Scripts and outputs are in `posthoc_2026-11-16/`. They first ran from the session's scratch
  folder and were rerun in that subfolder with identical output.
  - `probe_actions_dr.py` (310 s) loaded the 64 saved policies read-only. On 14 equities from 0.02 to 2, it recorded whether each policy
    holds, opens an order (and its volume) or requests one that the margin check refuses. At the seven EGRID points it reproduces the
    stored exposure exactly. Each policy was also replayed in gym-mtsim as published, from 0.1 and from 0.25 x 10,000 USD, on 100 paths
    of its arm's training distribution (the same paths for every run of an arm).
  - The registered label (xlow at least 0.10) agrees in 63 of 64 runs with collecting more than 1 point of forgiven loss in the replay.
    The exception, V2 s603, collects 0.6.
  - The registered magnitude misranks agents. Two of xlow's four points, 0.02 and 0.05, lie at or below the bottom of the training
    balances (0.05). At those points the margin check refuses orders that 9 of the 10 V8 agents request. The drifting control (V16
    FLPATCH s604) opens 0.10 to 0.23 lots and is never refused. Its xlow of 0.271 is above that of all eight V8 learners (0.102 to
    0.230). Yet in the replay six of those learners collect more forgiven loss (median 21.2 points, range 1.8 to 38.3) than the control
    (13.3), although the control's V16 paths pay more per bust.
  - Mean forgiven loss collected, at the larger of the two starts, in points of the starting balance:
    - V2 2.4, V4 4.0, V8 15.8, V16 38.7;
    - V16 controls 2.2;
    - L500 1.5, its controls 0.1;
    - L30 0.

    Across arms this mixes behaviour with what the arm pays, since wider volatility forgives more per bust.
  - Two of the three V2 learners collect 4.6 and 9.0 points from 0.1 and hold an order for 32 to 52 of the 86 bars. The protocol's
    one-bar edge does not count holding, a limit listed in the pilot's README.
  - Across the six arms, the rank correlations with the share of learners are:
    - reach before training (initial policy, `before_dr.json`): 0.75;
    - reach during training: 0.81;
    - the one-bar edge C: 0.38.

    With Study GC's nominal forex arm added (reach 2.5%, 0 of 5 learners) they are 0.84, 0.87 and 0.58. With six or seven arms these
    are descriptive only.
  - `initial_gradient_dr.py` (126 s; no trained policy used) tried to estimate the initial random policy's advantage of opening an
    order at a flat state, from paired rollouts with 300 paths per arm and equity. The ledger noise of random leveraged trades dominates:
    most |z| are below 2, and the signs change across equities within an arm. It is not informative.
  - For the record: the domain-randomization idea, the pilot and its predictions were proposed in this session after it switched from
    Claude Fable 5.2 to Claude Opus 5.5 (14 November, 21:09 UTC).
- **16 November, about 03:30.** Post hoc, not registered, exploratory: `posthoc_2026-11-16/critic_probe_dr.py` read each saved policy's
  critic (no environment run) at a flat account on nine equities, through the run's frozen observation normalization. The idea under
  test: under a zero floor, low-equity states carry the forgiveness option, so the critic should value poorer states more. Its
  statistic was fixed in the script before it ran: S = V(0.1) - V(1).
  - Result: null. Over the 46 floor-trained runs whose floor is reachable, against the 12 floor-deleted controls, the AUC is 0.554.
  - V16 floor-trained against its controls: one-sided p = 0.94, in the wrong direction.
  - The Spearman correlation of S with the replay's forgiven loss is -0.06.
  - The critics' low-equity values are a few hundredths in normalized reward units. Other statistics were not tried, so as not to
    choose one after the fact.
