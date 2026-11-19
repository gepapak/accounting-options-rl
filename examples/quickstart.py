"""Quick start: read a hidden loss floor from a simulator's reward, without its ledger.

The toy simulator below is one trading step of an account whose cash is floored at zero, the accounting that the paper audits:
the reward is the change in floored equity, so a loss beyond the cash is silently forgiven. The trainer sees only the reward of
clones of the step in which the market move is scaled by t (t = 1 is the real step, t = -1 the sign twin, t = 0 no move), and its
own cost. It never sees the cash, the position or the payoff formula.

Run from the repository root (needs numpy and gymnasium, as in requirements-studies.txt):

    python examples/quickstart.py

Each printed number is checked by an assertion, so the script doubles as a smoke test of the released package
(ACCOUNTING_OPTIONS_2026-10-14/twin_reward/). The exact checks of the theory are in that folder's test_*.py files.
"""
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ACCOUNTING_OPTIONS_2026-10-14" / "twin_reward"))
import gymnasium as gym  # noqa: E402
from reward_diagram import geometric_grid, line_estimate, minimax_estimate, read_diagram, scan  # noqa: E402
from twin_reward import TwinRewardWrapper, spr, tir, twin_bracket  # noqa: E402


def floored_step(cash, position, move, cost):
    """One step of a zero-floor account: (reported reward, booked P&L). Hidden from the trainer."""
    booked = position * move - cost
    reported = max(cash + booked, 0.0) - cash
    return reported, booked


# 1. One step whose loss exceeds the cash.
cash, position, move, cost = 100.0, 4.0, -40.0, 1.0          # booked -161 on a cash of 100: 61 is forgiven
def query(t):                                                 # the reward of a clone with the move scaled by t
    return floored_step(cash, position, t * move, cost)[0]

r, booked = floored_step(cash, position, move, cost)
print(f"1. reported reward {r:+.1f}, booked P&L {booked:+.1f}: the reward hides {r - booked:.1f} of forgiven loss")

# 2. The sign twin (Propositions 4, 5 and 7): two clones.
r_twin = query(-1.0)
lo, hi = (float(v) for v in twin_bracket(r, r_twin, cost))
x_tir, x_spr = float(tir(r, r_twin, cost)), float(spr(r, r_twin, cost))
print(f"2. sign twin: TIR {x_tir:+.1f} (exact here), SPR {x_spr:+.1f} (a lower bound on the gross), "
      f"bracket [{lo:+.1f}, {hi:+.1f}] of width {hi - lo:.1f} = the credit on the pair")
assert x_tir == booked and lo <= booked <= hi and abs((hi - lo) - (r + r_twin + 2 * cost)) < 1e-12

# 3. The reward diagram (Proposition 9, Algorithm 2): the whole line reads the floor, the exposure and the booked P&L.
ts, phi = scan(query, geometric_grid(64.0))
d = read_diagram(ts, phi)
x_line, n_queries, _ = line_estimate(query, cost)
print(f"3. reward diagram: convex kinks at t = {[k for k, _ in d.convex_kinks]}, flat tail at {d.level_right:+.1f} "
      f"(hidden cash {d.hidden_level:.1f}), gross {d.gross:+.1f}; adaptive reading {x_line:+.1f} from {n_queries} clones")
assert d.hidden_level == cash and d.gross == position * move and x_line == booked and not d.concave_kinks

# 4. The range law (Proposition 10): the endpoint estimate errs by at most (E + A + 2 eta) / Lambda, whatever the number of queries.
for lam in (1.0, 10.0, 100.0, 1000.0):
    err = abs(minimax_estimate(query, cost, lam) - booked)
    print(f"4. range {lam:6.0f}: endpoint error {err:8.3f} <= bound {(cash + cost) / lam:8.3f}")
    assert err <= (cash + cost) / lam + 1e-9

# 5. Logs do not buy range (Proposition 11): 5,000 logged moves that never reach the kink cannot see the floor.
rng = np.random.default_rng(0)
log_t = rng.uniform(-0.5, 0.5, 5000)                          # the kink is at t = (cash - cost) / 160 = 0.62
floored = np.array([query(t) for t in log_t])
unfloored = np.array([position * t * move - cost for t in log_t])
slope = np.polyfit(log_t, floored, 1)[0]
print(f"5. log of 5,000 small moves: identical with and without the floor (max gap {np.max(np.abs(floored - unfloored)):.1e}); "
      f"its fitted line predicts {slope - cost:+.1f} at t = 1, where the simulator reports {r:+.1f}")
assert np.max(np.abs(floored - unfloored)) <= 1e-9          # identical up to rounding (about 1e-14 on a typical run)


# 6. The wrapper: train on TIR instead of the reported reward. A placebo market (fair signs), an empty account, a random policy.
class ZeroFloorEnv(gym.Env):
    """Toy zero-floor account. reverse_next_sign() is the clone hook the wrapper needs; info['own_cost'] is the agent's own cost."""
    action_space = gym.spaces.Discrete(3)                     # short, flat, long
    observation_space = gym.spaces.Box(-np.inf, np.inf, (1,))

    def __init__(self, cash=0.0, fee=0.1, seed=0):
        self.cash, self.fee, self.rng, self._flip, self.booked_total = cash, fee, np.random.default_rng(seed), False, 0.0

    def reverse_next_sign(self):
        self._flip = True

    def step(self, action):
        q = 4.0 * (int(action) - 1)
        s = self.rng.choice((-1.0, 1.0)) * self.rng.exponential(10.0)   # draw first, so a clone keeps the same random state
        s = -s if self._flip else s
        self._flip = False
        c = self.fee * abs(q)
        r, x = floored_step(self.cash, q, s, c)
        self.cash += r
        self.booked_total += x
        return np.array([self.cash]), r, False, False, {"own_cost": c, "booked": x}

    def reset(self, *, seed=None, options=None):
        return np.array([self.cash]), {}


env = TwinRewardWrapper(ZeroFloorEnv(), estimator="tir")
env.action_space.seed(0)                                      # a reproducible random policy
reported = estimated = 0.0
for _ in range(2000):
    _, x_hat, _, _, info = env.step(env.action_space.sample())
    reported += info["reported_reward"]
    estimated += x_hat
booked_total = env.unwrapped.booked_total
print(f"6. 2,000 random steps from an empty account in a fair market: reported {reported:+.0f}, booked {booked_total:+.0f}, "
      f"TIR {estimated:+.1f} (TIR misses the booked total by {abs(estimated - booked_total):.2f}; the reported total by "
      f"{reported - booked_total:.0f})")
# The floor only forgives, so the reported total is at least the booked one. TIR is exact on every step whose twin pair identifies
# the booked P&L and pays minus the cost on the rest (a move smaller than the fee from an empty account censors both branches), so its
# total is close to the booked total, not equal to it. Whether one random path books a loss is luck, so it is not asserted.
assert reported >= booked_total
assert abs(estimated - booked_total) <= 0.01 * (reported - booked_total)
print("all quickstart checks pass")
