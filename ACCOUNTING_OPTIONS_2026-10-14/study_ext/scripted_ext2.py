"""What the engine's accounting pays in the edge market, with no learning: follow the signal at each leverage. Exploratory (development of Study EXT). Also the reward diagram of one bust step."""
import numpy as np

from envs_ext import ExtEnv, POSITIONS, P_EDGE

for split in ("train", "test"):
    env = ExtEnv(split, "edge", "REP", seed=2)
    print(f"--- {split}: follow the signal at leverage L, 3000 edge episodes (p={P_EDGE}): reported, booked, bust, steps")
    for L in (0, 1, 2, 5, 10):
        act = {+1: POSITIONS.index(L), -1: POSITIONS.index(-L)}
        rep, boo, bust, steps = [], [], [], []
        for ep in range(3000):
            o, _ = env.reset()
            R = B = 0.0
            d = tr = False
            n = 0
            while not (d or tr):
                z = 1 if o[0] >= 0 else -1
                o, _, d, tr, i = env.step(act[z])
                R += i["reported"]
                B += i["booked"]
                n += 1
            rep.append(R)
            boo.append(B)
            bust.append(i["bust"])
            steps.append(n)
        print(f"  L={L:>3d}: reported {np.mean(rep):+8.2f}  booked {np.mean(boo):+8.2f} (se {np.std(boo) / np.sqrt(3000):.2f})  bust {np.mean(bust):.3f}  steps {np.mean(steps):.1f}")

env = ExtEnv("test", "placebo", "REP", seed=3)
env.reset()
a = POSITIONS.index(10)
grid = [-64, -32, -16, -8, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 2, 4, 8, 16, 32, 64]
print("\nreward diagram of the first step at position +10, percent of W0:")
print("  t  :", " ".join(f"{t:>8}" for t in grid))
print("  phi:", " ".join(f"{env.query(a, t):>8.2f}" for t in grid))
move = float(env.core._price_array[env.core._idx + 1] / env.core._price_array[env.core._idx] - 1)
print("  realized move of the next price:", round(move, 4))
