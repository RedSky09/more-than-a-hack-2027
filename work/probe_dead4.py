import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cache_atd import load  # noqa: E402
import features as F  # noqa: E402

players, ball, team = load()
ps = F.rolling_mean(F.player_speed(players), 50)
miss = np.isnan(ball[:, 0])
print("bucket (5 min)   p10   p25   p50   p75  | <1.3 share | ball missing share")
for half, a, b in [(1, 11185, 90651), (2, 110808, 190197)]:
    for s in range(a, b, 7500):
        e = min(s + 7500, b + 1)
        x = ps[s:e]
        x = x[~np.isnan(x)]
        print("h{} {:6d}-{:6d}  {:4.2f}  {:4.2f}  {:4.2f}  {:4.2f} |   {:4.0f}%    |   {:4.0f}%".format(
            half, s, e, *np.percentile(x, [10, 25, 50, 75]), 100 * np.mean(x < 1.3),
            100 * miss[s:e].mean()))
