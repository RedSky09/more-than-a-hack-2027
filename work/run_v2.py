"""v2: v1 state machine + player-speed stoppage detector (+ ball off-pitch)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from cache_atd import load  # noqa: E402
import possession as P  # noqa: E402

players, ball, team = load()
out_dir = os.path.join(ROOT, "evaluation", "results")

variants = {
    "v2b_len100_br75": dict(k=18, use_dead=True, speed_dead=dict(thr=1.3, min_len=100, bridge=75)),
    "v2b_len125_br75": dict(k=18, use_dead=True, speed_dead=dict(thr=1.3, min_len=125, bridge=75)),
    "v2b_len125_br125": dict(k=18, use_dead=True, speed_dead=dict(thr=1.3, min_len=125, bridge=125)),
    "v2b_len150_br125": dict(k=18, use_dead=True, speed_dead=dict(thr=1.3, min_len=150, bridge=125)),
}
for name, kw in variants.items():
    labels = P.build_labels(players, ball, team, **kw)
    P.write_submission(os.path.join(out_dir, name + ".csv"), labels)
    vals, cnt = np.unique(labels, return_counts=True)
    print(name, {str(v): int(c) for v, c in zip(vals, cnt)})
