"""Write a few v1 variants into evaluation/results/ (gitignored) so run_evaluation.py scores them."""
import itertools
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
    "v1_k12": dict(k=12, gap=50),
    "v1_k25": dict(k=25, gap=50),
    "v1_k25_dead": dict(k=25, gap=50, use_dead=True),
    "v1_k50": dict(k=50, gap=75),
}
for name, kw in variants.items():
    labels = P.build_labels(players, ball, team, **kw)
    P.write_submission(os.path.join(out_dir, name + ".csv"), labels)
    vals, cnt = np.unique(labels, return_counts=True)
    print(name, dict(zip(vals, cnt)))
