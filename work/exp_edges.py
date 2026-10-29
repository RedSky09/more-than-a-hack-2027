"""Experiment: after the ball leaves the pitch, bridge a short gap to the next stoppage run.

Monkeypatches solution/make_submission.stoppage_mask; does not change the solution file.
usage: exp_edges.py <gapfill_frames> [<gapfill_frames> ...]   (0 = unchanged)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "solution"))
from cache_atd import load  # noqa: E402
import make_submission as M  # noqa: E402

players, ball, team = load()
original = M.stoppage_mask


def make_mask(gapfill):
    def mask(pl, bl):
        dead = original(pl, bl)
        if not gapfill:
            return dead
        bx, by = bl[:, 0] * M.PITCH_L, bl[:, 1] * M.PITCH_W
        with np.errstate(invalid="ignore"):
            off = ((bx < -M.OFFPITCH_M) | (bx > M.PITCH_L + M.OFFPITCH_M)
                   | (by < -M.OFFPITCH_M) | (by > M.PITCH_W + M.OFFPITCH_M))
        off = np.nan_to_num(off, nan=0).astype(bool)
        out = dead.copy()
        edges = np.flatnonzero(np.diff(np.concatenate(([0], dead.astype(np.int8), [0]))))
        runs = list(zip(edges[::2], edges[1::2]))
        filled = 0
        for (s0, e0), (s1, e1) in zip(runs[:-1], runs[1:]):
            gap = s1 - e0
            if 0 < gap <= gapfill and off[max(s0, e0 - 50):e0].any():
                out[e0:s1] = True
                filled += gap
        mask.filled = filled
        return out
    mask.filled = 0
    return mask


out_dir = os.path.join(ROOT, "evaluation", "results")
for arg in sys.argv[1:]:
    g = int(arg)
    fn = make_mask(g)
    M.stoppage_mask = fn
    labels = M.predict(players, ball, team)
    name = "exp_gap{}".format(g)
    M.write_submission(os.path.join(out_dir, name + ".csv"), labels)
    print(name, "frames filled:", fn.filled, {c: int((labels == c).sum()) for c in "HADX"})
