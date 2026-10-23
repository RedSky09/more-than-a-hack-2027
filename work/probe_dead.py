import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from cache_atd import load  # noqa: E402

players, ball, team = load()
L, W = 105.3, 68.0
gt = pd.read_csv(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv"))["label"].values

bx, by = ball[:, 0] * L, ball[:, 1] * W
px, py = players[:, :, 0] * L, players[:, :, 1] * W
dist = np.hypot(px - bx[:, None], py - by[:, None])
dist_h = np.where(team[None, :] == 0, dist, np.nan)
dist_a = np.where(team[None, :] == 1, dist, np.nan)


def nmin(a):
    with np.errstate(all="ignore"):
        r = np.nanmin(np.where(np.isnan(a), np.inf, a), axis=1)
    r[np.isinf(r)] = np.nan
    return r


dh, da = nmin(dist_h), nmin(dist_a)

segs = [("D", 12851, 12988), ("D", 15889, 16399), ("D", 17380, 17837), ("D", 18027, 18388),
        ("A", 16400, 17292), ("A", 12665, 12850)]
for lab, s, e in segs:
    print("== {} {}-{}".format(lab, s, e))
    for f in range(s, e + 1, 25 * 2 if e - s > 300 else 25):
        print("  f={} gt={} ball=({:6.1f},{:5.1f}) dH={:5.1f} dA={:5.1f}".format(
            f, gt[f], bx[f], by[f], dh[f], da[f]))
