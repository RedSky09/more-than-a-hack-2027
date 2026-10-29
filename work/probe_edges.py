"""Signals around the GT dead-ball boundaries (public window) vs what the detector does."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "solution"))
from cache_atd import load  # noqa: E402
import make_submission as M  # noqa: E402

players, ball, team = load()
gt = pd.read_csv(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv"))["label"].values
sub = pd.read_csv(os.path.join(ROOT, "evaluation/results/v3_final.csv"))["label"].values
ps = M.rolling_mean(M.mean_player_speed(players), M.SPEED_WINDOW)
bx, by = ball[:, 0] * M.PITCH_L, ball[:, 1] * M.PITCH_W
bs = np.full(len(ball), np.nan)
bs[1:] = np.hypot(np.diff(bx), np.diff(by)) * 25
bs = M.rolling_mean(bs, 13)

edges = [("onset", 12851), ("end", 12988), ("onset", 15889), ("end", 16399),
         ("onset", 17380), ("end", 17837), ("onset", 18027), ("end", 18388)]
for kind, f in edges:
    print("\n== GT D {} at {}".format(kind, f))
    for g in range(f - 100, f + 101, 20):
        print("  {:6d} gt={} sub={} playerSpd={:4.2f} ballSpd={:5.1f} ball=({:6.1f},{:5.1f})".format(
            g, gt[g], sub[g], ps[g], bs[g], bx[g], by[g]))
