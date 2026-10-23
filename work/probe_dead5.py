import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from cache_atd import load  # noqa: E402
import features as F  # noqa: E402
import possession as P  # noqa: E402

players, ball, team = load()
gt = pd.read_csv(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv"))["label"].values
dead = P.speed_dead_mask(players, thr=1.3, min_len=100, bridge=75)
off = P.dead_mask(ball)
dh, da = P.team_distances(players, ball, team)
dmin = np.fmin(dh, da)
bx, by = ball[:, 0] * P.L, ball[:, 1] * P.W
bs = np.full(len(ball), np.nan)
bs[1:] = np.hypot(np.diff(bx), np.diff(by)) * 25
bs_s = F.rolling_mean(bs, 25)
ps = F.player_speed(players)

edges = np.flatnonzero(np.diff(np.concatenate(([0], dead.astype(np.int8), [0]))))
print("run             len(s) gtD%  ballVis%  ballSpd  dMin_med  dMin_p25  playerSpd  offPitch%")
for s, e in zip(edges[::2], edges[1::2]):
    if not (11185 <= s <= 18684):
        continue
    sl = slice(s, e)
    g = 100 * np.mean(gt[sl] == "D")
    vis = 100 * np.mean(~np.isnan(ball[sl, 0]))
    print("{:6d}-{:6d} {:6.1f} {:4.0f}  {:6.0f}   {:6.2f}  {:7.2f}  {:7.2f}  {:7.2f}  {:6.0f}".format(
        s, e, (e - s) / 25, g, vis, np.nanmean(bs_s[sl]) if vis > 0 else np.nan,
        np.nanmedian(dmin[sl]) if vis > 0 else np.nan, np.nanpercentile(dmin[sl], 25) if vis > 0 else np.nan,
        np.nanmean(ps[sl]), 100 * off[sl].mean()))
