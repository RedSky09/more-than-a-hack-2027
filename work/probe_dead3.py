import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from cache_atd import load  # noqa: E402
import features as F  # noqa: E402

players, ball, team = load()
gt = pd.read_csv(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv"))["label"].values
lo, hi = 11185, 18684
graded = np.zeros(len(ball), bool)
for a, b in [(11185, 90651), (110808, 190197)]:
    graded[a:b + 1] = True

ps = F.player_speed(players)
miss = np.isnan(ball[:, 0])
truth = gt[lo:hi + 1] == "D"

print("thr  win  mingap | D prec  rec   F1 | D share on public window | D share full match | #D runs full")
for win in (50, 100, 150):
    sm = F.rolling_mean(ps, win)
    for thr in (1.3, 1.5, 1.7, 1.9):
        for need_miss in (False, True):
            d = sm < thr
            if need_miss:
                d &= F.rolling_mean(miss.astype(float), win) > 0.5
            d = F.min_run(d, 75)          # >= 3 s
            d &= graded
            pr = d[lo:hi + 1]
            tp = (pr & truth).sum()
            prec = tp / max(pr.sum(), 1)
            rec = tp / truth.sum()
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            runs = ((d[1:] & ~d[:-1]).sum())
            print("{:.1f}  {:3d}  miss={} | {:.2f} {:.2f} {:.2f} | {:5.1f}% | {:5.1f}% | {}".format(
                thr, win, int(need_miss), prec, rec, f1, 100 * pr.mean(),
                100 * d[graded].mean(), runs))
