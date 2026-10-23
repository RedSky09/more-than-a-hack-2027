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
lo, hi = 11185, 18684
n = len(ball)

px, py = players[:, :, 0] * L, players[:, :, 1] * W

# mean player displacement over +/- 12 frames (m/s), averaged over players visible at both ends
H = 12
dx = np.full_like(px, np.nan)
dy = np.full_like(py, np.nan)
dx[H:-H] = px[2 * H:] - px[: -2 * H]
dy[H:-H] = py[2 * H:] - py[: -2 * H]
pspeed = np.hypot(dx, dy) / (2 * H / 25)
with np.errstate(all="ignore"):
    mean_ps = np.nanmean(pspeed, axis=1)
    med_ps = np.nanmedian(pspeed, axis=1)
    p80_ps = np.nanpercentile(pspeed, 80, axis=1)

# ball gap length (frames) for each missing frame
miss = np.isnan(ball[:, 0])
gap = np.zeros(n, int)
i = 0
while i < n:
    if miss[i]:
        j = i
        while j < n and miss[j]:
            j += 1
        gap[i:j] = j - i
        i = j
    else:
        i += 1

# players in the 'centre' vs spread: std of positions
with np.errstate(all="ignore"):
    spread_x = np.nanstd(px, axis=1)

print("label  n    meanPS  medPS  p80PS   | missing%   | gap len median (frames, missing only)")
for c in "HAD":
    idx = np.where(gt[lo:hi + 1] == c)[0] + lo
    m = miss[idx]
    print("{}  {:5d}  {:5.2f}  {:5.2f}  {:5.2f}   | {:5.1f}      | {}".format(
        c, len(idx), np.nanmean(mean_ps[idx]), np.nanmean(med_ps[idx]), np.nanmean(p80_ps[idx]),
        100 * m.mean(), np.median(gap[idx][m]) if m.any() else None))

print("\nsegment-level view (per GT segment): mean player speed, % missing, ball start/end area")
cur, s = gt[lo], lo
for f in range(lo + 1, hi + 2):
    if f > hi or gt[f] != cur:
        e = f - 1
        sl = slice(s, e + 1)
        print("{} {:6d}-{:6d} {:5.1f}s  meanPS {:4.2f} p80PS {:4.2f} miss {:5.1f}%".format(
            cur, s, e, (e - s + 1) / 25, np.nanmean(mean_ps[sl]), np.nanmean(p80_ps[sl]),
            100 * miss[sl].mean()))
        if f <= hi:
            cur, s = gt[f], f

# full-match distribution of ball gaps
print("\nball-missing runs in graded halves (seconds):")
runs = []
for a, b in [(11185, 90651), (110808, 190197)]:
    i = a
    while i <= b:
        if miss[i]:
            j = i
            while j <= b and miss[j]:
                j += 1
            runs.append((j - i) / 25)
            i = j
        else:
            i += 1
runs = np.array(runs)
print("  count {}  >2s: {}  >5s: {}  >10s: {}  total missing share {:.1f}%".format(
    len(runs), (runs > 2).sum(), (runs > 5).sum(), (runs > 10).sum(),
    100 * runs.sum() * 25 / 158857))
