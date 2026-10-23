import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from cache_atd import load  # noqa: E402
from games import asset_paths  # noqa: E402

players, ball, team = load()
L, W = 105.3, 68.0
gt = pd.read_csv(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv"))
lab = gt["label"].values
lo, hi = 11185, 18684
win = np.arange(lo, hi + 1)

print("== GT segments (label, start, end, seconds, ball visible %)")
seg = []
cur = lab[lo]
s = lo
for f in range(lo + 1, hi + 2):
    if f > hi or lab[f] != cur:
        e = f - 1
        vis = 100 * np.mean(~np.isnan(ball[s:e + 1, 0]))
        seg.append((cur, s, e))
        print("  {} {:>6}-{:>6} {:6.1f}s  ball {:5.1f}%".format(cur, s, e, (e - s + 1) / 25, vis))
        if f <= hi:
            cur, s = lab[f], f

print("\n== ball visibility by GT label")
for c in "HAD":
    idx = np.where(lab[lo:hi + 1] == c)[0] + lo
    print("  {} n={} visible={:.1f}%".format(c, len(idx), 100 * np.mean(~np.isnan(ball[idx, 0]))))

# ball speed (m/s) from consecutive visible frames
bx, by = ball[:, 0] * L, ball[:, 1] * W
v = np.hypot(np.diff(bx), np.diff(by)) * 25
v = np.concatenate(([np.nan], v))
print("\n== ball speed m/s by GT label (median, p90)")
for c in "HAD":
    idx = np.where(lab[lo:hi + 1] == c)[0] + lo
    x = v[idx]
    x = x[~np.isnan(x)]
    print("  {} median {:.2f} p90 {:.2f}".format(c, np.median(x), np.percentile(x, 90)))

# is ball inside pitch? 
inside = (ball[:, 0] >= 0) & (ball[:, 0] <= 1) & (ball[:, 1] >= 0) & (ball[:, 1] <= 1)
print("\n== ball inside [0,1]^2 by label")
for c in "HAD":
    idx = np.where(lab[lo:hi + 1] == c)[0] + lo
    ok = ~np.isnan(ball[idx, 0])
    print("  {} visible&inside {:.1f}%  visible&outside {:.1f}%".format(
        c, 100 * np.mean(inside[idx]), 100 * np.mean(ok & ~inside[idx])))

# distance to nearest player, by label
d = np.full(len(ball), np.nan)
px = players[:, :, 0] * L
py = players[:, :, 1] * W
dd = np.hypot(px - bx[:, None], py - by[:, None])
with np.errstate(all="ignore"):
    d = np.nanmin(np.where(np.isnan(dd), np.inf, dd), axis=1)
d[np.isinf(d)] = np.nan
print("\n== nearest player distance (m) by label: median, p90")
for c in "HAD":
    idx = np.where(lab[lo:hi + 1] == c)[0] + lo
    x = d[idx]
    x = x[~np.isnan(x)]
    print("  {} median {:.2f} p90 {:.2f}".format(c, np.median(x), np.percentile(x, 90)))

print("\n== player count visible, by label (median)")
n_vis = np.sum(~np.isnan(players[:, :, 0]), axis=1)
for c in "HAD":
    idx = np.where(lab[lo:hi + 1] == c)[0] + lo
    print("  {} median {}".format(c, np.median(n_vis[idx])))

# smart tagging in window
print("\n== Smart tagging instances overlapping the public window")
root = ET.parse(asset_paths()["smart_tagging"]).getroot()
rows = []
for inst in root.iter("instance"):
    st, en = round(float(inst.findtext("start")) * 25), round(float(inst.findtext("end")) * 25)
    labels = {l.findtext("group"): l.findtext("text") for l in inst.iter("label")}
    rows.append(dict(id=int(inst.findtext("ID")), s=st, e=en, code=inst.findtext("code"), **labels))
st = pd.DataFrame(rows).sort_values("s")
pd.set_option("display.width", 200)
print(st[(st.e >= lo) & (st.s <= hi)][["id", "s", "e", "code", "Team", "Type", "Half"]].to_string(index=False))
st.to_csv(os.path.join(HERE, "smart_tagging.csv"), index=False)
