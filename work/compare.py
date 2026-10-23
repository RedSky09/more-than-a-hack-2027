"""Print GT vs prediction run lists over the public window.  usage: compare.py <name>"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
name = sys.argv[1]
lo, hi = 11185, 18684
gt = pd.read_csv(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv"))["label"].values
pr = pd.read_csv(os.path.join(ROOT, "evaluation/results/%s.csv" % name))["label"].values


def runs(a):
    out, s = [], lo
    for f in range(lo + 1, hi + 2):
        if f > hi or a[f] != a[s]:
            out.append((a[s], s, f - 1))
            s = f
    return out


g, p = runs(gt), runs(pr)
print("GT: {} runs   PRED: {} runs".format(len(g), len(p)))
print("{:<28} | {}".format("GT", name))
for i in range(max(len(g), len(p))):
    a = "{} {:6d}-{:6d} {:5.1f}s".format(*g[i][:3], (g[i][2] - g[i][1] + 1) / 25) if i < len(g) else ""
    b = "{} {:6d}-{:6d} {:5.1f}s".format(*p[i][:3], (p[i][2] - p[i][1] + 1) / 25) if i < len(p) else ""
    print("{:<28} | {}".format(a, b))
