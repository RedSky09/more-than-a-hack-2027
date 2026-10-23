import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
name = sys.argv[1]
lab = pd.read_csv(os.path.join(ROOT, "evaluation/results/%s.csv" % name))["label"].values
blocks = [(11185, 90651), (110808, 190197)]

durs, dead_runs = [], []
for a, b in blocks:
    s = lab[a:b + 1]
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        (durs if s[i] in "HA" else dead_runs if s[i] == "D" else []).append((j - i) / 25)
        i = j
d = np.array(durs)
print("{}: {} possessions mean {:.1f}s median {:.1f}s  <3s {}  >60s {}".format(
    name, len(d), d.mean(), np.median(d), (d < 3).sum(), (d > 60).sum()))
print("  D runs {}  mean {:.1f}s".format(len(dead_runs), np.mean(dead_runs)))
print("  D / H / A / X share of graded:")
g = np.concatenate([lab[a:b + 1] for a, b in blocks])
print("   ", {c: round(float(np.mean(g == c)), 3) for c in "DHAX"})
print("  D share per 5-min bucket:")
for a, b in blocks:
    print("   ", [int(100 * np.mean(lab[s:min(s + 7500, b + 1)] == "D")) for s in range(a, b, 7500)])
