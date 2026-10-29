"""Where do two submissions differ?  usage: diff_versions.py old.csv new.csv"""
import sys

import numpy as np
import pandas as pd

a = pd.read_csv(sys.argv[1])["label"].values
b = pd.read_csv(sys.argv[2])["label"].values
d = a != b
print("frames differing:", int(d.sum()))
edges = np.flatnonzero(np.diff(np.concatenate(([0], d.astype(np.int8), [0]))))
runs = list(zip(edges[::2], edges[1::2]))
kinds = {}
for s, e in runs:
    key = "{}->{}".format("".join(sorted(set(a[s:e]))), "".join(sorted(set(b[s:e]))))
    kinds.setdefault(key, []).append(e - s)
for k, v in sorted(kinds.items()):
    print("  {:<8} runs {:3d} frames {:6d} longest {:5d}".format(k, len(v), sum(v), max(v)))
for s, e in runs:
    if 11185 <= s <= 18684 or 110808 <= s <= 114500:
        print("  {}-{} {} -> {}".format(s, e - 1, "".join(sorted(set(a[s:e]))), "".join(sorted(set(b[s:e])))))
