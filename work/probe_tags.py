import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
gt = pd.read_csv(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv"))["label"].values
st = pd.read_csv(os.path.join(HERE, "smart_tagging.csv"))
lo, hi = 11185, 18684
side = {"Barcelona": "H", "Atletic Club": "A"}
st = st[st.Team.isin(side)]
print("code                           n_in_window  agree%(alive GT frames)  agree%(if team is the defender -> opposite)")
for code, g in st.groupby("code"):
    ag = tot = 0
    n = 0
    for _, r in g.iterrows():
        s, e = max(r.s, lo), min(r.e, hi)
        if e <= s:
            continue
        n += 1
        seg = gt[s:e + 1]
        alive = seg[(seg == "H") | (seg == "A")]
        tot += len(alive)
        ag += (alive == side[r.Team]).sum()
    if tot:
        print("{:<30} {:3d}   {:5.1f}%   {:5.1f}%".format(code, n, 100 * ag / tot, 100 - 100 * ag / tot))
