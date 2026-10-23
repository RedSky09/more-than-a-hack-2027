"""Cross-check the stoppage detector against Smart Tagging over the full match (no GT needed).

For every SET PIECES / GOALS instance: how much of the 5 s before the tag end is D, and does the
detector see a D run touching the interval. Compared with the same measure at random times.
usage: crosscheck_tags.py <submission name>
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
name = sys.argv[1] if len(sys.argv) > 1 else "v3_final"
lab = pd.read_csv(os.path.join(ROOT, "evaluation/results/%s.csv" % name))["label"].values
st = pd.read_csv(os.path.join(HERE, "smart_tagging.csv"))
blocks = [(11185, 90651), (110808, 190197)]
graded = np.zeros(len(lab), bool)
for a, b in blocks:
    graded[a:b + 1] = True
isD = lab == "D"


def touch(s, e):
    s, e = max(s, 0), min(e, len(lab) - 1)
    return isD[s:e + 1].mean()


rng = np.random.default_rng(0)
cand = np.flatnonzero(graded)
rand_share = np.mean([touch(c, c + 125) for c in rng.choice(cand, 2000)])
rand_any = np.mean([isD[c:c + 126].any() for c in rng.choice(cand, 2000)])
print("random 5 s windows: D share {:.2f}, contains any D {:.2f}\n".format(rand_share, rand_any))

print("{:<22} {:>4} {:>12} {:>12}".format("tag", "n", "D share[s,e]", "any D[s-4s,e]"))
sel = st[st.code.isin(["SET PIECES", "GOALS"])].copy()
sel["kind"] = np.where(sel.code == "GOALS", "GOALS", sel.Type.fillna("?"))
for kind, g in sel.groupby("kind"):
    g = g[graded[np.clip(g.s.values, 0, len(lab) - 1)]]
    if not len(g):
        continue
    share = np.mean([touch(r.s, r.e) for r in g.itertuples()])
    anyd = np.mean([isD[max(r.s - 100, 0):r.e + 1].any() for r in g.itertuples()])
    print("{:<22} {:>4} {:>12.2f} {:>12.2f}".format(kind, len(g), share, anyd))

# what is D doing at the *end* of D runs: is a set piece tag active near the restart?
sp = sel[sel.code == "SET PIECES"]
edges = np.flatnonzero(np.diff(np.concatenate(([0], (isD & graded).astype(np.int8), [0]))))
runs = list(zip(edges[::2], edges[1::2]))
near = 0
for s, e in runs:
    if ((sp.s <= e + 75) & (sp.e >= e - 150)).any():
        near += 1
print("\nD runs: {}   with a set-piece tag around their end (-6 s..+3 s): {} ({:.0f}%)".format(
    len(runs), near, 100 * near / max(len(runs), 1)))
miss = sp[graded[np.clip(sp.s.values, 0, len(lab) - 1)]]
missed = [r for r in miss.itertuples() if not isD[max(r.s - 100, 0):r.e + 1].any()]
print("set pieces with NO D in [s-4s, e]: {} / {}".format(len(missed), len(miss)))
for r in missed[:12]:
    print("   frame {:6d}-{:6d} {} {}".format(r.s, r.e, r.Type, r.Team))
