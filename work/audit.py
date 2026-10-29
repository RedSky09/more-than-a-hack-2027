"""Audit of the current submission: graded X frames, half edges, kick-off, file bytes."""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAMES, anchor_video_frame, half_video_blocks  # noqa: E402

path = os.path.join(ROOT, "submission", "INVICTA.csv")
raw = open(path, "rb").read()
print("bytes", len(raw), "CRLF lines", raw.count(b"\r\n"), "sha256", hashlib.sha256(raw).hexdigest()[:16])
lab = pd.read_csv(path)["label"].values
gt = pd.read_csv(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv"))["label"].values

blocks = half_video_blocks("BAR-ATH")
print("halves", blocks)
graded = np.zeros(len(lab), bool)
for _, a, b in blocks:
    graded[a:b + 1] = True
x = graded & (lab == "X")
print("graded X frames:", int(x.sum()), "in public window:", int(x[11185:18685].sum()))
edges = np.flatnonzero(np.diff(np.concatenate(([0], x.astype(np.int8), [0]))))
runs = list(zip(edges[::2], edges[1::2]))
print("graded X runs:", len(runs))
for s, e in runs[:8]:
    print("   {}-{} ({:.1f}s) before={} after={}".format(s, e - 1, (e - s) / 25, lab[s - 1], lab[e]))

k1 = anchor_video_frame(GAMES["BAR-ATH"]["kickoff"])
k2 = anchor_video_frame(GAMES["BAR-ATH"]["kickoff_2nd"])
print("kickoff anchors (video frame):", k1, k2)
for name, a, b in (("2nd half start -> kickoff", blocks[1][1], k2), ):
    seg = lab[a:b]
    print(name, a, b, "({:.1f}s)".format((b - a) / 25), {c: int((seg == c).sum()) for c in "HADX"})
for _, a, b in blocks:
    print("half", a, b, "first 3 s:", "".join(lab[a:a + 75:5]), " last 3 s:", "".join(lab[b - 74:b + 1:5]))
