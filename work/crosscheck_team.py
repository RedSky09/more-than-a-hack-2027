"""For set pieces that name a team: compare the tag's team with the first H/A label after the
nearest D run end. Also print the timing offset between that D end and the tag start."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
name = sys.argv[1] if len(sys.argv) > 1 else "v3_final"
lab = pd.read_csv(os.path.join(ROOT, "evaluation/results/%s.csv" % name))["label"].values
gt = pd.read_csv(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv"))["label"].values
st = pd.read_csv(os.path.join(HERE, "smart_tagging.csv"))
side = {"Barcelona": "H", "Atletic Club": "A"}
isD = lab == "D"
edges = np.flatnonzero(np.diff(np.concatenate(([0], isD.astype(np.int8), [0]))))
runs = list(zip(edges[::2], edges[1::2]))  # [s, e) half-open
ends = np.array([e for _, e in runs])

sp = st[(st.code == "SET PIECES") & st.Team.isin(side)]
rows = []
for r in sp.itertuples():
    k = np.argmin(np.abs(ends - (r.s + 125)))     # nearest D end to tag start + 5 s
    e = ends[k]
    if abs(e - r.s) > 25 * 30 or e >= len(lab):
        continue
    after = lab[e:e + 200]
    after = after[(after == "H") | (after == "A")]
    first = after[0] if len(after) else "?"
    g = gt[e:e + 200]
    g = g[(g == "H") | (g == "A")]
    rows.append(dict(type=r.Type, frame=r.s, off=(e - r.s) / 25, tag=side[r.Team], pred=first,
                     gt=g[0] if len(g) else "-"))
df = pd.DataFrame(rows)
print("matched set pieces with a team:", len(df))
print("D-run end minus tag start (s): median {:.1f}  p10 {:.1f}  p90 {:.1f}".format(
    df.off.median(), df.off.quantile(.1), df.off.quantile(.9)))
print("\ntag team == first label after D (by type):")
print(df.assign(agree=df.tag == df.pred).groupby("type").agree.agg(["mean", "count"]).round(2))
print("overall agreement {:.2f}".format((df.tag == df.pred).mean()))
pub = df[df["gt"] != "-"]
pub = pub[(pub.frame >= 11185) & (pub.frame <= 18684)]
print("\npublic window examples:\n", pub.to_string(index=False))
