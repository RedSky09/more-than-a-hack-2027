"""v3: v2b + merge short sandwiched spells; adaptive speed threshold variants."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from cache_atd import load  # noqa: E402
import possession as P  # noqa: E402

players, ball, team = load()
out_dir = os.path.join(ROOT, "evaluation", "results")

for name, ad in (("v3_adapt_lo7", dict(lo=0.7)), ("v3_adapt_lo9", dict(lo=0.9)),
                 ("v3_adapt_r45_lo8", dict(ratio=0.45, lo=0.8))):
    kw = dict(k=18, use_dead=True, merge_max=75,
              speed_dead=dict(thr=1.3, min_len=100, bridge=75, adaptive=ad))
    labels = P.build_labels(players, ball, team, **kw)
    P.write_submission(os.path.join(out_dir, name + ".csv"), labels)
