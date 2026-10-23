"""Parse the tactical ATD once and cache it as numpy arrays (work/atd_cache.npz).

players: (N, 46, 3) float32 -> x, y (normalised 0-1), speed (m/s); NaN when missing
ball:    (N, 2)     float32 -> x, y (normalised 0-1); NaN when missing
team:    (46,)      int8    -> 0 home, 1 away
Row i == video frame i (frame_id - 1).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import require, source_frames  # noqa: E402
from baseline import team_of_each_track  # noqa: E402

CACHE = os.path.join(HERE, "atd_cache.npz")


def parse():
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    team = team_of_each_track(meta_path)
    n = source_frames()
    players = np.full((n, len(team), 3), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            groups = parts[1].split(";")
            players[i] = np.array([[float(v) for v in g.split(",")] for g in groups],
                                  dtype=np.float32)
            ball[i] = [float(v) for v in parts[2].split(",")[:2]]
    np.savez_compressed(CACHE, players=players, ball=ball, team=team)
    return players, ball, team


def load():
    if not os.path.isfile(CACHE):
        return parse()
    z = np.load(CACHE)
    return z["players"], z["ball"], z["team"]


if __name__ == "__main__":
    p, b, t = parse()
    print(p.shape, b.shape, t.shape, "home slots", int((t == 0).sum()), "away", int((t == 1).sum()))
