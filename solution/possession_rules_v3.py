"""Possession labels v3: v2 plus ball-out-of-play evidence and short-gap interpolation.

v2 (possession_rules_v2.py) calls a stretch between two controlled runs dead only when it is long
(6 s), or when a Smart Tagging restart overlaps it. That over-predicts dead ball whenever the ball
is simply not tracked. v3 adds two ideas and keeps everything else from v2. Fixed rules, no learned
parameters.

  1. Ball out of play. The ball is out when the tracking shows it beyond a touchline or goal line,
     or when it was last seen right on one and then vanishes for a while (it left the camera's
     view with the play). A stretch between two controlled runs that contains such a moment and
     lasts at least OUT_MIN_GAP_S is dead from that moment on, not from the end of the previous
     run. The previous team keeps the ball until then. A ball that vanishes in the middle of the
     pitch is an occlusion, not a stoppage, and still counts as held possession.
  2. Short occlusions are interpolated. A gap of up to INTERP_MAX_GAP_S between two observations of
     the ball is filled in a straight line, unless that would need an impossible ball speed
     (INTERP_MAX_SPEED). This gives the control rules more evidence.

Not implemented: using team shape to guess possession through long gaps. It cannot be checked
without validation data we are allowed to use.

Every constant is a fixed default chosen from football reasoning. None was fitted, and none may be
tuned on the provided ground truth or the video (see RULES.md). Change one only on external data
with a checked licence, and record why.

    python solution/possession_rules_v3.py              # writes evaluation/results/possession_rules_v3.csv
    python evaluation/validate_submission.py <file>      # check the file before upload
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, require, source_frames  # noqa: E402
from baseline import team_of_each_track  # noqa: E402
from possession_rules import DEAD, read_atd, write_submission  # noqa: E402
import load_smart_tagging  # noqa: E402
import possession_rules_v2 as v2  # noqa: E402

# Params
INTERP_MAX_GAP_S = 1.0       # fill ball gaps up to this long
INTERP_MAX_SPEED = 30.0      # m/s; a gap that would need a faster ball is not filled
OUT_MARGIN_M = 0.0           # ball centre at or beyond a line by this much is outside
OUT_CONFIRM_FRAMES = 3       # outside frames needed within a 5 frame window, against tracker noise
DISAPPEAR_MIN_S = 1.0        # a ball missing at least this long after being seen on a line
DISAPPEAR_MARGIN_M = 1.5     # "on a line" means within this distance of it
OUT_MIN_GAP_S = 2.0          # shortest stretch that ball-out evidence can turn into dead ball
PITCH = GAMES[GAME]["pitch"]
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
OUT_NAME = "possession_rules_v3.csv"


def interpolate_short_gaps(ball, fps):
    """Copy of `ball` with short, physically plausible gaps filled in a straight line."""
    out = ball.copy()
    seen = np.flatnonzero(~np.isnan(ball[:, 0]))
    max_gap = int(round(INTERP_MAX_GAP_S * fps))
    for before, after in zip(seen[:-1], seen[1:]):
        missing = after - before - 1
        if missing < 1 or missing > max_gap:
            continue
        distance = np.hypot(*(ball[after] - ball[before]))
        if distance / ((after - before) / fps) > INTERP_MAX_SPEED:
            continue
        weight = (np.arange(before + 1, after) - before) / float(after - before)
        out[before + 1:after] = ball[before] + weight[:, None] * (ball[after] - ball[before])
    return out


def ball_out_flags(ball, fps, pitch=PITCH):
    """True at frames where the ball is shown to have left play. `ball` is the raw, uninterpolated one."""
    length, width = pitch
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        outside = ((x <= OUT_MARGIN_M) | (x >= length - OUT_MARGIN_M)
                   | (y <= OUT_MARGIN_M) | (y >= width - OUT_MARGIN_M))
    outside &= ~np.isnan(x)
    count = np.convolve(outside.astype(np.int8), np.ones(5, dtype=np.int8), mode="same")
    flags = outside & (count >= OUT_CONFIRM_FRAMES)

    seen = np.flatnonzero(~np.isnan(x))
    if seen.size > 1:
        lost = np.flatnonzero(np.diff(seen) > int(round(DISAPPEAR_MIN_S * fps)))
        last = seen[lost]
        with np.errstate(invalid="ignore"):
            near = ((x[last] <= DISAPPEAR_MARGIN_M) | (x[last] >= length - DISAPPEAR_MARGIN_M)
                    | (y[last] <= DISAPPEAR_MARGIN_M) | (y[last] >= width - DISAPPEAR_MARGIN_M))
        flags[last[near]] = True
    return flags


def label_block(out, runs, first, last, windows, flags, fps):
    """Fill out[first..last] from the controlled runs of one half."""
    if not runs:
        out[first:last + 1] = DEAD
        return
    out_min = int(round(OUT_MIN_GAP_S * fps))
    out[first:runs[0][1]] = runs[0][0]
    for index, (team, start, end) in enumerate(runs):
        out[start:end + 1] = team
        if index + 1 == len(runs):
            out[end + 1:last + 1] = team
            continue
        a, b = end + 1, runs[index + 1][1]
        hits = np.flatnonzero(flags[a:b])
        dead = v2.gap_is_dead(a, b, windows, fps) or (hits.size > 0 and b - a >= out_min)
        cut = a + int(hits[0]) if (dead and hits.size) else a
        out[a:cut] = team
        out[cut:b] = DEAD if dead else team


def label_frames(ball, players, team, graded, fps, windows=(), pitch=PITCH):
    """One label per frame: H, A, D or X. Pure function of its inputs."""
    flags = ball_out_flags(ball, fps, pitch)
    evidence = v2.control_evidence(interpolate_short_gaps(ball, fps), players, team, fps)
    out = np.full(len(ball), v2.X_CODE, dtype=np.int8)
    for first, last in v2.blocks_of(graded):
        runs = v2.confirm_runs(evidence[first:last + 1], fps, offset=first)
        label_block(out, runs, first, last, windows, flags, fps)
    return v2.TABLE[out]


def build(game=GAME, data_dir=None, use_tagging=True):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    team = team_of_each_track(meta_path)
    ball, players = read_atd(raw_path, n_frames, len(team), GAMES[game]["pitch"])

    windows = []
    if use_tagging:
        try:
            windows = v2.restart_windows(load_smart_tagging.load(game, data_dir))
        except IOError as error:
            print("Smart Tagging not available, continuing without it:\n{}".format(error))
    print("restart windows used: {}".format(len(windows)))
    flags = ball_out_flags(ball, GAMES[game]["fps"], GAMES[game]["pitch"])
    print("ball-out frames flagged: {}".format(int(flags.sum())))
    return label_frames(ball, players, team, v2.graded_mask(n_frames, game),
                        GAMES[game]["fps"], windows, GAMES[game]["pitch"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, OUT_NAME))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--no-tagging", action="store_true",
                        help="do not use Smart Tagging restart windows")
    args = parser.parse_args()

    labels = build(data_dir=args.data_dir, use_tagging=not args.no_tagging)
    counts = {label: int(np.sum(labels == label)) for label in ("H", "A", "D", "X")}
    print("H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("possessions over the whole match: {}".format(v2.count_possessions(labels)))
    print("-> {}".format(os.path.relpath(write_submission(args.out, labels), ROOT)))
