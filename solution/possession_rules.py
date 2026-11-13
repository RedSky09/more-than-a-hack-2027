"""Possession labels from the provided tactical tracking. Fixed rules, no learned parameters.

Per video frame:

  1. The ball is located by the ATD. Where it is missing, the frame has no team answer yet.
  2. The player nearest the ball owns it, if that player is within CONTROL_RADIUS_M of the ball.
  3. Unanswered runs up to MAX_GAP_S are bridged. If the same team owns the ball on both sides,
     that team keeps it. If the two sides differ, the gap splits at its midpoint. Longer
     unanswered runs are dead ball (D).
  4. Possession runs shorter than MIN_POSSESSION_S are merged into the run before them, so
     per-frame flicker does not create extra possessions.
  5. Frames outside the two graded halves are X. They are never scored.

The constants are fixed defaults chosen from football reasoning. They were not fitted, and they
must not be tuned on the provided ground truth or the video (see RULES.md).

    python solution/possession_rules.py                 # writes evaluation/results/possession_rules.csv
    python evaluation/validate_submission.py <file>      # check the file before upload
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import FIELD_SEP, GROUP_SEP, MISSING, team_of_each_track  # noqa: E402

# Params
CONTROL_RADIUS_M = 3.0
MAX_GAP_S = 2.0
MIN_POSSESSION_S = 1.0
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
OUT_NAME = "possession_rules.csv"

HOME, AWAY, UNKNOWN, DEAD = 0, 1, -1, 2
LABEL_OF = {HOME: "H", AWAY: "A", DEAD: "D"}


def read_atd(raw_path, n_frames, n_slots, pitch):
    """Ball (n, 2) and player (n, slots, 2) positions in metres. NaN where not observed.

    Frame ids count from 1 in the raw file, so video frame = frame_id - 1.
    """
    length, width = pitch
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue

            ball_part = parts[2]
            if ball_part[0] != MISSING:
                comma = ball_part.index(",")
                ball[frame] = (float(ball_part[:comma]) * length,
                               float(ball_part[comma + 1:ball_part.index(",", comma + 1)]) * width)

            for slot, group in enumerate(parts[1].split(GROUP_SEP)[:n_slots]):
                if group[0] == MISSING:
                    continue
                comma = group.index(",")
                players[frame, slot] = (float(group[:comma]) * length,
                                        float(group[comma + 1:group.index(",", comma + 1)]) * width)
    return ball, players


def nearest_team(ball, players, team):
    """HOME/AWAY for the player nearest the ball within the control radius, else UNKNOWN."""
    gap = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
    gap = np.where(np.isnan(gap), np.inf, gap)
    slot = np.argmin(gap, axis=1)
    nearest = gap[np.arange(len(slot)), slot]
    codes = np.where(nearest <= CONTROL_RADIUS_M, team[slot], UNKNOWN).astype(np.int8)
    codes[np.isnan(ball[:, 0])] = UNKNOWN
    return codes


def runs(codes):
    change = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(codes)]))
    return starts, ends


def fill_gaps(codes, fps):
    max_gap = int(round(MAX_GAP_S * fps))
    out = codes.copy()
    n = len(codes)
    starts, ends = runs(codes)
    for start, end in zip(starts, ends):
        if codes[start] != UNKNOWN:
            continue
        left = codes[start - 1] if start > 0 else UNKNOWN
        right = codes[end] if end < n else UNKNOWN
        length = end - start
        if length > max_gap or left == UNKNOWN or right == UNKNOWN:
            out[start:end] = DEAD
        elif left == right:
            out[start:end] = left
        else:
            middle = start + length // 2
            out[start:middle] = left
            out[middle:end] = right
    return out


def absorb_short(codes, fps):
    min_run = int(round(MIN_POSSESSION_S * fps))
    out = codes.copy()
    starts, ends = runs(out)
    for start, end in zip(starts, ends):
        if out[start] in (HOME, AWAY) and end - start < min_run and start > 0:
            out[start:end] = out[start - 1]
    return out


def graded_mask(n_frames, game):
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_video_blocks(game):
        mask[max(first, 0):min(last, n_frames - 1) + 1] = True
    return mask


def label_frames(ball, players, team, graded, fps):
    """One label per frame: H, A, D or X. Pure function of its inputs."""
    codes = nearest_team(ball, players, team)
    codes = fill_gaps(codes, fps)
    codes = absorb_short(codes, fps)
    labels = np.array([LABEL_OF.get(int(code), "D") for code in codes])
    labels[~graded] = "X"
    return labels


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.write("".join("{},{}\n".format(frame, label)
                             for frame, label in enumerate(labels)))
    return path


def build(game=GAME, data_dir=None):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    team = team_of_each_track(meta_path)
    ball, players = read_atd(raw_path, n_frames, len(team), GAMES[game]["pitch"])
    return label_frames(ball, players, team, graded_mask(n_frames, game), GAMES[game]["fps"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, OUT_NAME))
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    labels = build(data_dir=args.data_dir)
    counts = {label: int(np.sum(labels == label)) for label in ("H", "A", "D", "X")}
    print("H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("-> {}".format(os.path.relpath(write_submission(args.out, labels), ROOT)))
