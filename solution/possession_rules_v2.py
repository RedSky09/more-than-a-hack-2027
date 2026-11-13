"""Possession labels v2: hold possession until the other team shows controlled possession.

v1 (possession_rules.py) answered "who is nearest the ball?" frame by frame and called everything
else dead ball. That contradicts CHALLENGE.md, which says a loose or contested ball stays with the
last team that had it and that only *controlled* possession changes hands. v2 follows that
definition directly. Fixed rules, no learned parameters.

Per graded half:

  1. Control evidence, per frame. The ball must be located, and the nearest player of one team must
     be within CONTROL_RADIUS_M of it and clearly closer than the nearest opponent
     (CONTROL_MARGIN_M). Frames where the ball moves faster than a player can dribble
     (MAX_CONTROL_BALL_SPEED), or lies still (a restart being set up), give no evidence.
  2. Controlled runs. Consecutive evidence frames of one team form a run. A run shorter than
     MIN_CONTROL_S is a touch, deflection or flicker, not control, and is discarded. Evidence
     separated by more than LINK_GAP_S does not join into one run.
  3. Possession is held. Between two controlled runs the previous team keeps the ball until the
     next run starts, which is when the opponent gains control. Before the first run the first
     team is used, because play restarts with a clear possessor.
  4. Dead ball needs evidence. A stretch between two runs is D only if no run happened for
     DEAD_GAP_S, or for TAGGED_MIN_GAP_S while overlapping a Smart Tagging restart (throw-in,
     corner, goal kick, free kick, kick-off after a goal). Dead stretches split possessions, as the
     ground-truth definition requires.
  5. Frames outside the two graded halves are X. They are never scored.

Every constant below is a fixed default chosen from football reasoning. None was fitted, and none
may be tuned on the provided ground truth or the video (see RULES.md). Change one only on external
data, and record why.

    python solution/possession_rules_v2.py              # writes evaluation/results/possession_rules_v2.csv
    python evaluation/validate_submission.py <file>      # check the file before upload
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import team_of_each_track  # noqa: E402
from possession_rules import (AWAY, DEAD, HOME, UNKNOWN, read_atd,  # noqa: E402
                              write_submission)
import load_smart_tagging  # noqa: E402

# Params
CONTROL_RADIUS_M = 3.0       # a player further than this is not controlling the ball
CONTROL_MARGIN_M = 1.0       # nearest team must beat the other team's nearest player by this much
MAX_CONTROL_BALL_SPEED = 8.0  # m/s; faster than any dribble, so the ball is in a pass or shot
SPEED_WINDOW_S = 0.4         # ball speed is measured across this window
REST_WINDOW_S = 1.0          # a ball inside REST_EXTENT_M for +/- this long is at rest
REST_EXTENT_M = 1.0
MIN_CONTROL_S = 1.5          # evidence needed before a team counts as controlling the ball
LINK_GAP_S = 3.0             # evidence further apart than this is not one run
DEAD_GAP_S = 6.0             # no controlled run for this long: the ball is dead
TAGGED_MIN_GAP_S = 2.0       # shorter gap that is dead if a Smart Tagging restart overlaps it
RESTART_CODE = "set pieces"
RESTART_TYPES = {"throw-in", "corner kick", "goal kick", "free kick", "kick off (after goal)"}
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
OUT_NAME = "possession_rules_v2.csv"

X_CODE = 3
TABLE = np.array(["H", "A", "D", "X"])


def ball_speed(ball, fps):
    """Ball speed in m/s from positions SPEED_WINDOW_S apart. NaN where either end is missing."""
    half = max(1, int(round(SPEED_WINDOW_S * fps / 2)))
    speed = np.full(len(ball), np.nan, dtype=np.float32)
    if len(ball) > 2 * half:
        span = np.hypot(ball[2 * half:, 0] - ball[:-2 * half, 0],
                        ball[2 * half:, 1] - ball[:-2 * half, 1])
        speed[half:len(ball) - half] = span / (2 * half / fps)
    return speed


def ball_at_rest(ball, fps):
    """True where the ball stayed within REST_EXTENT_M for +/- REST_WINDOW_S around the frame."""
    half = int(round(REST_WINDOW_S * fps))
    extent = []
    for axis in (0, 1):
        series = pd.Series(ball[:, axis].astype(np.float64))
        roll = series.rolling(2 * half + 1, center=True, min_periods=half + 1)
        extent.append(roll.max().to_numpy() - roll.min().to_numpy())
    with np.errstate(invalid="ignore"):
        return np.hypot(extent[0], extent[1]) <= REST_EXTENT_M


def control_evidence(ball, players, team, fps):
    """HOME/AWAY where one team clearly controls the ball, else UNKNOWN."""
    with np.errstate(invalid="ignore"):
        gap = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
        gap = np.where(np.isnan(gap), np.inf, gap)
        home = np.where(team == HOME, gap, np.inf).min(axis=1)
        away = np.where(team == AWAY, gap, np.inf).min(axis=1)
        nearest = np.minimum(home, away)
        clear = (np.maximum(home, away) - nearest) >= CONTROL_MARGIN_M
        slow = ~(ball_speed(ball, fps) > MAX_CONTROL_BALL_SPEED)
        ok = (nearest <= CONTROL_RADIUS_M) & clear & slow & ~ball_at_rest(ball, fps)
    owner = np.where(home < away, HOME, AWAY)
    return np.where(ok, owner, UNKNOWN).astype(np.int8)


def confirm_runs(codes, fps, offset=0):
    """[(team, first_frame, last_frame)] for runs of control evidence lasting MIN_CONTROL_S."""
    frames = np.flatnonzero(codes != UNKNOWN)
    if frames.size == 0:
        return []
    team = codes[frames]
    link = int(round(LINK_GAP_S * fps))
    need = int(round(MIN_CONTROL_S * fps))
    breaks = np.flatnonzero((team[1:] != team[:-1]) | (np.diff(frames) > link)) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [len(team)]))
    return [(int(team[s]), int(frames[s]) + offset, int(frames[e - 1]) + offset)
            for s, e in zip(starts, ends) if e - s >= need]


def gap_is_dead(start, end, windows, fps):
    """Is the unanswered stretch [start, end) a stoppage rather than live play?"""
    length = end - start
    if length >= int(round(DEAD_GAP_S * fps)):
        return True
    if length >= int(round(TAGGED_MIN_GAP_S * fps)):
        return any(first < end and last > start for first, last in windows)
    return False


def label_block(out, runs, first, last, windows, fps):
    """Fill out[first..last] from the controlled runs of one half."""
    if not runs:
        out[first:last + 1] = DEAD
        return
    out[first:runs[0][1]] = runs[0][0]
    for index, (team, start, end) in enumerate(runs):
        out[start:end + 1] = team
        if index + 1 == len(runs):
            out[end + 1:last + 1] = team
            continue
        next_start = runs[index + 1][1]
        dead = gap_is_dead(end + 1, next_start, windows, fps)
        out[end + 1:next_start] = DEAD if dead else team


def blocks_of(mask):
    """[(first, last)] for each contiguous run of True."""
    edges = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [len(mask)]))
    return [(int(s), int(e) - 1) for s, e in zip(starts, ends) if mask[s]]


def restart_windows(events):
    """Video-frame intervals of the Smart Tagging restarts that end a dead-ball stretch."""
    windows = []
    for event in events:
        if (event["code"] or "").strip().lower() != RESTART_CODE:
            continue
        if (event["tags"].get("Type") or "").strip().lower() in RESTART_TYPES:
            windows.append((event["start_frame"], event["end_frame"]))
    return windows


def label_frames(ball, players, team, graded, fps, windows=()):
    """One label per frame: H, A, D or X. Pure function of its inputs."""
    evidence = control_evidence(ball, players, team, fps)
    out = np.full(len(ball), X_CODE, dtype=np.int8)
    for first, last in blocks_of(graded):
        runs = confirm_runs(evidence[first:last + 1], fps, offset=first)
        label_block(out, runs, first, last, windows, fps)
    return TABLE[out]


def graded_mask(n_frames, game):
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_video_blocks(game):
        mask[max(first, 0):min(last, n_frames - 1) + 1] = True
    return mask


def build(game=GAME, data_dir=None, use_tagging=True):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    team = team_of_each_track(meta_path)
    ball, players = read_atd(raw_path, n_frames, len(team), GAMES[game]["pitch"])

    windows = []
    if use_tagging:
        try:
            windows = restart_windows(load_smart_tagging.load(game, data_dir))
        except IOError as error:
            print("Smart Tagging not available, continuing without it:\n{}".format(error))
    print("restart windows used: {}".format(len(windows)))
    return label_frames(ball, players, team, graded_mask(n_frames, game),
                        GAMES[game]["fps"], windows)


def count_possessions(labels):
    """Maximal runs of one team with the ball alive. D (or X) between two runs splits them."""
    count, previous = 0, None
    for label in labels:
        if label in ("H", "A"):
            if label != previous:
                count += 1
            previous = label
        else:
            previous = None
    return count


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
    print("possessions over the whole match: {}".format(count_possessions(labels)))
    print("-> {}".format(os.path.relpath(write_submission(args.out, labels), ROOT)))
