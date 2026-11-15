"""Possession labels v4: v3 plus stoppages the documentation already tells us about.

Two changes, both from the rules and the provided data rather than from any score:

  1. Dead until kickoff. A half can be graded before the ball is put in play: the 2nd half is
     graded from video frame 110808 but kicks off at 113742, and FAQ.md states those 2934 frames
     are all D. v3 filled them with the first team to gain control, inventing a possession before
     the match restarted. v4 marks everything before the half's tagged kick-off as D.
  2. Set pieces are dead balls. CHALLENGE.md and DATA.md both state that every set piece is a
     restart and every restart ends a dead-ball stretch. v4 marks a set-piece window D from its
     start until the first frame where a team shows control of the ball. If play is already live
     inside the window, that first evidence ends the dead stretch immediately.

Everything else is v3: ball-out-of-play evidence, short-gap interpolation, held possession, and
the control rules from v2. Fixed rules, no learned parameters, nothing fitted.

Every constant is a fixed default chosen from football reasoning and the repository's own
documentation. None was fitted, and none may be tuned on the provided ground truth or the video
(see RULES.md).

    python solution/possession_rules_v4.py              # writes evaluation/results/possession_rules_v4.csv
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
from possession_rules import DEAD, UNKNOWN, read_atd, write_submission  # noqa: E402
import load_smart_tagging  # noqa: E402
import possession_rules_v2 as v2  # noqa: E402
import possession_rules_v3 as v3  # noqa: E402

# Params
KICKOFF_TYPE = "kick off (start)"
PITCH = GAMES[GAME]["pitch"]
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
OUT_NAME = "possession_rules_v4.csv"


def kickoff_windows(events):
    """Video-frame intervals tagged as the start of a half or the restart after a goal is excluded.

    Only 'Kick Off (Start)' -- the two half kickoffs. 'Kick Off (After Goal)' is a set piece in the
    middle of a half and is handled with the other restarts, because play was already live.
    """
    return [(event["start_frame"], event["end_frame"]) for event in events
            if (event["code"] or "").strip().lower() == v2.RESTART_CODE
            and (event["tags"].get("Type") or "").strip().lower() == KICKOFF_TYPE]


def restart_dead_mask(evidence, windows, first, last):
    """Boolean mask over [first..last]: dead from each restart window's start to its first evidence.

    A set piece is a dead ball. It stops being dead the moment a team shows control, which is where
    the restart has been taken and the ball is live again. A window with evidence at its first frame
    contributes nothing.
    """
    mask = np.zeros(last - first + 1, dtype=bool)
    for start, end in windows:
        a, b = max(start, first), min(end + 1, last + 1)
        if b <= a:
            continue
        shown = np.flatnonzero(evidence[a:b] != UNKNOWN)
        stop = a + int(shown[0]) if shown.size else b
        mask[a - first:stop - first] = True
    return mask


def label_block(out, runs, first, last, windows, flags, fps, dead_until):
    """Fill out[first..last] from the controlled runs of one half.

    `dead_until` is the first frame of the half that may carry a possession, which is the tagged
    kickoff. Frames before it are dead whatever the runs say.
    """
    if runs:
        start = max(first, dead_until)
        if runs[0][1] > start:
            out[start:runs[0][1]] = runs[0][0]
        for index, (team, run_start, run_end) in enumerate(runs):
            out[run_start:run_end + 1] = team
            if index + 1 == len(runs):
                out[run_end + 1:last + 1] = team
                continue
            a, b = run_end + 1, runs[index + 1][1]
            hits = np.flatnonzero(flags[a:b])
            out_min = int(round(v3.OUT_MIN_GAP_S * fps))
            dead = v2.gap_is_dead(a, b, windows, fps) or (hits.size > 0 and b - a >= out_min)
            cut = a + int(hits[0]) if (dead and hits.size) else a
            out[a:cut] = team
            out[cut:b] = DEAD if dead else team
    else:
        out[first:last + 1] = DEAD

    if dead_until > first:
        out[first:min(dead_until, last + 1)] = DEAD


def label_frames(ball, players, team, graded, fps, windows=(), kickoffs=(), pitch=PITCH):
    """One label per frame: H, A, D or X. Pure function of its inputs."""
    flags = v3.ball_out_flags(ball, fps, pitch)
    evidence = v2.control_evidence(v3.interpolate_short_gaps(ball, fps), players, team, fps)
    out = np.full(len(ball), v2.X_CODE, dtype=np.int8)
    for first, last in v2.blocks_of(graded):
        dead_until = first
        inside = [start for start, _ in kickoffs if first <= start <= last]
        if inside:
            dead_until = min(inside)
        runs = v2.confirm_runs(evidence[first:last + 1], fps, offset=first)
        label_block(out, runs, first, last, windows, flags, fps, dead_until)
        dead = restart_dead_mask(evidence, windows, first, last)
        block = out[first:last + 1]
        block[dead] = DEAD
        out[first:last + 1] = block
    return v2.TABLE[out]


def build(game=GAME, data_dir=None, use_tagging=True):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    team = team_of_each_track(meta_path)
    ball, players = read_atd(raw_path, n_frames, len(team), GAMES[game]["pitch"])

    windows, kickoffs = [], []
    if use_tagging:
        try:
            events = load_smart_tagging.load(game, data_dir)
            windows = v2.restart_windows(events)
            kickoffs = kickoff_windows(events)
        except IOError as error:
            print("Smart Tagging not available, continuing without it:\n{}".format(error))
    print("restart windows used: {}   kickoffs found: {}".format(len(windows), len(kickoffs)))
    flags = v3.ball_out_flags(ball, GAMES[game]["fps"], GAMES[game]["pitch"])
    print("ball-out frames flagged: {}".format(int(flags.sum())))
    return label_frames(ball, players, team, v2.graded_mask(n_frames, game), GAMES[game]["fps"],
                        windows, kickoffs, GAMES[game]["pitch"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, OUT_NAME))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--no-tagging", action="store_true",
                        help="do not use Smart Tagging restart or kickoff windows")
    args = parser.parse_args()

    labels = build(data_dir=args.data_dir, use_tagging=not args.no_tagging)
    counts = {label: int(np.sum(labels == label)) for label in ("H", "A", "D", "X")}
    print("H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("possessions over the whole match: {}".format(v2.count_possessions(labels)))
    print("-> {}".format(os.path.relpath(write_submission(args.out, labels), ROOT)))
