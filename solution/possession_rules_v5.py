"""Possession labels v5: v4 with three robustness corrections to the control rules.

v4 decides possession from a per-frame "who is clearly nearest the ball" test, requires 1.5 s of
evidence before accepting a change, fills ball gaps up to 1 s, and labels the rest from stoppages.
Reviewing it against the definition and against what the tracker is documented to do, three
corrections are justified. None of them is a value search; each removes a mechanism that contradicts
something the repository states.

  1. The per-frame margin was doing the job of the temporal rule. v4 required the nearest opponent
     to be 1.0 m further from the ball than the owner, on every frame. But the definition says a
     change needs *controlled* possession, and the 1.5 s confirmation below is what tests that: a
     deflection or a lost duel cannot last 1.5 s. The per-frame margin therefore only fired in
     crowded play, where it suppressed real control and produced either held possession or D. v5
     keeps a margin as a tie-break against tracker noise (TIE_MARGIN_M) and lets sustained control
     decide the rest. This cannot invent possessions on its own: a brief opponent touch is still
     rejected by the confirmation rule.
  2. A run's length was counted in *evidence frames*, so a team showing one frame of control every
     few seconds for two minutes could confirm as one continuous run spanning those two minutes, and
     the whole span was labelled that team. v5 requires the run to contain a *dense core*: a stretch
     with no internal gap longer than CORE_GAP_S that spans at least MIN_CONTROL_S. The run's extent
     is unchanged, so genuine control interrupted by an occlusion is not lost.
  3. Interpolation could manufacture evidence across a stoppage. v4 filled any ball gap up to 1 s
     with a straight line, so a ball that went out and reappeared 0.8 s later near a player produced
     a path that looked like control over a period when the ball was dead. v5 does not interpolate a
     gap that touches a ball-out frame or a tagged restart/kickoff window.

Everything else is v4: dead until the half's kickoff, set pieces as dead balls, ball-out-of-play
evidence, held possession, and the 3 m control radius from v2.

Every constant is a fixed default chosen from football reasoning, the written definition and the
repository's documentation. None was fitted, and none may be tuned on the provided ground truth or
the video (see RULES.md).

    python solution/possession_rules_v5.py              # writes evaluation/results/possession_rules_v5.csv
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
import possession_rules_v4 as v4  # noqa: E402

# Params
TIE_MARGIN_M = 0.5     # only breaks near-ties between two positional estimates; noise, not a duel test
CORE_GAP_S = 1.0       # the densest part of a run may have no gap longer than this
PITCH = GAMES[GAME]["pitch"]
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
OUT_NAME = "possession_rules_v5.csv"


def control_evidence(ball, players, team, fps):
    """HOME/AWAY where one team's nearest player is nearest the ball by a tie-breaking margin.

    Sustained control is what decides possession -- see confirm_runs. This per-frame test only
    rejects frames with no plausible owner: no ball, nobody in range, a ball travelling too fast to
    be dribbled, a ball at rest, or two players so close to the ball that its own positional noise
    cannot separate them.
    """
    with np.errstate(invalid="ignore"):
        gap = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
        gap = np.where(np.isnan(gap), np.inf, gap)
        home = np.where(team == v2.HOME, gap, np.inf).min(axis=1)
        away = np.where(team == v2.AWAY, gap, np.inf).min(axis=1)
        nearest = np.minimum(home, away)
        other = np.maximum(home, away)
        owner = np.where(home <= away, v2.HOME, v2.AWAY)
        usable = ((nearest <= v2.CONTROL_RADIUS_M)
                  & ((other - nearest) >= TIE_MARGIN_M)
                  & ~(v2.ball_speed(ball, fps) > v2.MAX_CONTROL_BALL_SPEED)
                  & ~v2.ball_at_rest(ball, fps))
    return np.where(usable, owner, UNKNOWN).astype(np.int8)


def dense_core(frames, core, need):
    """Does a group of evidence frames contain a stretch, gaps <= core, spanning >= need frames?"""
    if frames[-1] - frames[0] + 1 < need:
        return False
    cuts = np.flatnonzero(np.diff(frames) > core) + 1
    for start, end in zip(np.concatenate(([0], cuts)), np.concatenate((cuts, [len(frames)]))):
        if frames[end - 1] - frames[start] + 1 >= need:
            return True
    return False


def confirm_runs(codes, fps, offset=0):
    """[(team, first_frame, last_frame)] for runs of control evidence that count as control.

    A run is grouped as before -- same team, no gap longer than LINK_GAP_S -- but it only counts if
    part of it is continuously observed for MIN_CONTROL_S. A team that shows a single frame of
    possession every three seconds for two minutes has not controlled the ball.
    """
    frames = np.flatnonzero(codes != UNKNOWN)
    if frames.size == 0:
        return []
    team = codes[frames]
    link = int(round(v2.LINK_GAP_S * fps))
    core = int(round(CORE_GAP_S * fps))
    need = int(round(v2.MIN_CONTROL_S * fps))
    breaks = np.flatnonzero((team[1:] != team[:-1]) | (np.diff(frames) > link)) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [len(team)]))
    return [(int(team[start]), int(frames[start]) + offset, int(frames[end - 1]) + offset)
            for start, end in zip(starts, ends)
            if dense_core(frames[start:end], core, need)]


def blocked_frames(flags, windows, kickoffs, n_frames):
    """Frames an interpolation must not cross: ball shown out of play, or a tagged stoppage."""
    blocked = flags.copy()
    for start, end in list(windows) + list(kickoffs):
        a, b = max(start, 0), min(end + 1, n_frames)
        if b > a:
            blocked[a:b] = True
    return blocked


def label_frames(ball, players, team, graded, fps, windows=(), kickoffs=(), pitch=PITCH):
    """One label per frame: H, A, D or X. Pure function of its inputs."""
    evidence = evidence_mask(ball, players, team, fps, windows, kickoffs, pitch)
    out = np.full(len(ball), v2.X_CODE, dtype=np.int8)
    for first, last in v2.blocks_of(graded):
        dead_until = first
        inside = [start for start, _ in kickoffs if first <= start <= last]
        if inside:
            dead_until = min(inside)
        runs = confirm_runs(evidence[first:last + 1], fps, offset=first)
        v4.label_block(out, runs, first, last, windows, flags_of(ball, fps, pitch), fps, dead_until)
        dead = v4.restart_dead_mask(evidence, windows, first, last)
        block = out[first:last + 1]
        block[dead] = DEAD
        out[first:last + 1] = block
    return v2.TABLE[out]


def flags_of(ball, fps, pitch=PITCH):
    """Ball-out-of-play frames. Small wrapper so label_frames and the report agree."""
    return v3.ball_out_flags(ball, fps, pitch)


def evidence_mask(ball, players, team, fps, windows=(), kickoffs=(), pitch=PITCH):
    """Per-frame control evidence, with interpolation that will not cross a stoppage."""
    flags = flags_of(ball, fps, pitch)
    blocked = blocked_frames(flags, windows, kickoffs, len(ball))
    return control_evidence(v3.interpolate_short_gaps(ball, fps, blocked), players, team, fps)


def diagnose(ball, players, team, graded, fps, windows=(), kickoffs=(), pitch=PITCH):
    """(frames with evidence, confirmed runs) over the graded blocks, for the run summary."""
    evidence = evidence_mask(ball, players, team, fps, windows, kickoffs, pitch)
    runs = sum(len(confirm_runs(evidence[first:last + 1], fps, offset=first))
               for first, last in v2.blocks_of(graded))
    return int((evidence != UNKNOWN).sum()), runs


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
            kickoffs = v4.kickoff_windows(events)
        except IOError as error:
            print("Smart Tagging not available, continuing without it:\n{}".format(error))

    labels = label_frames(ball, players, team, v2.graded_mask(n_frames, game),
                          GAMES[game]["fps"], windows, kickoffs, GAMES[game]["pitch"])
    evidence_frames, runs = diagnose(ball, players, team, v2.graded_mask(n_frames, game),
                                     GAMES[game]["fps"], windows, kickoffs, GAMES[game]["pitch"])
    print("restart windows used: {}   kickoffs found: {}".format(len(windows), len(kickoffs)))
    print("ball-out frames flagged: {}".format(int(v3.ball_out_flags(
        ball, GAMES[game]["fps"], GAMES[game]["pitch"]).sum())))
    print("frames with control evidence: {} of {}".format(evidence_frames, n_frames))
    print("confirmed control runs: {}".format(runs))
    return labels


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
