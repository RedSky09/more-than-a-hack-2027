"""Possession labels v6: v5 plus the second control proxy CHALLENGE.md offers.

CHALLENGE.md gives two candidate proxies for "controlled" possession:

    - a player keeps the ball for a few seconds
    - a team completes more than two consecutive passes

v5 implements the first: a run of control evidence counts only if part of it is continuously
observed for 1.5 s. That proxy is blind to the way most possession is actually kept, because a pass
travels faster than MAX_CONTROL_BALL_SPEED and therefore produces no control evidence at all. A team
can intercept, complete six passes and gain forty metres while v5 sees nothing and leaves the ball
with the team that lost it. On the public window v5 misses ten of seventeen true possessions, and
this is the likeliest reason.

v6 adds the second proxy as a second source of confirmed runs. It does not add a per-frame test: it
takes the same evidence v5 already computes and aggregates it a second way.

  1. A *control episode* is a maximal stretch of consecutive evidence frames of one team. Inside a
     dribble the evidence is contiguous, so a dribble is one episode. Inside a passing move every
     touch is its own episode, separated from the next by the flight.
  2. A *completed pass* is an interval between two episodes in which the ball was observed
     travelling clear of every player -- at least one frame where it is further than
     CONTROL_RADIUS_M from the nearest player of either team. That distance is what separates a pass
     from a duel or a dribble, where the ball never leaves the players, and requiring the ball to be
     *observed* means a stretch where the tracker simply lost it does not count as one. The interval
     must also fit inside LINK_GAP_S and contain no dead-ball frame. (A player who pushes the ball
     past an opponent and runs onto it is credited the same way as a pass. That is not a pass, but
     the conclusion it supports -- the same team kept the ball -- is the one we want.)
  3. A *run* is a maximal sequence of same-team episodes joined by completed passes, at least four
     episodes long. Four episodes is the interception plus three completed passes, which is
     "more than two consecutive passes" read literally. Such a run is a controlled possession from
     its first episode -- the moment control was gained -- to its last, because the three passes are
     themselves the evidence that the earlier receptions were controlled.

Runs from both proxies are merged, and where two overlap the earlier one owns the frame. The rest is
v5 unchanged: the 0.5 m tie-break margin, the dense core, interpolation that stops at a stoppage,
dead until kickoff, set pieces as dead balls, held possession, ball-out evidence.

Two consequences worth stating in advance. The count is conservative on purpose: a one-touch exchange
that leaves no slow frame at the receiver produces no episode, so a move can be under-counted and no
run credited. And a run is recognised only after its third pass, so possession is moved back to an
interception several seconds after the fact; that is what the proxy costs, and the public row of
CHALLENGE.md accepts it as a description of control rather than a detector of it.

No tuned value: the transfer test reuses CONTROL_RADIUS_M, the interval reuses LINK_GAP_S, and the
four-episode minimum is the definition's own wording.

    python solution/possession_rules_v6.py              # writes evaluation/results/possession_rules_v6.csv
    python evaluation/validate_submission.py <file>      # check the file before upload
"""

import argparse
import os
import sys
from itertools import chain

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
import possession_rules_v5 as v5  # noqa: E402

# Params
PASS_CHAIN_TOUCHES = 4   # "more than two consecutive passes" = 3 transfers = 4 control episodes
PITCH = GAMES[GAME]["pitch"]
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
OUT_NAME = "possession_rules_v6.csv"


def nearest_player_distance(ball, players):
    """Distance from the ball to the nearest player on each frame; NaN where the ball is missing."""
    with np.errstate(invalid="ignore"):
        gap = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
        gap = np.where(np.isnan(gap), np.inf, gap)
        nearest = gap.min(axis=1)
    return np.where(np.isnan(ball[:, 0]), np.nan, nearest)


def clear_frames(ball, players):
    """Frames where the ball was observed travelling clear of every player.

    That is: the ball is tracked, and it is further than CONTROL_RADIUS_M from the nearest player
    of either team. Computing this once for the match turns the per-pass test into index arithmetic.
    """
    distance = nearest_player_distance(ball, players)
    return np.isfinite(distance) & (distance > v2.CONTROL_RADIUS_M)


def episodes_of(evidence, first, last):
    """[(team, start, end)] for each maximal stretch of consecutive same-team evidence frames."""
    block = evidence[first:last + 1]
    frames = np.flatnonzero(block != UNKNOWN)
    if frames.size == 0:
        return []
    team = block[frames]
    cuts = np.flatnonzero((team[1:] != team[:-1]) | (np.diff(frames) > 1)) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [frames.size]))
    return [(int(team[s]), int(frames[s]) + first, int(frames[e - 1]) + first)
            for s, e in zip(starts, ends)]


def transfer(clear, blocked, after, before, fps):
    """Was the ball played from one control episode to the next, i.e. is this a completed pass?

    `after` is the last frame of the earlier episode, `before` the first frame of the later one.
    """
    length = before - after - 1
    if length < 1 or length > int(round(v2.LINK_GAP_S * fps)):
        return False
    if blocked[after + 1:before].any():
        return False
    return bool(clear[after + 1:before].any())


def chains_from(bursts, clear, blocked, fps):
    """[(team, start, end)] for runs of PASS_CHAIN_TOUCHES episodes joined by completed passes."""
    runs = []
    start = 0
    for index in range(1, len(bursts) + 1):
        joined = (index < len(bursts)
                  and bursts[index][0] == bursts[index - 1][0]
                  and transfer(clear, blocked, bursts[index - 1][2], bursts[index][1], fps))
        if joined:
            continue
        if index - start >= PASS_CHAIN_TOUCHES:
            runs.append((bursts[start][0], bursts[start][1], bursts[index - 1][2]))
        start = index
    return runs


def transfers_from(bursts, clear, blocked, fps):
    """How many completed passes the episodes contain. For the run summary only."""
    return sum(1 for earlier, later in zip(bursts[:-1], bursts[1:])
               if earlier[0] == later[0]
               and transfer(clear, blocked, earlier[2], later[1], fps))


def chain_runs(evidence, clear, blocked, fps, first, last):
    """Pass-chain runs inside one graded block."""
    return chains_from(episodes_of(evidence, first, last), clear, blocked, fps)


def merge_runs(*groups):
    """Runs from every rule, earliest first. A run starting inside an earlier one is trimmed."""
    merged = []
    for team, start, end in sorted(chain.from_iterable(groups), key=lambda run: run[1]):
        if merged and start <= merged[-1][2]:
            start = merged[-1][2] + 1
        if start <= end:
            merged.append((team, start, end))
    return merged


def prepared(ball, players, team, fps, windows=(), kickoffs=(), pitch=PITCH):
    """(flags, blocked, evidence, clear) -- the arrays both the labels and the report need."""
    flags = v5.flags_of(ball, fps, pitch)
    blocked = v5.blocked_frames(flags, windows, kickoffs, len(ball))
    filled = v3.interpolate_short_gaps(ball, fps, blocked)
    evidence = v5.control_evidence(filled, players, team, fps)
    return flags, blocked, evidence, clear_frames(ball, players)


def runs_of(evidence, clear, blocked, fps, first, last):
    """Confirmed runs in one graded block, from both control proxies."""
    core = v5.confirm_runs(evidence[first:last + 1], fps, offset=first)
    return merge_runs(core, chain_runs(evidence, clear, blocked, fps, first, last))


def label_frames(ball, players, team, graded, fps, windows=(), kickoffs=(), pitch=PITCH):
    """One label per frame: H, A, D or X. Pure function of its inputs."""
    flags, blocked, evidence, clear = prepared(ball, players, team, fps, windows, kickoffs, pitch)
    out = np.full(len(ball), v2.X_CODE, dtype=np.int8)
    for first, last in v2.blocks_of(graded):
        dead_until = first
        inside = [start for start, _ in kickoffs if first <= start <= last]
        if inside:
            dead_until = min(inside)
        runs = runs_of(evidence, clear, blocked, fps, first, last)
        v4.label_block(out, runs, first, last, windows, flags, fps, dead_until)
        dead = v4.restart_dead_mask(evidence, windows, first, last)
        block = out[first:last + 1]
        block[dead] = DEAD
        out[first:last + 1] = block
    return v2.TABLE[out]


def diagnose(ball, players, team, graded, fps, windows=(), kickoffs=(), pitch=PITCH):
    """Counts for the run summary: what each proxy contributed before and after merging."""
    _, blocked, evidence, clear = prepared(ball, players, team, fps, windows, kickoffs, pitch)
    report = {"evidence": int((evidence != UNKNOWN).sum()), "episodes": 0, "transfers": 0,
              "core": 0, "chains": 0, "merged": 0}
    for first, last in v2.blocks_of(graded):
        bursts = episodes_of(evidence, first, last)
        report["episodes"] += len(bursts)
        report["transfers"] += transfers_from(bursts, clear, blocked, fps)
        chains = chains_from(bursts, clear, blocked, fps)
        report["chains"] += len(chains)
        report["core"] += len(v5.confirm_runs(evidence[first:last + 1], fps, offset=first))
        report["merged"] += len(merge_runs(
            v5.confirm_runs(evidence[first:last + 1], fps, offset=first), chains))
    return report


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

    graded = v2.graded_mask(n_frames, game)
    labels = label_frames(ball, players, team, graded, GAMES[game]["fps"], windows, kickoffs,
                          GAMES[game]["pitch"])
    report = diagnose(ball, players, team, graded, GAMES[game]["fps"], windows, kickoffs,
                      GAMES[game]["pitch"])
    print("restart windows used: {}   kickoffs found: {}".format(len(windows), len(kickoffs)))
    print("ball-out frames flagged: {}".format(int(v3.ball_out_flags(
        ball, GAMES[game]["fps"], GAMES[game]["pitch"]).sum())))
    print("frames with control evidence: {} of {}".format(report["evidence"], n_frames))
    print("control episodes: {}   completed passes between them: {}".format(
        report["episodes"], report["transfers"]))
    print("dense-core runs: {}   pass-chain runs: {}   after merging: {}".format(
        report["core"], report["chains"], report["merged"]))
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
