"""Regenerate the possession submission from the raw tracking (ATD) alone.

    python solution/make_submission.py                       # writes evaluation/results/submission.csv
    python solution/make_submission.py --out my.csv

Needs numpy and the tactical ATD in ./data (or MTAH_DATA_DIR). No ground truth is read, no hand
labels are used, nothing is trained: every label comes out of the rules below.

Method
------
1. Control evidence. For each frame where the ball is tracked, find the nearest player of each
   team. A team "shows control" when its nearest player is within RADIUS metres of the ball and at
   least MARGIN metres closer than the other team's nearest player.
2. Stoppages (D), two independent cues, either one is enough:
     a. the ball is tracked outside the pitch (> OFFPITCH_M beyond a line);
     b. the 22 players are moving slowly -- the mean displacement speed of all visible players,
        averaged over ~2 s, stays under SLOW_SPEED for at least MIN_STOP frames. Slow stretches
        separated by less than BRIDGE frames are merged first.
   Cue (b) does not need the ball, which is missing on ~41% of graded frames.
3. Possession (H/A) inside each half, between stoppages. The state is sticky: it changes only
   after the *other* team has shown control on SWITCH_FRAMES frames (not more than GAP apart)
   without the current team showing control in between, and the change is back-dated to the first
   of those frames. Frames without a ball never change the state. After a stoppage the first
   control evidence sets the possessor at once -- a restart always has a clear possessor.
4. A spell of fewer than MERGE_FRAMES frames sitting directly between two spells of the other team
   is absorbed into them (a touch without control is not a change of possession).
5. Frames outside the two graded halves are X.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

PITCH_L, PITCH_W = GAMES[GAME]["pitch"]
FPS = 25
HOME, AWAY, NOBODY = 0, 1, -1

# Parameters. They come from tracking statistics and a handful of public-window checks, not from
# a search against the public score -- see README_solution.md for how each was chosen.
RADIUS = 3.0          # m, nearest player must be this close to count as control
MARGIN = 1.0          # m, and this much closer than the opposing team's nearest player
SWITCH_FRAMES = 18    # frames of opposing control needed to hand over possession
GAP = 50              # frames, opposing evidence further apart than this does not accumulate
OFFPITCH_M = 0.5      # m beyond the touchlines / goal lines
SLOW_SPEED = 1.3      # m/s, mean player speed under this means the game is stopped
SPEED_WINDOW = 50     # frames, smoothing window of that speed (~2 s)
SPEED_HALF = 12       # frames, speed is displacement over +/- this many frames
MIN_STOP = 100        # frames (4 s), shortest stoppage
BRIDGE = 75           # frames (3 s), slow stretches closer than this are one stoppage
MERGE_FRAMES = 75     # frames (3 s), sandwiched spells shorter than this are absorbed


# --------------------------------------------------------------------------------------------
# Reading the ATD
# --------------------------------------------------------------------------------------------
def team_of_each_track(meta_path):
    """0 for a home slot, 1 for an away slot (first team in the metadata is home)."""
    import xml.etree.ElementTree as ET
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    slots = [p.get("teamId") for p in root.iter("Player")]
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)


def read_atd(meta_path, raw_path, n_frames):
    """players (N, slots, 3) x/y normalised + speed, ball (N, 2); NaN where missing."""
    team = team_of_each_track(meta_path)
    players = np.full((n_frames, len(team), 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1                      # frame_id counts from 1
            groups = parts[1].split(";")
            players[i] = [[float(v) for v in g.split(",")] for g in groups]
            ball[i] = [float(v) for v in parts[2].split(",")[:2]]
    return players, ball, team


# --------------------------------------------------------------------------------------------
# Small array helpers
# --------------------------------------------------------------------------------------------
def rolling_mean(x, window):
    """Centred rolling mean ignoring NaN; `window` is the total width in frames."""
    x = np.asarray(x, float)
    valid = ~np.isnan(x)
    cs = np.concatenate(([0], np.cumsum(np.where(valid, x, 0.0))))
    cn = np.concatenate(([0], np.cumsum(valid)))
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - window // 2)
    hi = np.minimum(len(x), idx + window // 2 + 1)
    num, den = cs[hi] - cs[lo], cn[hi] - cn[lo]
    out = np.full(len(x), np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def drop_short_runs(mask, min_len):
    """Remove True-runs shorter than min_len."""
    mask = np.asarray(mask, bool)
    out = mask.copy()
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    for s, e in zip(edges[::2], edges[1::2]):
        if e - s < min_len:
            out[s:e] = False
    return out


# --------------------------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------------------------
def nearest_distance_per_team(players, ball, team):
    """(home, away): metres from the ball to that team's nearest tracked player (NaN: none)."""
    bx, by = ball[:, 0] * PITCH_L, ball[:, 1] * PITCH_W
    dist = np.hypot(players[:, :, 0] * PITCH_L - bx[:, None],
                    players[:, :, 1] * PITCH_W - by[:, None])
    out = []
    for t in (HOME, AWAY):
        sub = np.where(team[None, :] == t, dist, np.inf)
        sub = np.where(np.isnan(sub), np.inf, sub)
        m = sub.min(axis=1)
        m[np.isinf(m)] = np.nan
        out.append(m)
    return out


def control_evidence(dh, da):
    """Per frame: HOME / AWAY if that team shows control of the ball, NOBODY otherwise."""
    ev = np.full(len(dh), NOBODY, dtype=np.int8)
    with np.errstate(invalid="ignore"):
        home = (dh < RADIUS) & (np.nan_to_num(da, nan=1e9) - dh > MARGIN)
        away = (da < RADIUS) & (np.nan_to_num(dh, nan=1e9) - da > MARGIN)
    ev[home] = HOME
    ev[away] = AWAY
    return ev


def mean_player_speed(players):
    """Mean over visible players of displacement speed (m/s) over +/- SPEED_HALF frames."""
    h = SPEED_HALF
    px, py = players[:, :, 0] * PITCH_L, players[:, :, 1] * PITCH_W
    dx, dy = np.full_like(px, np.nan), np.full_like(py, np.nan)
    dx[h:-h] = px[2 * h:] - px[:-2 * h]
    dy[h:-h] = py[2 * h:] - py[:-2 * h]
    speed = np.hypot(dx, dy) / (2 * h / FPS)
    valid = ~np.isnan(speed)
    count = valid.sum(axis=1)
    out = np.full(len(px), np.nan)
    ok = count > 0
    out[ok] = np.where(valid, speed, 0).sum(axis=1)[ok] / count[ok]
    return out


def stoppage_mask(players, ball):
    """True where the ball is dead: ball off the pitch, or everybody slow for long enough."""
    bx, by = ball[:, 0] * PITCH_L, ball[:, 1] * PITCH_W
    with np.errstate(invalid="ignore"):
        off = ((bx < -OFFPITCH_M) | (bx > PITCH_L + OFFPITCH_M)
               | (by < -OFFPITCH_M) | (by > PITCH_W + OFFPITCH_M))
    off = np.nan_to_num(off, nan=0).astype(bool)

    slow = rolling_mean(mean_player_speed(players), SPEED_WINDOW) < SLOW_SPEED
    slow = ~drop_short_runs(~slow, BRIDGE)          # merge slow stretches closer than BRIDGE
    slow = drop_short_runs(slow, MIN_STOP)
    return off | slow


# --------------------------------------------------------------------------------------------
# Possession
# --------------------------------------------------------------------------------------------
def possession_states(ev, dead, first, last):
    """Sticky possession state for frames first..last (NOBODY on stoppages / before any control)."""
    n = last - first + 1
    state = np.full(n, NOBODY, dtype=np.int8)
    cur = NOBODY
    pending, pending_start, last_opp = 0, -1, -1
    alive_from = 0
    for i in range(n):
        if dead[first + i]:
            cur, pending, pending_start, alive_from = NOBODY, 0, -1, i + 1
            continue
        e = ev[first + i]
        if e == NOBODY:
            state[i] = cur
            continue
        if cur == NOBODY:                           # first control after a stoppage / kick-off
            cur = e
            state[alive_from:i + 1] = cur
            continue
        if e == cur:
            pending, pending_start = 0, -1
        else:
            if pending and i - last_opp > GAP:
                pending = 0
            if pending == 0:
                pending_start = i
            pending += 1
            last_opp = i
            if pending >= SWITCH_FRAMES:
                state[pending_start:i] = e          # back-date the hand-over
                cur, pending, pending_start = e, 0, -1
        state[i] = cur
    return state


def absorb_sandwiched(labels, max_len):
    """Relabel an H/A spell shorter than max_len that sits directly between two spells of the
    other team. Repeats until stable."""
    labels = labels.copy()
    changed = True
    while changed:
        changed = False
        edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
        starts = np.concatenate(([0], edges))
        ends = np.concatenate((edges, [len(labels)]))
        for i in range(1, len(starts) - 1):
            cur, prev, nxt = labels[starts[i]], labels[starts[i - 1]], labels[starts[i + 1]]
            if (cur in "HA" and prev in "HA" and prev == nxt and prev != cur
                    and ends[i] - starts[i] < max_len):
                labels[starts[i]:ends[i]] = prev
                changed = True
                break
    return labels


def predict(players, ball, team):
    """One label per video frame: H / A / D, or X outside the graded halves."""
    dh, da = nearest_distance_per_team(players, ball, team)
    ev = control_evidence(dh, da)
    dead = stoppage_mask(players, ball)

    labels = np.full(len(ball), "X", dtype="<U1")
    for _, first, last in half_video_blocks(GAME):
        state = possession_states(ev, dead, first, last)
        half = np.where(state == HOME, "H", np.where(state == AWAY, "A", "X"))
        half[dead[first:last + 1]] = "D"
        labels[first:last + 1] = absorb_sandwiched(half, MERGE_FRAMES)
    return labels


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(f, l) for f, l in enumerate(labels))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results",
                                                      "submission.csv"))
    args = parser.parse_args()

    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    print("reading {} ...".format(os.path.basename(raw_path)))
    players, ball, team = read_atd(meta_path, raw_path, source_frames())
    labels = predict(players, ball, team)
    write_submission(args.out, labels)
    counts = {c: int(np.sum(labels == c)) for c in "HADX"}
    print("wrote {}  {}".format(os.path.relpath(args.out, ROOT), counts))


if __name__ == "__main__":
    main()
