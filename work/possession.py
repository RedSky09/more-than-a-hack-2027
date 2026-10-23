"""Sticky possession state machine on top of the ATD (v1).

Idea
----
* A team "shows control" on a frame when its nearest player is within R metres of the ball and is
  at least MARGIN metres closer than the other team's nearest player.
* The possession state only changes after the *other* team has shown control on at least K frames
  without the current team showing control in between (and no more than GAP frames apart). The
  switch is back-dated to the first of those frames.
* Frames where the ball is missing never change the state (the state is simply held).
* Dead ball: optional, from the ball being tracked outside the pitch (see dead_mask).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import half_video_blocks  # noqa: E402

L, W = 105.3, 68.0
HOME, AWAY = 0, 1


def team_distances(players, ball, team):
    """(dH, dA): distance in metres from the ball to the nearest player of each team (NaN: none)."""
    bx, by = ball[:, 0] * L, ball[:, 1] * W
    px, py = players[:, :, 0] * L, players[:, :, 1] * W
    dist = np.hypot(px - bx[:, None], py - by[:, None])
    out = []
    for t in (HOME, AWAY):
        sub = np.where(team[None, :] == t, dist, np.inf)
        sub = np.where(np.isnan(sub), np.inf, sub)
        m = sub.min(axis=1)
        m[np.isinf(m)] = np.nan
        out.append(m)
    return out[0], out[1]


def control_evidence(dh, da, radius=3.0, margin=1.0):
    """+1/-1 -> not used; returns int8 array: 0 home shows control, 1 away, -1 nobody."""
    ev = np.full(len(dh), -1, dtype=np.int8)
    with np.errstate(invalid="ignore"):
        home = (dh < radius) & (np.nan_to_num(da, nan=1e9) - dh > margin)
        away = (da < radius) & (np.nan_to_num(dh, nan=1e9) - da > margin)
    ev[home] = HOME
    ev[away] = AWAY
    return ev


def run_state_machine(ev, first, last, k=12, gap=50, dead=None):
    """State per frame in [first, last]; -1 before the first control is seen (back-filled).

    `dead` (bool per video frame) marks stoppages: the state is forced to -1 there and, because a
    restart has a clear possessor, the first control evidence afterwards sets the state at once
    (back-filled to the end of the stoppage) instead of waiting for k frames.
    """
    n = last - first + 1
    state = np.full(n, -1, dtype=np.int8)
    cur = -1
    pending = 0        # opponent-evidence frames since the state team last showed control
    pending_start = -1
    last_opp = -1
    after_dead_start = 0   # first frame of the current alive stretch (for back-filling)
    for i in range(n):
        if dead is not None and dead[first + i]:
            cur = -1
            pending = 0
            pending_start = -1
            after_dead_start = i + 1
            continue
        e = ev[first + i]
        if e == -1:
            state[i] = cur
            continue
        if cur == -1:
            cur = e
            state[after_dead_start: i + 1] = cur
            continue
        if e == cur:
            pending = 0
            pending_start = -1
        else:
            if pending and i - last_opp > gap:
                pending = 0
            if pending == 0:
                pending_start = i
            pending += 1
            last_opp = i
            if pending >= k:
                state[pending_start:i] = e
                cur = e
                pending = 0
                pending_start = -1
        state[i] = cur
    return state


def dead_mask(ball, margin_m=0.5, pad_before=0, pad_after=0):
    """Ball visible and outside the pitch by more than margin_m metres."""
    bx, by = ball[:, 0] * L, ball[:, 1] * W
    with np.errstate(invalid="ignore"):
        out = (bx < -margin_m) | (bx > L + margin_m) | (by < -margin_m) | (by > W + margin_m)
    out = np.nan_to_num(out, nan=0).astype(bool)
    if pad_before or pad_after:
        idx = np.where(out)[0]
        res = out.copy()
        for i in idx:
            res[max(0, i - pad_before): i + pad_after + 1] = True
        return res
    return out


def local_threshold(speed, ratio=0.49, lo=0.7, hi=1.3, window=15000, step=500):
    """Per-frame threshold = ratio x the 75th percentile of player speed in a +/- window frames
    neighbourhood (computed every `step` frames, interpolated), clipped to [lo, hi].
    Ratio 0.49 reproduces 1.3 on the first five minutes, where p75 is 2.66 m/s."""
    n = len(speed)
    centres = np.arange(0, n, step)
    p = np.empty(len(centres))
    for i, c in enumerate(centres):
        x = speed[max(0, c - window): c + window]
        x = x[~np.isnan(x)]
        p[i] = np.percentile(x, 75) if len(x) else np.nan
    ok = ~np.isnan(p)
    return np.clip(ratio * np.interp(np.arange(n), centres[ok], p[ok]), lo, hi)


def speed_dead_mask(players, thr=1.3, win=50, min_len=75, bridge=0, adaptive=None):
    """Stoppage = the 22 players are moving slowly (mean speed over a ~2 s window) for >= min_len
    frames. Slow stretches separated by less than `bridge` frames are merged first."""
    import features as F
    base = F.player_speed(players)
    if adaptive is not None:
        thr = local_threshold(base, **adaptive)
    slow = F.rolling_mean(base, win) < thr
    if bridge:
        slow = F.fill_short_gaps(slow, bridge)
    return F.min_run(slow, min_len)


def build_labels(players, ball, team, radius=3.0, margin=1.0, k=12, gap=50, use_dead=False,
                 dead_kwargs=None, speed_dead=None, merge_max=0):
    """speed_dead: dict(thr=, win=, min_len=) to enable the player-speed stoppage detector."""
    n = len(ball)
    dh, da = team_distances(players, ball, team)
    ev = control_evidence(dh, da, radius, margin)
    labels = np.full(n, "X", dtype="<U1")
    dead = np.zeros(n, bool)
    if use_dead:
        dead |= dead_mask(ball, **(dead_kwargs or {}))
    if speed_dead is not None:
        dead |= speed_dead_mask(players, **speed_dead)
    for _, first, last in half_video_blocks("BAR-ATH"):
        st = run_state_machine(ev, first, last, k, gap, dead if dead.any() else None)
        lab = np.where(st == HOME, "H", np.where(st == AWAY, "A", "X"))
        lab[dead[first:last + 1]] = "D"
        if merge_max:
            lab = merge_sandwiched(lab, merge_max)
        labels[first:last + 1] = lab
    return labels


def merge_sandwiched(labels, max_len):
    """Relabel an H/A spell shorter than max_len frames that sits directly between two spells of
    the other team (no D/X between them). Repeats until stable."""
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


def write_submission(path, labels):
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(f, l) for f, l in enumerate(labels))
