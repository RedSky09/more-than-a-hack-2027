"""Synthetic checks for possession_rules_v5.py. No match data is read.

    python solution/test_possession_rules_v5.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import possession_rules_v2 as v2  # noqa: E402
import possession_rules_v3 as v3  # noqa: E402
import possession_rules_v4 as v4  # noqa: E402
import possession_rules_v5 as pr  # noqa: E402
from test_possession_rules_v2 import AWAY_AT, HOME_AT, TEAM, dribble, scene  # noqa: E402

FPS = 25


def run(ball, players, graded=None, windows=(), kickoffs=()):
    graded = np.ones(len(ball), dtype=bool) if graded is None else graded
    return "".join(pr.label_frames(ball, players, TEAM, graded, FPS, windows, kickoffs))


def evidence(ball, players, windows=(), kickoffs=()):
    return pr.evidence_mask(ball, players, TEAM, FPS, windows, kickoffs)


# -- item 2: the per-frame margin is a tie-break, not a duel test --------------------------------

def test_crowded_frame_no_longer_becomes_dead_ball():
    """The ball is carried with an opponent 0.8 m away: v4 demands 1.0 m, so it finds nothing at all."""
    ball = np.full((300, 2), np.nan, dtype=np.float32)
    ball[:, 0] = 20.0 + 0.08 * np.arange(300)     # slow enough to be a carry, not a pass
    ball[:, 1] = 20.0
    players = np.zeros((300, 4, 2), dtype=np.float32)
    players[:, 0] = ball                          # the owner is on the ball
    players[:, 1] = (5, 5)
    players[:, 2] = ball + (0.8, 0)               # a challenger inside v4's margin
    players[:, 3] = (50, 5)
    assert (v2.control_evidence(ball, players, TEAM, FPS) == v2.UNKNOWN).all()
    assert set(run(ball, players)) == {"H"}, "the ball is being carried, so it is not dead"
    assert set("".join(v4.label_frames(ball, players, TEAM, np.ones(300, bool), FPS))) == {"D"}


def test_margin_does_not_flip_a_clear_owner():
    """A challenger further than the tie margin does not take the ball."""
    ball, players = scene(200)
    dribble(ball, 0, 200, HOME_AT)
    players[:, 2] = (12.6, 10)
    evidence = pr.evidence_mask(ball, players, TEAM, FPS)
    assert (evidence == v2.HOME).all(), "the nearest player is the owner, and is well clear"


def test_near_tie_frames_give_no_evidence():
    """Where the two players are closer together than the tie margin, the frame stays open."""
    ball, players = scene(200)
    dribble(ball, 0, 200, HOME_AT)
    players[:, 2] = (12.0, 10)
    ev = evidence(ball, players)
    peak = int(np.argmax(ball[:, 0]))       # the ball is exactly as near the opponent as the owner
    assert abs(ball[peak, 0] - 11.0) < 0.05
    assert ev[peak] == v2.UNKNOWN, "positional noise cannot separate them"
    assert (ev == v2.HOME).any(), "away from the tie the owner is still found"


def test_nobody_near_the_ball_is_no_evidence():
    ball, players = scene(200)
    ball[:] = (30, 40)
    assert (evidence(ball, players) == v2.UNKNOWN).all()


def test_brief_opponent_touch_still_is_not_a_change():
    """The temporal rule, not the margin, is what rejects a deflection."""
    ball, players = scene(400)
    dribble(ball, 0, 100, HOME_AT)
    dribble(ball, 150, 162, AWAY_AT)
    dribble(ball, 200, 300, HOME_AT)
    assert run(ball, players) == "H" * 400


# -- item 3: a run needs a dense core -----------------------------------------------------------

def test_sparse_evidence_does_not_confirm():
    codes = np.full(3000, v2.UNKNOWN, dtype=np.int8)
    codes[::50] = v2.HOME
    assert pr.confirm_runs(codes, FPS) == [], "one frame every 2 s is not two minutes of control"


def test_dense_core_confirms_the_whole_run():
    codes = np.full(300, v2.UNKNOWN, dtype=np.int8)
    codes[:60] = v2.HOME
    codes[100:110] = v2.HOME        # 1.6 s later, still inside the 3 s link
    assert pr.confirm_runs(codes, FPS) == [(v2.HOME, 0, 109)], "the run's extent is unchanged"


def test_dense_core_survives_one_occlusion():
    codes = np.full(300, v2.UNKNOWN, dtype=np.int8)
    codes[:40] = v2.HOME
    codes[40:60] = v2.UNKNOWN       # 0.8 s with no ball
    codes[60:150] = v2.HOME
    assert pr.confirm_runs(codes, FPS) == [(v2.HOME, 0, 149)]


def test_short_dense_burst_below_minimum_does_not_confirm():
    codes = np.full(300, v2.UNKNOWN, dtype=np.int8)
    codes[100:120] = v2.HOME        # 0.8 s of control
    assert pr.confirm_runs(codes, FPS) == []


# -- item 4: no interpolation across a stoppage --------------------------------------------------

def test_interpolation_skips_blocked_gaps():
    ball = np.full((100, 2), np.nan, dtype=np.float32)
    ball[0] = (10, 10)
    ball[3] = (10.3, 10)
    assert not np.isnan(v3.interpolate_short_gaps(ball, FPS)[1, 0])
    blocked = np.zeros(100, dtype=bool)
    blocked[2] = True
    assert np.isnan(v3.interpolate_short_gaps(ball, FPS, blocked)[1, 0]), "a stoppage is not crossed"


def test_blocked_frames_covers_windows_and_flags():
    flags = np.zeros(100, dtype=bool)
    flags[50] = True
    blocked = pr.blocked_frames(flags, [(10, 20)], [(80, 85)], 100)
    assert blocked[50] and blocked[10:21].all() and blocked[80:86].all()
    assert not blocked[30] and not blocked[95]


def test_ball_out_gap_is_not_bridged_into_evidence():
    """The ball leaves the pitch, and reappears 0.8 s later near the home player."""
    ball, players = scene(200)
    dribble(ball, 0, 40, HOME_AT)
    ball[40:70] = (10, -1.0)            # out of play for 1.2 s
    ball[70:78] = np.nan                # missing for 0.32 s
    dribble(ball, 78, 200, HOME_AT)
    flags = pr.flags_of(ball, FPS)
    assert flags[45:70].any(), "the out-of-play stretch is detected"
    filled = v3.interpolate_short_gaps(ball, FPS, pr.blocked_frames(flags, [], [], 200))
    assert np.isnan(filled[70:78]).all(), "the gap touching the ball-out frames is left missing"


# -- unchanged behaviour -------------------------------------------------------------------------

def test_kickoff_and_set_piece_rules_are_inherited():
    ball, players = scene(500)
    dribble(ball, 320, 420, HOME_AT)
    out = run(ball, players, kickoffs=[(300, 310)])
    assert out[:300] == "D" * 300 and set(out[320:420]) == {"H"}

    ball, players = scene(400)
    dribble(ball, 0, 100, HOME_AT)
    dribble(ball, 200, 300, HOME_AT)
    assert v2.count_possessions(run(ball, players, windows=[(120, 140)])) == 2


def test_ungraded_is_x_and_alphabet_is_valid():
    ball, players = scene(400)
    dribble(ball, 0, 100, HOME_AT)
    graded = np.zeros(400, dtype=bool)
    graded[:250] = True
    out = run(ball, players, graded, windows=[(200, 220)])
    assert out[250:] == "X" * 150 and len(out) == 400 and set(out) <= set("HADX")
    assert out[200:220] == "D" * 20


def test_own_possession_is_held_across_a_long_ball_absence():
    ball, players = scene(600)
    dribble(ball, 0, 100, HOME_AT)
    out = run(ball, players)
    assert set(out) == {"H"}, "the ball is lost for 20 s with no restart, so last owner keeps it"


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok  {}".format(test.__name__))
