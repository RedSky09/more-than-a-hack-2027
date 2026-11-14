"""Synthetic checks for possession_rules_v3.py. No match data is read.

    python solution/test_possession_rules_v3.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import possession_rules_v2 as v2  # noqa: E402
import possession_rules_v3 as pr  # noqa: E402
from test_possession_rules_v2 import AWAY_AT, HOME_AT, TEAM, dribble, scene  # noqa: E402

FPS = 25


def run(ball, players, graded=None, windows=()):
    graded = np.ones(len(ball), dtype=bool) if graded is None else graded
    return "".join(pr.label_frames(ball, players, TEAM, graded, FPS, windows))


def test_interpolates_short_plausible_gap():
    ball = np.full((100, 2), np.nan, dtype=np.float32)
    ball[:40, 0] = np.arange(40) * 0.1
    ball[60:, 0] = 6.0 + np.arange(40) * 0.1
    ball[:, 1] = 20.0
    ball[40:60, 1] = np.nan
    filled = pr.interpolate_short_gaps(ball, FPS)
    assert not np.isnan(filled[:, 0]).any()
    assert abs(filled[50, 0] - 5.0) < 0.05, "3.9 m at frame 39 to 6.0 m at frame 60"


def test_does_not_interpolate_long_or_impossible_gaps():
    ball = np.full((200, 2), np.nan, dtype=np.float32)
    ball[0] = (10, 10)
    ball[60] = (10.5, 10)
    assert np.isnan(pr.interpolate_short_gaps(ball, FPS)[30, 0]), "60 frames is longer than 1 s"
    ball = np.full((200, 2), np.nan, dtype=np.float32)
    ball[0] = (10, 10)
    ball[10] = (60, 10)
    assert np.isnan(pr.interpolate_short_gaps(ball, FPS)[5, 0]), "50 m in 0.4 s is not a ball"


def test_flags_ball_beyond_a_line():
    ball, _ = scene(60)
    ball[:] = (30, 30)
    ball[20:30, 1] = -1.0
    flags = pr.ball_out_flags(ball, FPS)
    assert flags[22:28].all() and not flags[:15].any() and not flags[40:].any()


def test_single_noisy_frame_is_not_out():
    ball, _ = scene(60)
    ball[:] = (30, 30)
    ball[20, 1] = -1.0
    assert not pr.ball_out_flags(ball, FPS).any()


def test_flags_ball_that_vanishes_on_a_line():
    ball, _ = scene(200)
    ball[:50] = (30, 0.8)
    ball[150:] = (30, 30)
    flags = pr.ball_out_flags(ball, FPS)
    assert flags[49] and flags.sum() == 1


def test_ball_vanishing_mid_pitch_is_not_out():
    ball, _ = scene(200)
    ball[:50] = (30, 30)
    ball[150:] = (30, 30)
    assert not pr.ball_out_flags(ball, FPS).any()


def test_ball_seen_leaving_makes_dead_from_that_moment():
    ball, players = scene(400)
    dribble(ball, 0, 100, HOME_AT)
    ball[100:130] = np.linspace(HOME_AT, (10, -1.0), 30)
    ball[130:140] = (10, -1.0)
    dribble(ball, 200, 300, HOME_AT)
    out = run(ball, players)
    assert out[:120] == "H" * 120, "alive until the ball crosses the line"
    assert out[140:200] == "D" * 60
    assert v2.count_possessions(out) == 2
    assert set(run(ball.copy(), players, windows=())[:100]) == {"H"}


def test_ball_vanishing_on_the_line_makes_dead():
    ball, players = scene(400)
    dribble(ball, 0, 100, HOME_AT)
    ball[100:120] = np.linspace(HOME_AT, (10, 0.8), 20)
    dribble(ball, 200, 300, HOME_AT)
    out = run(ball, players)
    assert out[:100] == "H" * 100
    assert out[125:200] == "D" * 75
    assert v2.count_possessions(out) == 2


def test_mid_pitch_occlusion_still_held():
    ball, players = scene(400)
    dribble(ball, 0, 100, HOME_AT)
    ball[100:120] = np.linspace(HOME_AT, (30, 34), 20)
    dribble(ball, 200, 300, HOME_AT)
    assert set(run(ball, players)) == {"H"}


def test_short_stretch_with_out_flag_is_not_dead():
    ball, players = scene(300)
    dribble(ball, 0, 100, HOME_AT)
    ball[105:110] = (10, -1.0)
    dribble(ball, 140, 240, HOME_AT)
    assert set(run(ball, players)) == {"H"}, "a 1.6 s stretch is too short to call a stoppage"


def test_possession_changes_after_ball_out():
    ball, players = scene(500)
    dribble(ball, 0, 100, HOME_AT)
    ball[100:130] = np.linspace(HOME_AT, (10, -1.0), 30)
    ball[130:140] = (10, -1.0)
    dribble(ball, 250, 350, AWAY_AT)
    out = run(ball, players)
    assert out[:120] == "H" * 120
    assert out[150:250] == "D" * 100
    assert out[250:350] == "A" * 100


def test_ungraded_stays_x_and_alphabet_is_valid():
    ball, players = scene(600)
    dribble(ball, 0, 100, HOME_AT)
    graded = np.zeros(600, dtype=bool)
    graded[:300] = True
    out = run(ball, players, graded)
    assert out[300:] == "X" * 300 and len(out) == 600 and set(out) <= set("HADX")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok  {}".format(test.__name__))
