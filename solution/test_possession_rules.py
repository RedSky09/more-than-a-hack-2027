"""Synthetic checks for possession_rules.py. No match data is read.

    python solution/test_possession_rules.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import possession_rules as pr  # noqa: E402

FPS = 25
TEAM = np.array([pr.HOME, pr.AWAY], dtype=np.int8)


def scene(n):
    """Home player at (10, 10), away player at (50, 10), ball missing everywhere."""
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    players = np.zeros((n, 2, 2), dtype=np.float32)
    players[:, 0] = (10, 10)
    players[:, 1] = (50, 10)
    return ball, players


def run(ball, players, graded=None):
    graded = np.ones(len(ball), dtype=bool) if graded is None else graded
    return "".join(pr.label_frames(ball, players, TEAM, graded, FPS))


def test_owner_and_dead_ball():
    ball, players = scene(300)
    ball[:100] = (10.5, 10)
    ball[200:] = (49.5, 10)
    out = run(ball, players)
    assert out[:100] == "H" * 100
    assert out[100:200] == "D" * 100, "a 4 s gap is dead ball"
    assert out[200:] == "A" * 100


def test_short_gap_bridged():
    ball, players = scene(200)
    ball[:] = (10.5, 10)
    ball[90:110] = np.nan
    assert run(ball, players) == "H" * 200


def test_short_gap_between_teams_splits():
    ball, players = scene(200)
    ball[:90] = (10.5, 10)
    ball[110:] = (49.5, 10)
    assert run(ball, players) == "H" * 100 + "A" * 100


def test_flicker_absorbed():
    ball, players = scene(300)
    ball[:] = (10.5, 10)
    ball[150:160] = (49.5, 10)
    assert run(ball, players) == "H" * 300


def test_ball_far_from_everyone_is_not_owned():
    ball, players = scene(200)
    ball[:] = (30, 40)
    assert run(ball, players) == "D" * 200


def test_ungraded_is_x():
    ball, players = scene(100)
    ball[:] = (10.5, 10)
    graded = np.zeros(100, dtype=bool)
    graded[50:] = True
    assert run(ball, players, graded) == "X" * 50 + "H" * 50


def test_submission_shape(tmp="submission_test.csv"):
    path = os.path.join(os.environ.get("TEMP", "."), tmp)
    pr.write_submission(path, np.array(list("XHAD")))
    with open(path) as handle:
        assert handle.read() == "frame,label\n0,X\n1,H\n2,A\n3,D\n"
    os.remove(path)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok  {}".format(test.__name__))
