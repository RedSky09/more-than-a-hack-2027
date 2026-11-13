"""Synthetic checks for possession_rules_v2.py. No match data is read.

    python solution/test_possession_rules_v2.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import possession_rules as v1  # noqa: E402
import possession_rules_v2 as pr  # noqa: E402

FPS = 25
TEAM = np.array([pr.HOME, pr.HOME, pr.AWAY, pr.AWAY], dtype=np.int8)
HOME_AT = (10.0, 10.0)
AWAY_AT = (50.0, 10.0)


def scene(n):
    """Two home players around (10, 10), two away players around (50, 10). Ball missing."""
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    players = np.zeros((n, 4, 2), dtype=np.float32)
    players[:, 0] = (10, 10)
    players[:, 1] = (20, 30)
    players[:, 2] = (50, 10)
    players[:, 3] = (40, 30)
    return ball, players


def dribble(ball, start, end, point):
    """Ball moves slowly near `point`, so it is controlled and not at rest."""
    steps = np.arange(end - start)
    ball[start:end, 0] = point[0] + 1.0 * np.sin(0.1 * steps)
    ball[start:end, 1] = point[1] + 0.3 * np.cos(0.1 * steps)


def run(ball, players, graded=None, windows=()):
    graded = np.ones(len(ball), dtype=bool) if graded is None else graded
    return "".join(pr.label_frames(ball, players, TEAM, graded, FPS, windows))


def test_ball_missing_keeps_last_owner():
    ball, players = scene(600)
    dribble(ball, 0, 100, HOME_AT)
    out = run(ball, players)
    assert set(out) == {"H"}, "a ball lost for 20 s with no restart stays with the last owner"


def test_loose_ball_far_from_everyone_keeps_owner():
    ball, players = scene(400)
    dribble(ball, 0, 100, HOME_AT)
    ball[100:200] = np.linspace((10, 10), (30, 50), 100)
    out = run(ball, players)
    assert "D" not in out[:200] and set(out[:200]) == {"H"}


def test_touch_is_not_control():
    ball, players = scene(400)
    dribble(ball, 0, 100, HOME_AT)
    dribble(ball, 150, 150 + 10, AWAY_AT)
    dribble(ball, 200, 300, HOME_AT)
    assert run(ball, players) == "H" * 400


def test_opponent_control_changes_possession():
    ball, players = scene(400)
    dribble(ball, 0, 100, HOME_AT)
    dribble(ball, 200, 300, AWAY_AT)
    out = run(ball, players)
    assert out[:100] == "H" * 100
    assert out[200:300] == "A" * 100
    assert out[100:200] in {"H" * 100}, "held until the opponent's run starts"
    assert set(out[300:]) == {"A"}


def test_long_gap_is_dead_ball_and_splits_possession():
    ball, players = scene(900)
    dribble(ball, 0, 100, HOME_AT)
    dribble(ball, 800, 900, HOME_AT)
    out = run(ball, players)
    assert out[300:700] == "D" * 400
    assert pr.count_possessions(out) == 2, "H D H is two possessions"


def test_short_gap_with_restart_tag_is_dead():
    ball, players = scene(400)
    dribble(ball, 0, 100, HOME_AT)
    dribble(ball, 220, 320, HOME_AT)
    plain = run(ball, players)
    tagged = run(ball, players, windows=[(130, 200)])
    assert set(plain) == {"H"}
    assert "D" in tagged and pr.count_possessions(tagged) == 2


def test_ball_at_rest_is_no_evidence():
    ball, players = scene(300)
    ball[:] = (10.5, 10.0)
    evidence = pr.control_evidence(ball, players, TEAM, FPS)
    assert (evidence == v1.UNKNOWN).all()


def test_fast_ball_is_no_evidence():
    ball, players = scene(100)
    ball[:, 0] = 10 + 0.6 * np.arange(100)
    ball[:, 1] = 10.0
    players[:, 0] = ball
    evidence = pr.control_evidence(ball, players, TEAM, FPS)
    assert (evidence[10:90] == v1.UNKNOWN).all(), "15 m/s is a pass, not a dribble"


def test_contested_ball_is_no_evidence():
    ball, players = scene(100)
    dribble(ball, 0, 100, HOME_AT)
    players[:, 2] = (11, 10)
    evidence = pr.control_evidence(ball, players, TEAM, FPS)
    assert (evidence == v1.UNKNOWN).all(), "opponent as close as the owner: no clear control"


def test_ungraded_is_x_and_halves_are_independent():
    ball, players = scene(1000)
    dribble(ball, 0, 100, HOME_AT)
    dribble(ball, 600, 700, AWAY_AT)
    graded = np.zeros(1000, dtype=bool)
    graded[:300] = True
    graded[500:] = True
    out = run(ball, players, graded)
    assert out[300:500] == "X" * 200
    assert set(out[:300]) == {"H"}
    assert set(out[500:]) == {"A"}, "no leaking of the first half's owner into the second"


def test_no_evidence_in_a_half_is_dead():
    ball, players = scene(200)
    assert run(ball, players) == "D" * 200


def test_restart_windows_filter():
    events = [
        {"code": "SET PIECES", "start_frame": 10, "end_frame": 20, "tags": {"Type": "Throw-in"}},
        {"code": "SET PIECES", "start_frame": 30, "end_frame": 40,
         "tags": {"Type": "Kick Off (Start)"}},
        {"code": "BUILD UP", "start_frame": 50, "end_frame": 60, "tags": {"Type": "Throw-in"}},
    ]
    assert pr.restart_windows(events) == [(10, 20)]


def test_output_alphabet_and_length():
    ball, players = scene(500)
    dribble(ball, 0, 100, HOME_AT)
    out = run(ball, players)
    assert len(out) == 500 and set(out) <= set("HADX")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok  {}".format(test.__name__))
