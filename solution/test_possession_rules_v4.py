"""Synthetic checks for possession_rules_v4.py. No match data is read.

    python solution/test_possession_rules_v4.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import possession_rules_v2 as v2  # noqa: E402
import possession_rules_v4 as pr  # noqa: E402
from test_possession_rules_v2 import AWAY_AT, HOME_AT, TEAM, dribble, scene  # noqa: E402

FPS = 25


def run(ball, players, graded=None, windows=(), kickoffs=()):
    graded = np.ones(len(ball), dtype=bool) if graded is None else graded
    return "".join(pr.label_frames(ball, players, TEAM, graded, FPS, windows, kickoffs))


def test_dead_before_kickoff_then_play():
    """The 2nd-half case: graded from frame 0, kickoff at frame 300."""
    ball, players = scene(500)
    dribble(ball, 320, 420, HOME_AT)
    out = run(ball, players, kickoffs=[(300, 310)])
    assert out[:300] == "D" * 300, "nothing is in play before the kickoff"
    assert set(out[320:420]) == {"H"}
    assert v2.count_possessions(out) == 1


def test_no_kickoff_tag_keeps_first_owner():
    ball, players = scene(500)
    dribble(ball, 0, 100, HOME_AT)
    out = run(ball, players)
    assert set(out) == {"H"}, "without a tagged kickoff, v3 behaviour is kept"


def test_set_piece_forces_dead_and_splits_possession():
    ball, players = scene(400)
    dribble(ball, 0, 100, HOME_AT)
    dribble(ball, 200, 300, HOME_AT)
    without = run(ball, players)
    with_tag = run(ball, players, windows=[(120, 140)])
    assert set(without) == {"H"}, "a 4 s gap is not dead on its own"
    assert with_tag[120:140] == "D" * 20, "a throw-in is a dead ball"
    assert v2.count_possessions(with_tag) == 2, "the stoppage splits the possession"


def test_set_piece_ends_when_play_restarts():
    ball, players = scene(400)
    dribble(ball, 0, 400, HOME_AT)
    out = run(ball, players, windows=[(120, 140)])
    assert set(out) == {"H"}, "evidence at the window's first frame means the ball is already live"


def test_set_piece_dead_is_shorter_than_its_window():
    ball, players = scene(400)
    dribble(ball, 0, 100, HOME_AT)
    dribble(ball, 130, 300, HOME_AT)
    out = run(ball, players, windows=[(110, 200)])
    assert out[110:130] == "D" * 20
    assert out[130:300] == "H" * 170
    assert v2.count_possessions(out) == 2


def test_kickoff_window_outside_the_block_is_ignored():
    ball, players = scene(400)
    dribble(ball, 0, 100, HOME_AT)
    graded = np.zeros(400, dtype=bool)
    graded[:200] = True
    out = run(ball, players, graded, kickoffs=[(300, 310)])
    assert out[200:] == "X" * 200, "a kickoff in the other half must not leak across"
    assert set(out[:120]) == {"H"}


def test_two_half_blocks_use_their_own_kickoff():
    """1st half kicks off at the block's first frame; the 2nd half's block opens long before its kickoff."""
    ball, players = scene(900)
    dribble(ball, 20, 200, HOME_AT)
    dribble(ball, 700, 800, AWAY_AT)
    graded = np.zeros(900, dtype=bool)
    graded[:300] = True
    graded[600:] = True
    out = run(ball, players, graded, kickoffs=[(0, 10), (650, 660)])
    assert set(out[20:200]) == {"H"}, "a kickoff at the block's first frame blanks nothing"
    assert out[600:650] == "D" * 50, "the 2nd half is graded before it kicks off (FAQ.md)"
    assert set(out[700:800]) == {"A"}
    assert out[300:600] == "X" * 300
    assert v2.count_possessions(out) == 2


def test_output_alphabet_and_unscored_frames():
    ball, players = scene(600)
    dribble(ball, 0, 100, HOME_AT)
    graded = np.zeros(600, dtype=bool)
    graded[:400] = True
    out = run(ball, players, graded, windows=[(200, 220)], kickoffs=[(10, 20)])
    assert len(out) == 600 and set(out) <= set("HADX")
    assert out[400:] == "X" * 200
    assert out[200:220] == "D" * 20


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok  {}".format(test.__name__))
