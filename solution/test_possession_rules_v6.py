"""Synthetic checks for possession_rules_v6.py. No match data is read.

    python solution/test_possession_rules_v6.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import possession_rules_v2 as v2  # noqa: E402
import possession_rules_v5 as v5  # noqa: E402
import possession_rules_v6 as pr  # noqa: E402
from test_possession_rules_v2 import TEAM, dribble, scene  # noqa: E402

FPS = 25
EPISODE = 20          # 0.8 s of contact: too short for v5's dense core on its own
FLIGHT = 30           # 1.2 s in flight: longer than CORE_GAP_S, so no core spans two episodes
HOME_AT = (12.0, 12.0)
AWAY_AT = (50.0, 10.0)


def play(ball, players, start, moves, episode=EPISODE, flight=FLIGHT):
    """Ball episodes at each (team, point) from `start`, with observed flights between them.

    Home episodes are carried by track 0 and away episodes by track 2; the carrier holds its
    position through the flight that follows, so the ball leaves that player. Returns the frame
    after the last episode.
    """
    cursor = start
    for index, (team, point) in enumerate(moves):
        track = 0 if team == v2.HOME else 2
        end = cursor + episode
        dribble(ball, cursor, end, point)
        players[cursor:end + flight, track] = point
        cursor = end
        if index + 1 < len(moves):
            ball[cursor:cursor + flight] = np.linspace(point, moves[index + 1][1], flight + 1)[1:]
            cursor += flight
    return cursor


def passage(moves, frames=400):
    """A fresh scene with the given moves played from frame 0."""
    ball, players = scene(frames)
    play(ball, players, 0, moves)
    return ball, players


def run(ball, players, graded=None, windows=(), kickoffs=()):
    graded = np.ones(len(ball), dtype=bool) if graded is None else graded
    return "".join(pr.label_frames(ball, players, TEAM, graded, FPS, windows, kickoffs))


def prepared(ball, players):
    """(blocked, evidence, clear, episodes) for a synthetic scene."""
    _, blocked, evidence, clear = pr.prepared(ball, players, TEAM, FPS)
    return blocked, evidence, clear, pr.episodes_of(evidence, 0, len(ball) - 1)


def strong_turn(ball, start, end, point, amplitude=2.5):
    """A ball moving fast but never leaving the player at `point`."""
    steps = np.arange(end - start)
    ball[start:end, 0] = point[0] + amplitude * np.sin(2 * np.pi * steps / 20.0)
    ball[start:end, 1] = point[1]


HOME_MOVES = [(v2.HOME, HOME_AT), (v2.HOME, (30, 14)), (v2.HOME, (58, 36)), (v2.HOME, (62, 14))]


# -- the pass proxy: four episodes of one team, joined by completed passes -----------------------

def test_four_episodes_of_one_team_are_a_run():
    ball, players = passage(HOME_MOVES)
    blocked, evidence, clear, episodes = prepared(ball, players)
    assert len(episodes) == 4, "each touch is its own episode"
    assert pr.transfers_from(episodes, clear, blocked, FPS) == 3
    assert pr.chain_runs(evidence, clear, blocked, FPS, 0, len(ball) - 1) == [
        (v2.HOME, episodes[0][1], episodes[-1][2])]


def test_three_episodes_are_not_enough():
    ball, players = passage(HOME_MOVES[:3])
    blocked, evidence, clear, episodes = prepared(ball, players)
    assert len(episodes) == 3
    assert pr.chain_runs(evidence, clear, blocked, FPS, 0, len(ball) - 1) == [], \
        "two consecutive passes are not 'more than two'"


def test_v5_misses_the_passing_move_and_v6_finds_it():
    ball, players = passage(HOME_MOVES)
    assert set(run(ball, players)) == {"H"}
    v5_labels = "".join(v5.label_frames(ball, players, TEAM, np.ones(len(ball), bool), FPS))
    assert set(v5_labels) == {"D"}, "no single touch lasts 1.5 s, so v5 sees nothing"


def test_chain_run_starts_at_its_first_episode():
    ball, players = scene(760)
    dribble(ball, 0, 120, AWAY_AT)                      # a long carry: v5 confirms this one
    played = play(ball, players, 120, HOME_MOVES)
    _, _, _, episodes = prepared(ball, players)         # [away carry, four home touches]
    assert [episode[0] for episode in episodes] == [v2.AWAY] + [v2.HOME] * 4
    out = run(ball, players)
    first = episodes[1][1]
    assert set(out[:first]) == {"A"}, "the away carry is held until the home team takes over"
    assert set(out[first:played]) == {"H"}, "the run is dated from the interception, not the 3rd pass"


# -- what makes a completed pass ----------------------------------------------------------------

def test_transfer_needs_the_ball_to_leave_the_players():
    ball, players = passage(HOME_MOVES[:2])
    blocked, _, clear, episodes = prepared(ball, players)
    assert pr.transfer(clear, blocked, episodes[0][2], episodes[1][1], FPS)
    strong_turn(ball, episodes[0][2] + 1, episodes[1][1], HOME_AT)
    clear = pr.clear_frames(ball, players)
    assert not pr.transfer(clear, blocked, episodes[0][2], episodes[1][1], FPS), \
        "a ball that never leaves the players is a duel or a dribble, not a pass"


def test_transfer_needs_the_ball_to_be_observed():
    ball, players = passage(HOME_MOVES[:2])
    blocked, _, clear, episodes = prepared(ball, players)
    ball[episodes[0][2] + 1:episodes[1][1]] = np.nan
    clear = pr.clear_frames(ball, players)
    assert not pr.transfer(clear, blocked, episodes[0][2], episodes[1][1], FPS), \
        "a lost ball is not a completed pass"


def test_transfer_needs_the_interval_to_be_short():
    ball, players = passage(HOME_MOVES[:2])
    blocked, _, clear, _ = prepared(ball, players)
    far = int(v2.LINK_GAP_S * FPS) + 5
    assert not pr.transfer(clear, blocked, 0, far, FPS)


def test_dead_ball_frame_breaks_the_pass():
    ball, players = passage(HOME_MOVES[:2])
    blocked, _, clear, episodes = prepared(ball, players)
    blocked[episodes[0][2] + 5] = True
    assert not pr.transfer(clear, blocked, episodes[0][2], episodes[1][1], FPS)


def test_opponent_episode_breaks_the_chain():
    moves = ([(v2.HOME, HOME_AT), (v2.HOME, (30, 14)), (v2.HOME, (52, 16))]
             + [(v2.AWAY, (46, 26))]
             + [(v2.HOME, (58, 36)), (v2.HOME, (62, 14)), (v2.HOME, (34, 10))])
    ball, players = passage(moves, frames=560)
    blocked, evidence, clear, episodes = prepared(ball, players)
    assert len(episodes) == 7
    assert pr.chain_runs(evidence, clear, blocked, FPS, 0, len(ball) - 1) == [], \
        "an away touch in the middle leaves only three home episodes either side"


def test_two_teams_passing_do_not_join():
    moves = [(v2.HOME, HOME_AT), (v2.HOME, (30, 14)), (v2.HOME, (58, 36)), (v2.HOME, (62, 14)),
             (v2.AWAY, (46, 26)), (v2.AWAY, (50, 34)), (v2.AWAY, (52, 24)), (v2.AWAY, (46, 26))]
    ball, players = passage(moves, frames=640)
    blocked, evidence, clear, episodes = prepared(ball, players)
    runs = pr.chain_runs(evidence, clear, blocked, FPS, 0, len(ball) - 1)
    assert runs == [(v2.HOME, episodes[0][1], episodes[3][2]),
                    (v2.AWAY, episodes[4][1], episodes[7][2])]


# -- merging and inherited behaviour ------------------------------------------------------------

def test_merge_runs_trims_an_overlapping_run():
    merged = pr.merge_runs([(v2.HOME, 10, 40)], [(v2.HOME, 30, 60), (v2.AWAY, 70, 90)])
    assert merged == [(v2.HOME, 10, 40), (v2.HOME, 41, 60), (v2.AWAY, 70, 90)]


def test_merge_runs_drops_a_swallowed_run():
    assert pr.merge_runs([(v2.HOME, 10, 90)], [(v2.AWAY, 30, 40)]) == [(v2.HOME, 10, 90)]


def test_a_scene_without_passes_is_unchanged_from_v5():
    ball, players = scene(400)
    dribble(ball, 0, 200, HOME_AT)
    play(ball, players, 200, [(v2.HOME, (30, 14))], episode=25)
    graded = np.ones(400, dtype=bool)
    assert run(ball, players, graded) == "".join(v5.label_frames(ball, players, TEAM, graded, FPS))


def test_dead_before_kickoff_beats_a_pass_chain():
    ball, players = passage(HOME_MOVES)
    kick = len(ball) - 100
    out = run(ball, players, kickoffs=[(kick, kick + 10)])
    assert out[:kick] == "D" * kick
    assert set(out[kick:]) == {"H"}, "the run resumes once play is allowed"


def test_ungraded_is_x_and_alphabet_is_valid():
    ball, players = passage(HOME_MOVES)
    graded = np.zeros(len(ball), dtype=bool)
    graded[:300] = True
    out = run(ball, players, graded)
    assert out[300:] == "X" * (len(ball) - 300) and set(out) <= set("HADX")


def test_nearest_player_distance_ignores_a_missing_ball():
    ball, players = passage(HOME_MOVES)
    distance = pr.nearest_player_distance(ball, players)
    assert np.isnan(distance[np.isnan(ball[:, 0])]).all()
    assert np.isfinite(distance[~np.isnan(ball[:, 0])]).all()


def test_clear_frames_means_the_ball_is_between_players():
    ball, players = passage(HOME_MOVES)
    clear = pr.clear_frames(ball, players)
    _, _, _, episodes = prepared(ball, players)
    assert not clear[episodes[0][1]:episodes[0][2] + 1].any(), "at an episode the ball is at a player"
    flight = slice(episodes[0][2] + 1, episodes[1][1])
    assert clear[flight].sum() > 5, "a pass spends most of its flight clear of everyone"
    assert not clear[np.isnan(ball[:, 0])].any(), "a lost ball is not evidence of a pass"


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok  {}".format(test.__name__))
