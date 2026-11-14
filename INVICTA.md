# INVICTA: technical document

More Than A Hack 2027, Phase 1 (possession labels for BAR-ATH).

Team INVICTA: Perdi, Leonardo Nifinluri, M. Ali Assad

## 1. Approach

A rule-based system with no learned components. It reads the provided tactical tracking (ATD) and
the Smart Tagging restart windows, and writes one label per video frame (`H`, `A`, `D`, `X`).

The idea follows CHALLENGE.md: possession is held by the last team that had **controlled** possession,
and it changes hands only when the other team gains control. A loose or contested ball stays with the
previous team. The first approach we tried, "whoever is nearest the ball right now", flickers and
produces far too many possessions.

Code: `solution/possession_rules_v3.py` (the submission). It builds on `possession_rules_v2.py`, which
holds the control rules, and `possession_rules.py`, which holds the ATD reader and writer.

## 2. Assets used

| asset | used | how |
|---|---|---|
| tactical ATD (players and ball) | yes | ball and player positions, team of each track slot |
| Smart Tagging | yes | only the restart windows (throw-in, corner kick, goal kick, free kick, kick-off after a goal), as evidence that a short gap is a dead ball |
| tactical and broadcast video | no | not read |
| homography | no | not needed, the ATD is already in pitch coordinates |
| broadcast ATD and homography | no | not on the submission timeline |

Smart Tagging `Team` tags were not used. They are inferred and `N/A` on every throw-in.

## 3. Possession rules implemented

Per graded half (the half boundaries come from `games.half_video_blocks`):

1. **Control evidence per frame.** The ball must be located. The nearest player of one team must be
   within 3.0 m of it and at least 1.0 m closer than the nearest opponent. Frames where the ball moves
   faster than 8 m/s (a pass or shot), or stays inside a 1 m box for 1 s either side of the frame (a
   restart being set up), give no evidence.
2. **Controlled runs.** Consecutive evidence frames of one team form a run. A run shorter than 1.5 s is
   treated as a touch, deflection or flicker and discarded. Evidence more than 3 s apart does not join.
3. **Possession is held.** Between two controlled runs, the previous team keeps the ball until the
   next run starts. Before the first run of a half, the first team to control the ball is used.
4. **Ball out of play (v3).** The ball is out when the tracking shows it on or beyond a touchline or
   goal line, confirmed by 3 frames within a 5 frame window. A ball last seen within 1.5 m of a line
   that then goes missing for 1 s or more also counts as out. A stretch between two controlled runs
   becomes dead from the first out moment if it lasts at least 2 s. A ball that vanishes in the middle
   of the pitch is an occlusion and keeps possession.
5. **Dead ball needs evidence.** A stretch between two runs is `D` if it lasts at least 6 s, or at
   least 2 s and overlaps a Smart Tagging restart window, or if rule 4 applies.
6. **Short ball gaps are interpolated (v3).** A missing ball gap of up to 1 s is filled in a straight
   line, unless that would need a ball speed above 30 m/s.
7. **Outside the graded halves** the label is `X`, which is never scored.

Constants are at the top of `possession_rules_v3.py` and `possession_rules_v2.py`. They were chosen
from the rules and football reasoning before the v3 public score was seen. None was fitted on any data.

## 4. Training data

None. There is no trained model and no parameter was fitted on any dataset.

We downloaded the Metrica Sports sample games (`metrica-sports/sample-data`) as a candidate for tuning.
That repository has no licence file, and the organisers' replies did not confirm it was usable. We
therefore did **not** use it for anything. It is excluded from git via `.gitignore`.

## 5. How the public ground truth was used

The 5 public minutes were used to check the file is well-formed and aligned, and to compare variants.

The design history matters for interpretation. v1 (nearest player within 3 m, short gaps bridged)
produced 34 possessions on that window, against 17 true ones. That prompted v2, which follows the
written definition of controlled possession. v3 then added ball-out-of-play evidence and interpolation,
motivated by v2's dead-ball over-prediction. The v3 constants were fixed from the rules before v3 was
scored. We did not change them after seeing its public score.

The public window has 17 possessions, so its score is not evidence that v3 is better than the baselines.
Public scores of v2 and v3 differ by about 0.03 spellF1, which is within noise.

Public results for v3 (macroF1 0.704, spellF1 0.471) are shown for completeness.

## 6. Limitations

- Dead ball is still over-predicted. About 38% of graded frames are `D` in v3, against 46% in v2. Dead-ball
  precision is about 0.47. Long stretches without a controlled run, often when the ball is not tracked,
  can still be labelled `D` while play is live.
- The ball is visible on 56.9% of tactical frames (DATA.md), so it is missing on about 43% of them.
  Control evidence is sparse and possession changes are sometimes detected late (median latency about
  6 frames, 0.25 s, on the public window).
- v3 predicts 11 possessions against 17 in the public window, with a mean length of 21 s against 14 s.
  Real changes are missed when two possessions are merged.
- Away possession is under-called. Of 2,577 away frames in the public window, about 700 are labelled `D`.
- Possession changes are detected from proximity and the ball path only. There is no pass or event
  recognition. Aerial duels and fast counter-attacks are the likely failures.
- The team-shape idea for long gaps is not implemented.
- The constants are not tuned. With external data of a clear licence they could be tuned later.
- We do not report a full-match score, since the ground truth is private.

## 7. Process and data handling

The code was written with help from AI assistants. The provided match data was kept out of the working
folder whenever an assistant had file access, and was restored only to run the pipeline locally. No
match data, video or tracking extract was uploaded to an AI service. Only code and scorer output were
shared with the assistants.

## 8. Reproduction

Python 3.12, `pip install -r requirements.txt` (numpy and pandas). Unpack the data bundle into `data/`
as described in `data/README.md`, then:

```
python solution/possession_rules_v3.py
python evaluation/validate_submission.py evaluation/results/possession_rules_v3.csv
```

The output is deterministic. Copy `evaluation/results/possession_rules_v3.csv` to `INVICTA.csv` for the
submission form. Unit tests on synthetic data (no match data) run with
`python solution/test_possession_rules_v2.py` and `python solution/test_possession_rules_v3.py`.
