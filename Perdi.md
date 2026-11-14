# Perdi: technical document

More Than A Hack 2027, Phase 1 (possession labels for BAR-ATH).

Team members: Leonardo Nifinluri, M. Ali Assad

## 1. Approach

A rule-based system with no learned components. It reads the provided tactical tracking (ATD) and
Smart Tagging, and writes one label per video frame: `H` (home in possession), `A` (away in
possession), `D` (dead ball) or `X` (no answer, never scored).

The rules follow CHALLENGE.md. Possession is held by the last team that had controlled possession,
and it changes hands only when the other team gains control. A loose or contested ball stays with the
previous team. Our first version (v1) assigned the ball to whichever player was nearest at each frame.
It flickered and produced about twice as many possessions as the ground truth, so we replaced it.

Final submission: `solution/possession_rules_v3.py`. It builds on v2 (`solution/possession_rules_v2.py`),
which is kept as a fallback. v1 (`solution/possession_rules.py`) is kept because v2 and v3 import its
ATD reader and file writer.

## 2. Assets used

| asset | used | how |
|---|---|---|
| tactical ATD (player and ball positions) | yes | ball and player positions in metres, team of each track slot |
| Smart Tagging | yes, restart windows only | throw-ins, corners, goal kicks, free kicks and kick-offs mark times when a dead ball is expected |
| Smart Tagging team tags | no | they are inferred and empty on throw-ins |
| tactical and broadcast video | no | not read |
| homography | no | not needed, the ATD is already in pitch coordinates |
| broadcast ATD and homography | no | not on the submission timeline |

## 3. Rules implemented (v3)

Each graded half is handled separately. The half boundaries come from `games.half_video_blocks`.

1. **Control evidence per frame.** The ball must be located, and the nearest player of one team must be
   within 3.0 m of it and at least 1.0 m closer than the nearest opponent. Frames where the ball moves
   faster than 8 m/s (a pass or shot), or sits within a 1 m box for 2 s (a restart being set up), give no
   evidence.
2. **Controlled runs.** Consecutive evidence frames of one team form a run. A run shorter than 1.5 s is a
   touch, deflection or flicker, and is discarded. Evidence more than 3 s apart does not join.
3. **Possession is held.** Between two controlled runs, the previous team keeps the ball until the next
   run starts. Before the first run of a half, the first team is used.
4. **Dead ball needs evidence.** A stretch between two runs is `D` if it lasts at least 6 s, or at least
   2 s and overlaps a Smart Tagging restart window.
5. **Ball out of play (new in v3).** A stretch between two runs is also dead if the ball was seen beyond a
   touchline or goal line and the stretch lasts at least 2 s. The same applies when the ball was last
   seen within 1.5 m of a line and then vanished for at least 1 s. Dead ball starts at that moment, not at
   the end of the previous run. A ball that vanishes in mid-pitch is treated as an occlusion, so the
   previous team keeps the ball.
6. **Short gaps are interpolated (new in v3).** Gaps of up to 1 s between two ball observations are filled
   with a straight line, unless that needs a ball speed above 30 m/s.
7. **Outside the graded halves** the label is `X`.

All constants are fixed defaults at the top of the two files. They were chosen from football reasoning
and were **not fitted or tuned on any data**.

## 4. Training data

None. There is no trained model, and no parameter was fitted on any dataset.

We looked at the Metrica Sports sample games (`metrica-sports/sample-data`) as a candidate for tuning. The
repository has no licence file, and the organisers' replies did not confirm it was usable. We did **not**
use it. Its files are excluded from git.

## 5. How the public ground truth was used

The 5 public minutes were used to check that the file is well-formed and aligned, and to see the failure
modes listed below. Two design decisions followed from that:

- v1 produced 34 possessions on that window against 17 true ones. That led to v2, which follows the written
  definition of controlled possession.
- v3 adds the ball-out rule because v2 labelled about 46% of frames dead, mostly where the ball was not
  tracked.

The constants of v2 and v3 were chosen from the definitions and from football reasoning. They were not
searched over the public window. We report the public window scores only for transparency. That window has
17 possessions and a noise band of about 0.04 spellF1, so the score is not evidence that v3 beats the
baselines.

Public window results (scorer output, `run_evaluation.py`): v3 macroF1 0.704, spellF1 0.471. v2 macroF1
0.678, spellF1 0.443. The baselines reach spellF1 0.424 at best.

## 6. Limitations

- **Too few possessions.** v3 predicts 11 possessions for the public window against 17 true ones, with a mean
  length of 21 s against 14 s. Controlled runs are merged across real changes of possession.
- **Dead ball is still over-predicted and under-recalled.** Dead-ball precision is 0.47 and recall 0.54. Away
  possession is labelled dead in 700 of 2,577 frames.
- **Sparse ball tracking.** The ball is visible on 56.9% of tactical frames (DATA.md), so it is missing on about 43% of them. Control evidence is sparse, and
  possession changes are detected late (median latency about 6 frames, 0.25 s, on the public window).
- **No event recognition.** Possession changes come from proximity alone. Aerial duels and fast counter-attacks
  are the likely failures.
- **Out-of-play thresholds are untested on other games.** The 1.5 m line margin and the 1 s vanish time may
  fire too often or too rarely on tracking that jitters at the lines.
- **No full-match score.** The ground truth is private, so we do not report one.

## 7. Process and data handling

The code was written with help from AI assistants. The provided match data was kept out of the working folder
while any assistant had file access, and was placed there only to run the pipeline locally. No match data,
video or tracking extract was uploaded to any AI service. Only code and scorer output were shared with the
assistants.

## 8. Reproduction

Python 3.12 with `pip install -r requirements.txt` (numpy and pandas). Unpack the data bundle into `data/` as
described in `data/README.md`, then run:

```
python solution/possession_rules_v3.py
python evaluation/validate_submission.py evaluation/results/possession_rules_v3.csv
```

The output is deterministic. Copy `evaluation/results/possession_rules_v3.csv` to `Perdi.csv` for the
submission. Synthetic unit tests (no match data): `python solution/test_possession_rules_v3.py`.
