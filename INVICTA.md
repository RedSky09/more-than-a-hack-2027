# INVICTA: technical document

More Than A Hack 2027, Phase 1 (possession labels for BAR-ATH).

Team: INVICTA. Members: Perdi, Leonardo Nifinluri, M. Ali Assad.

## 1. Approach

A rule-based system with no learned components. It reads the provided tactical tracking (ATD) and
the Smart Tagging, and writes one label per video frame (`H`, `A`, `D`, `X`).

The idea follows CHALLENGE.md: possession is held by the last team that had **controlled** possession,
and it changes hands only when the other team gains control. A loose or contested ball stays with the
previous team. The first version we tried, "whoever is nearest the ball right now", flickers and
produces several times too many possessions.

Code:

| file | role |
|---|---|
| `solution/possession_rules_v4.py` | the submitted system |
| `solution/possession_rules_v3.py` | ball-out-of-play evidence and short-gap interpolation, imported by v4 |
| `solution/possession_rules_v2.py` | control rules and held possession, imported by v3 and v4 |
| `solution/possession_rules.py` | v1; its ATD reader and CSV writer are reused by all later versions |

## 2. Assets used

| asset | used | how |
|---|---|---|
| tactical ATD (players and ball) | yes | ball and player positions, team of each track slot |
| Smart Tagging | yes | restart windows (throw-in, corner, goal kick, free kick, kick-off after a goal) and the two half kickoffs |
| tactical and broadcast video | no | not read |
| homography | no | not needed, the ATD is already in pitch coordinates |
| broadcast ATD and homography | no | not on the submission timeline, mapping them was out of scope |

Smart Tagging `Team` tags were not used. They are inferred and `N/A` on every throw-in.

We inspected the ATD metadata to see whether the export declares ball-state channels such as
`BallState` or `BallOwningTeam`, which would have given ball-dead and ball-owner directly. It
declares only `x`, `y` and `s`, and every record carries three values per object. The information is
therefore available only as positions and speeds, which is what this system uses.

## 3. Possession rules implemented

Per graded half (half boundaries from `games.half_video_blocks`):

1. **Ball interpolation.** Gaps in the ball track of up to 1.0 s are filled in a straight line,
   unless that would require a ball speed above 30 m/s. This only adds evidence; it never creates
   possession on its own.
2. **Control evidence per frame.** The ball must be located. The nearest player of one team must be
   within 3.0 m of it and at least 1.0 m closer than the nearest opponent. Frames where the ball moves
   faster than 8 m/s (a pass or shot), or stays inside a 1 m box for 1 s either side of the frame (a
   restart being set up), give no evidence.
3. **Controlled runs.** Consecutive evidence frames of one team form a run. A run shorter than 1.5 s is
   treated as a touch, deflection or flicker and discarded. Evidence more than 3 s apart does not join.
4. **Possession is held.** Between two controlled runs, the previous team keeps the ball until the
   next run starts. Dead-ball stretches split possessions, as the ground-truth definition requires
   (`H D H` is two possessions).
5. **Dead until kickoff.** Everything before a half's tagged `Kick Off (Start)` is `D`. FAQ.md states
   the 2nd half is graded from video frame 110808 but kicks off at 113742, and those 2934 frames are
   all dead. Filling them with a team would invent a possession that never happened.
6. **Set pieces are dead balls.** DATA.md states that every set piece is a restart and every restart
   ends a dead-ball stretch. A set-piece window is marked `D` from its start until the first frame
   where a team shows control of the ball, so the dead part is labelled and the possession splits,
   while play already live inside the window is not swallowed.
7. **Dead ball otherwise needs evidence.** A stretch between two runs is also `D` if it lasts at least
   6.0 s with no controlled run, or at least 2.0 s and contains a moment where the ball is shown to
   have left the pitch.

   **Ball out of play** means the ball is tracked at or beyond a touchline or goal line for at least
   3 of 5 consecutive frames, or was last seen within 1.5 m of a line and then vanished for at least
   1.0 s. When a stretch is dead for this reason, `D` starts at that moment and the previous team
   keeps the ball up to it. A ball that vanishes mid-pitch is an occlusion, not a stoppage, and
   possession is held through it.
8. **Outside the graded halves** the label is `X`, which is never scored. We write no `X` inside the
   graded halves.

All constants are fixed defaults at the top of the version modules. They were chosen from football
reasoning, the written definition and the repository's own documentation, and were **not fitted or
tuned on any data**.

## 4. Training data

None. There is no trained model and no parameter was fitted on any dataset.

We downloaded the Metrica Sports sample games (`metrica-sports/sample-data`) as a candidate for
tuning. That repository has no licence file, and the organisers' replies did not confirm it was
usable. We therefore did **not** use it for anything. It is excluded from git via `.gitignore`.

## 5. How the public ground truth was used

The 5 public minutes were used to check that the file is well-formed, aligned and that the labels mean
what we think. We describe the design history plainly rather than as if it were cleaner than it was:

- v1 (nearest player within 3 m, short gaps bridged) produced 34 possessions on that window against
  17 true ones. That prompted v2, which follows the written definition of controlled possession.
- v2 labelled about 58% of graded frames `D`, because a lost ball for 6 s or more became a stoppage.
  That prompted v3, which adds ball-out-of-play evidence and short-gap interpolation.
- v4 adds the pre-kickoff dead stretch and the set-piece dead ball. These came from FAQ.md and DATA.md
  and were **verified to change nothing inside the public window**: v3 and v4 score identically on it,
  and every frame v4 changes (4063) lies outside the 5 graded public minutes.

No constant was re-tuned against the public window, and we do not treat its score as evidence that
any version ranks above another; the differences are inside the noise band the scorer warns about.

Public results for v4: macroF1 0.704, spellF1 0.471 (identical to v3). Shown for completeness, not as
a claim of improvement.

## 6. Limitations

- **Possessions are merged.** On the public window v4 reports 11 possessions against 17 true ones,
  mean 21.1 s against 14.2 s. Real changes of possession are missed, most often when control is
  brief. `spellF1` penalises this heavily, and it is the largest known weakness.
- **Away possession is under-called.** On the public window, 700 of 2,577 true away frames are
  labelled `D` and 71 labelled `H`.
- **Dead ball is imperfect.** On the public window precision 0.47 and recall 0.54. Over the whole
  match 39.5% of frames are `D`, 241 possessions of mean 13.3 s. Long stretches with no controlled
  run still become `D` when play may be live.
- **The ball is missing most of the time.** DATA.md reports it visible on 56.9% of tactical frames,
  so about 43% are missing, disproportionately in crowded and aerial moments. Control evidence is
  therefore sparse, and possession changes are sometimes detected late (public-window latency: mean
  -6.5 frames).
- **Set-piece windows are inferred, not ground truth.** If a window starts before the ball is
  actually dead, rule 6 adds false `D`. The change is unmeasurable on the public window, so we
  accept it on the strength of the definition rather than on a score.
- **Possession changes are detected from proximity only.** There is no pass or event recognition.
  Aerial duels and fast counter-attacks are the likely failures.
- **The constants are not tuned.** Tuning them needs external data with a clear licence, which we did
  not obtain in time.
- **No full-match score is reported.** The full-match ground truth is private.

## 7. Process and data handling

The code was written with help from AI assistants. The provided match data was kept out of the working
folder whenever an assistant had file access, and was put back only to run the pipeline locally. No
match data, video or tracking extract was uploaded to an AI service; only code, scorer output and the
ATD's declared channel names were shared. <team: confirm this statement is accurate before submitting>

## 8. Reproduction

Python 3.12, `pip install -r requirements.txt` (numpy and pandas). Unpack the data bundle into `data/`
as described in `data/README.md`, then:

```
python solution/possession_rules_v4.py
python evaluation/validate_submission.py evaluation/results/possession_rules_v4.csv
```

The output is deterministic. Copy `evaluation/results/possession_rules_v4.csv` to `INVICTA.csv` for the
submission form; the CSV is not committed, the repository regenerates it.

Expected whole-match label counts: H 45801, A 34226, D 78830, X 40462; 241 possessions of mean 13.3 s.

Unit tests on synthetic data (no match data): `python solution/test_possession_rules_v4.py`.
