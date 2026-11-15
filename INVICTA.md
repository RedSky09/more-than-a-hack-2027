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
| `solution/possession_rules_v3.py` | the submitted system |
| `solution/possession_rules_v2.py` | predecessor, imports reused by v3 |
| `solution/possession_rules.py` | v1; its ATD reader and CSV writer are reused by v2 and v3 |

## 2. Assets used

| asset | used | how |
|---|---|---|
| tactical ATD (players and ball) | yes | ball and player positions, team of each track slot |
| Smart Tagging | yes | only the restart windows (throw-in, corner kick, goal kick, free kick, kick-off after a goal), as evidence that a short gap is a dead ball |
| tactical and broadcast video | no | not read |
| homography | no | not needed, the ATD is already in pitch coordinates |
| broadcast ATD and homography | no | not on the submission timeline, mapping them was out of scope |

Smart Tagging `Team` tags were not used. They are inferred and `N/A` on every throw-in.

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
   next run starts. Before the first run of a half, the first team is used, since play restarts with a
   clear possessor.
5. **Dead ball needs evidence.** The stretch between two runs is `D` if any of these holds:
   - it lasts at least 6.0 s with no controlled run;
   - it lasts at least 2.0 s and overlaps a Smart Tagging restart window;
   - it lasts at least 2.0 s and contains a moment where the ball is shown to have left the pitch.

   **Ball out of play** means the ball is tracked at or beyond a touchline or goal line for at least
   3 of 5 consecutive frames (so a single noisy frame does not count), or was last seen within 1.5 m
   of a line and then vanished for at least 1.0 s (it left with the play). When a stretch is dead for
   this reason, `D` starts at that moment and the previous team keeps the ball up to it. A ball that
   vanishes in the middle of the pitch is treated as an occlusion, not a stoppage, and possession is
   held through it.

   Dead stretches split possessions, as the ground-truth definition requires.
6. **Outside the graded halves** the label is `X`, which is never scored. We write no `X` inside the
   graded halves.

All constants are fixed defaults at the top of `possession_rules_v3.py`. They were chosen from
football reasoning and the written definition, and were **not fitted or tuned on any data**.

## 4. Training data

None. There is no trained model and no parameter was fitted on any dataset.

We downloaded the Metrica Sports sample games (`metrica-sports/sample-data`) as a candidate for
tuning. That repository has no licence file, and the organisers' replies did not confirm it was
usable. We therefore did **not** use it for anything. It is excluded from git via `.gitignore`.

## 5. How the public ground truth was used

The 5 public minutes were used to check that the file is well-formed, aligned and that the labels mean
what we think. Two design decisions were made after seeing a public-window score, and we state them
plainly rather than describing the process as if it were cleaner than it was:

- v1 (nearest player within 3 m, short gaps bridged) produced 34 possessions on that window against
  17 true ones. That prompted the redesign into v2, which follows the written definition of controlled
  possession.
- v2 labelled about 58% of graded frames `D`, because a lost ball for 6 s or more became a stoppage.
  That prompted v3, which adds ball-out-of-play evidence and short-gap interpolation.

In both cases the new constants were chosen from the definition and football reasoning, not by
searching over the public window, and nothing was re-tuned against it afterwards. The public window
has 17 possessions, so we do not treat its score as evidence that v3 is better than v2 or than the
published baselines; the differences there are inside the noise band the scorer itself warns about.

Public results for v3: macroF1 0.704, spellF1 0.471 (v2: 0.678 / 0.443). Shown for completeness, not
as a claim of improvement.

## 6. Limitations

- **Possessions are merged.** On the public window v3 reports 11 possessions against 17 true ones,
  mean 21.1 s against 14.2 s. Real changes of possession are missed, most often when control is
  brief. `spellF1` penalises this heavily.
- **Away possession is under-called.** On the public window, 700 of 2,577 true away frames are
  labelled `D` and 71 labelled `H`.
- **Dead ball is still imperfect.** Precision 0.47 and recall 0.54 on the public window. Over the
  whole match 37.5% of frames are `D`. Long stretches with no controlled run still become `D` even
  when play may be live, and some real stoppages away from a line are missed.
- **The ball is missing most of the time.** DATA.md reports it visible on 56.9% of tactical frames,
  so about 43% are missing, and they are disproportionately the crowded and aerial moments. Control
  evidence is therefore sparse, and possession changes are sometimes detected late (public-window
  latency: mean -6.5 frames).
- **Possession changes are detected from proximity only.** There is no pass or event recognition.
  Aerial duels and fast counter-attacks are the likely failures.
- **The constants are not tuned.** Tuning them needs external data with a clear licence, which we did
  not obtain in time.
- **No full-match score is reported.** The full-match ground truth is private.

## 7. Process and data handling

The code was written with help from AI assistants. The provided match data was kept out of the working
folder whenever an assistant had file access, and was put back only to run the pipeline locally. No
match data, video or tracking extract was uploaded to an AI service; only code and scorer output were
shared. <team: confirm this statement is accurate before submitting>

## 8. Reproduction

Python 3.12, `pip install -r requirements.txt` (numpy and pandas). Unpack the data bundle into `data/`
as described in `data/README.md`, then:

```
python solution/possession_rules_v3.py
python evaluation/validate_submission.py evaluation/results/possession_rules_v3.csv
```

The output is deterministic. Copy `evaluation/results/possession_rules_v3.csv` to `INVICTA.csv` for the
submission form; the CSV is not committed, the repository regenerates it.

Unit tests on synthetic data (no match data): `python solution/test_possession_rules_v3.py`.
