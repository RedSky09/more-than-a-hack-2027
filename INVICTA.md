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
| `solution/possession_rules_v5.py` | the submitted system |
| `solution/possession_rules_v6.py` | the pass-based proxy of CHALLENGE.md, implemented and rejected (section 5) |
| `solution/possession_rules_v4.py` | kickoff and set-piece dead ball, imported by v5 |
| `solution/possession_rules_v3.py` | ball-out-of-play evidence and short-gap interpolation, imported by v4 and v5 |
| `solution/possession_rules_v2.py` | control rules and held possession, imported by v3, v4 and v5 |
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
   unless that would require a ball speed above 30 m/s. A gap is **not** filled if it touches a
   frame where the ball is shown out of play or a tagged restart/kickoff window: a straight line
   drawn over a stoppage is a path for a ball that was dead, and it would manufacture control
   evidence exactly where the rules say there is none. Interpolation only ever adds evidence, so
   this restriction can only remove false evidence.
2. **Control evidence per frame.** The ball must be located. The nearest player of one team must be
   within 3.0 m of it and at least 0.5 m closer than the nearest opponent. Frames where the ball
   moves faster than 8 m/s (a pass or shot), or stays inside a 1 m box for 1 s either side of the
   frame (a restart being set up), give no evidence.

   The 0.5 m is a **tie-break against positional noise, not a test of a duel.** CHALLENGE.md's
   proxies for controlled possession are temporal ("a player keeps the ball for a few seconds"), and
   the things it says do not count are touches and deflections — which rule 3 below rejects on
   duration. Requiring the nearest opponent to be a full 1.0 m further away on *every* frame, as v4
   did, instead refused evidence whenever a defender pressed inside that margin, so genuinely
   carried ball became held possession or `D`.
3. **Controlled runs.** Consecutive evidence frames of one team form a run. Evidence more than 3 s
   apart does not join. A run is accepted only if part of it is **continuously observed**: a stretch
   of at least 1.5 s with no internal gap longer than 1.0 s. A run that never has such a core is a
   touch, deflection or flicker and is discarded. The run's *extent* is unchanged, so control
   interrupted by a single occlusion is kept. Counting raw evidence frames, as v4 did, let one
   frame of control every few seconds confirm as a run spanning minutes, and labelled the whole span
   for that team.
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
- v5 makes three corrections, each traced to CHALLENGE.md or to the trackers' documented behaviour
  rather than to a score: the per-frame margin becomes a tie-break, a run needs a dense core, and
  interpolation never crosses a stoppage. Before running it we wrote down what would falsify the
  margin change — more than ~400 possessions over the match, or `D` below ~30% of frames. It gave 256
  possessions and 40.6% `D`, so those checks do not reject it. We had expected `D` to fall; it rose,
  because the tighter run rule and the interpolation block remove more evidence than the margin
  change adds. Inside the graded window the predicted `D` share is 21.5% against a true 19.6% (v4:
  22.5%), so the whole-match `D` share is not over-dead — it comes from the second half and the
  pre-kickoff stretches.
- v5 against v4 on this window: macroF1 0.714 vs 0.704, spellF1 0.440 vs 0.471. The spellF1 gap is
  inside the ~0.04 band the scorer documents, and this window has 17 possessions, so we do not treat
  it as evidence that one version ranks above the other. The differences we do treat as informative
  are the structural ones, and they favour v5: 13 possessions of mean 18.1 s against 11 of 21.1 s
  (true: 17 of 14.2 s), 6 matched transitions against 4, latency −4.2 frames against −6.5, dead-ball
  F1 0.512 against 0.501, alive/dead F1 0.874 against 0.867. Two measures moved against v5 and we
  record them: spell precision falls (6 spurious spells against 4) and the home-share error grows
  (0.077 against 0.042), because when a run fails the stricter confirmation, rule 4 leaves the
  previous team in place.

No constant was re-tuned against the public window, and we do not treat its score as evidence that
any version ranks above another; we chose v5 over v4 on the strength of the definition, and confirmed
that the public reference does not contradict it. Its frame-level reading — macroF1, dead-ball F1,
alive/dead F1, both-alive team accuracy — agrees with v5; its spell-level reading is the one that
moved the other way, and it is the noisier of the two at this sample size.

v6 (not submitted) implemented the **second** proxy CHALLENGE.md offers, "a team completes more than
two consecutive passes", as a further source of confirmed runs. Over the whole match it found 1,393
control episodes, 401 completed passes between them and 48 pass chains, and it changed **0 frames**:
315 dense-core runs, 48 chains, 315 runs after merging, and a submission byte-identical to v5's. The
reason is structural, not luck. v5 already builds a run from every evidence frame linked to the next
by a gap of up to 3 s, and accepts that run if any part of it is continuously observed; a pass chain
links the same episodes with the same 3 s limit, so its span always falls inside a v5 run's span. A
chain can only add a possession where v5 rejects the whole group, which requires every pass between
two touches to take longer than 1 s — slower than a real pass. We rejected the rule rather than hunt
for a variant that changes labels: a rule that changes nothing is only added complexity, and one
chosen to move the public score would be tuning.

The idea had looked strong on synthetic data, where a passing move crossed 1.2 s flights and v5 saw
nothing at all. That test was a correct test of the code and a wrong model of the match: a 1.2 s
flight is a 14 m pass at 12 m/s, slow enough to be unusual, and real passes are quick enough that
consecutive touches fall inside v5's 1 s core gap. The place we were wrong is worth recording, because
"why not use the passes proxy?" is the first question this system invites.

## 6. Limitations

- **Possessions are merged.** On the public window v5 reports 13 possessions against 17 true ones,
  mean 18.1 s against 14.2 s. Real changes of possession are missed, most often when control is
  brief; 10 of the 17 true possessions still have no predicted spell matching them at IoU 0.5.
  `spellF1` penalises this heavily, and it is the largest known weakness.
- **Away possession is under-called, and `D` absorbs it.** On the public window 719 of 2,577 true
  away frames are labelled `D` and 100 labelled `H` (v4: 700 and 71). Away precision improved from
  0.806 to 0.852 and away F1 from 0.750 to 0.758, but away recall fell from 0.701 to 0.682.
- **Rejected runs fall back on the previous owner.** Because possession is held through contested
  play (rule 4), a run that fails rule 3 leaves the ball with whoever had it before. On the public
  window that pushes error toward one team: predicted home share 0.649 against a true 0.573.
- **Dead ball is imperfect.** On the public window precision 0.489 and recall 0.538. Over the whole
  match 40.6% of frames are `D`, 256 possessions of mean 12.2 s. Long stretches with no controlled
  run still become `D` when play may be live.
- **The ball is missing most of the time.** DATA.md reports it visible on 56.9% of tactical frames,
  so about 43% are missing, disproportionately in crowded and aerial moments. Control evidence is
  therefore sparse, and possession changes are sometimes detected late (public-window latency: mean
  -6.5 frames).
- **Set-piece windows are inferred, not ground truth.** If a window starts before the ball is
  actually dead, rule 6 adds false `D`. The change is unmeasurable on the public window, so we
  accept it on the strength of the definition rather than on a score.
- **Possession changes are detected from proximity only.** There is no pass or event recognition; the
  pass-based proxy CHALLENGE.md also offers was implemented and changed no frame (section 5), so it
  is not in the submitted system. Aerial duels and fast counter-attacks are the likely failures.
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
python solution/possession_rules_v5.py
python evaluation/validate_submission.py evaluation/results/possession_rules_v5.csv
```

The output is deterministic. Copy `evaluation/results/possession_rules_v5.csv` to `INVICTA.csv` for the
submission form; the CSV is not committed, the repository regenerates it.

Expected whole-match label counts: H 45728, A 32299, D 80830, X 40462; 256 possessions of mean 12.2 s.

Unit tests on synthetic data (no match data): `python solution/test_possession_rules_v5.py`.
