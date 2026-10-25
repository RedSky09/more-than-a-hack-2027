# Solution: rule-based possession from tracking geometry

Rule-based, no training, no hand labels. Every label is produced by the rules below from the
tactical ATD alone. The private ground truth is not used; the public 5-minute ground truth was used
only to check the pipeline and to sanity-check parameters (see "Where the parameters came from").

## Reproduction

Tested with Python 3.12.0. Put the data bundle in `data/` (or set `MTAH_DATA_DIR`), then:

```bash
pip install -r solution/requirements.txt
python solution/make_submission.py                              # about a minute
python evaluation/validate_submission.py evaluation/results/submission.csv
```

This writes `evaluation/results/submission.csv`. The file is deterministic; its label counts are
`H 52419, A 36170, D 66896, X 43834`. Rename it to `<team_name>.csv` before uploading. The match
data is gitignored and is not in this repository.

## Assets used, and not used

| asset | use |
|---|---|
| tactical ATD (players, ball, team of each track slot) | **the only input to the labels** |
| `games.py` (half boundaries, pitch size 105.3 x 68 m) | used |
| Smart Tagging | **not used for the labels.** Used only to cross-check the stoppage detector (see below) |
| homography, tactical and broadcast video, broadcast ATD | not used |

Tried and dropped:
- Smart Tagging `Team` as a possession prior: on the public window only `SET PIECES`, `LONG BALL`
  and `PROGRESSION` agreed with the ground truth above 70%, `DEFENDING IN ...` tags were inverted,
  and the rest were near 50%. Too inconsistent to use.
- A per-minute adaptive speed threshold for stoppages: it lowered the dead-ball share of the
  match to 29%, which is implausibly low, and lowered macroF1 on the public window.
- A plain nearest-player majority vote (the provided baseline): it fragments possession or erases
  short ones depending on the window.

## Method and possession rules

1. **Control evidence.** On frames where the ball is tracked, a team shows control when its nearest
   player is within 3 m of the ball and at least 1 m closer than the other team's nearest player.
2. **Stoppages (`D`).** Either cue is enough:
   - the ball is tracked more than 0.5 m outside the pitch;
   - the mean speed of all visible players (about 2 s window) stays below 1.3 m/s for at least 4 s.
     Slow stretches less than 3 s apart are merged. This cue does not depend on the ball, which is
     missing on about 41% of graded frames.
3. **Loose and contested balls.** A single touch, a deflection or a duel does not change
   possession. The state changes only after the other team has shown control on 18 frames
   (evidence further than 2 s apart does not accumulate); the change is back-dated to the first of
   those frames. Until then the last possessor keeps the ball, as the challenge definition says.
4. **Touches without control.** An `H`/`A` spell shorter than 3 s sitting directly between two
   spells of the other team is absorbed into them.
5. **Restarts.** After a stoppage the first control evidence sets the possessor immediately
   (and back-fills to the end of the stoppage), because a restart has a clear possessor.
6. **Missing ball.** Frames without a ball never change the state; it is held.
7. Frames outside the two graded halves are `X`. Inside the halves, `X` appears only between a
   stoppage (or the start of the half) and the first control evidence after it.

## Where the parameters came from

| parameter | value | basis |
|---|---|---|
| radius, margin | 3 m, 1 m | typical control distance; median nearest-player distance in the public window is 1.6 m in live play. Not tuned. |
| `SLOW_SPEED` | 1.3 m/s | mean player speed is 2.1-2.5 m/s in live play and 1.4 m/s over the public dead-ball frames. **It was also the value with the best dead-ball F1 on the public window** among the values tried (1.3, 1.5, 1.7, 1.9), so the public ground truth did influence it. 1.2 gave a higher spellF1 there (0.509 vs 0.482) and was not chosen, because the difference is within noise. |
| `SWITCH_FRAMES` | 18 | tried 12, 18, 25 (and 50 in an earlier version without stoppage detection, which produced far too few possessions). 18 also scored best of those on the public window (0.442 vs 0.419 and 0.411), a difference within noise; 12 gave many short possessions over the full match and 25 few. |
| `MERGE_FRAMES` | 75 | same public score for 2, 3 and 4 s. |
| `MIN_STOP`, `BRIDGE` | 100, 75 | picked from a handful of public-window runs; the full-match output is insensitive to them (below). |

The public window has 17 possessions and one missed possession moves spellF1 by about 0.06, so these
choices are weakly supported. The bootstrap interval on the public window is wider than any gap
between our variants.

## Checks that do not need the private ground truth

Over the full match the output has 265 possessions, mean 13.4 s, median 9.5 s (public window:
14.2 s), 144 stoppages, and 42% of graded frames labelled `D`.

`work/crosscheck_tags.py` compares it with the Smart Tagging, which is independent of the public
window. 55 of 58 set pieces have a `D` run in the 4 s before or inside them. Kick-offs after a goal
are 93% `D`, goals 80%. A random 5 s window contains any `D` 52% of the time, against 80-100% for
every restart type. This shows the stoppage detector finds restarts; it does not show that the
boundaries of the stoppages are right.

## Sensitivity

`work/sensitivity.py` moves one parameter at a time and scores the result against the base
configuration with the official scorer (`--gt <base.csv>`). This measures **consistency, not
accuracy**: there is no full-match ground truth to measure the latter against.

| parameter | tried | spellF1 vs base | `D` share of graded frames |
|---|---|---|---|
| `BRIDGE`, `MIN_STOP` | 50/125, 75/150 | 0.96-0.98 | 41-44% |
| `SWITCH_FRAMES` | 12 / 25 | 0.95 / 0.90 | 42% |
| `MARGIN`, `MERGE_FRAMES` | 0.5/1.5, 50/100 | 0.92-0.94 | 42% |
| `RADIUS` | 2 / 4 m | 0.84 / 0.96 | 42% |
| `SLOW_SPEED` | 1.1 / 1.2 / 1.4 / 1.5 m/s | 0.85 / 0.93 / 0.91 / 0.82 | 32 / 37 / 47 / 51% |

The team-assignment parameters barely matter: possession counts stay between 230 and 300 and mean
duration between 12 and 15 s. The result depends mostly on `SLOW_SPEED`, which sets how much of the
match is called dead ball. On the public window the score falls as it rises (1.2: 0.509, 1.3: 0.482,
1.4: 0.411, 1.5: 0.378), but that window is 20% `D` and is the fastest part of the match, so it is
weak evidence about the right value for the whole match.

## Limitations

- **Player speed is a proxy for a dead ball.** It was calibrated on the first five minutes, the
  fastest part of the match. A slow live passage can be labelled `D` and a quick restart can be
  missed. This is the largest uncertainty in the result.
- **Only the public window was available to check against.** The full-match score is unknown to us.
- **Team after a restart.** For corners the first possessor after the stoppage agrees with the
  Smart Tagging team only 33% of the time (12 cases), against 78% for goal kicks (9 cases).
  Too few cases to act on, so the tag is not used.
- **Long ball gaps.** When the ball is missing for many seconds inside a possession the state is
  held, so a real hand-over during the gap is detected late or not at all.
- **Nearest player is not control.** Contested balls and aerial duels can still produce a wrong
  switch; the 18-frame rule and the absorption of short spells reduce this but do not remove it.
- **With more time:** learn the stoppage and hand-over decisions from other matches' tracking with
  corrected labels (allowed by the rules), use ball height/velocity changes to detect touches, and
  use the video to resolve long ball gaps.
