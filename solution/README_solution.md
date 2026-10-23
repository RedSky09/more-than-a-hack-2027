# Solution: rule-based possession from tracking geometry

Regenerate the submission (numpy only, about a minute):

```bash
python solution/make_submission.py        # -> evaluation/results/submission.csv
python evaluation/validate_submission.py evaluation/results/submission.csv
```

No ground truth is read, no hand labels are used and nothing is trained. Every label is the output
of the rules below, applied to the tactical ATD only.

## Method

1. **Control evidence.** On frames where the ball is tracked, a team shows control when its nearest
   player is within 3 m of the ball and at least 1 m closer than the other team's nearest player.
2. **Stoppages (`D`).** Either cue is enough:
   - the ball is tracked more than 0.5 m outside the pitch;
   - the mean speed of all visible players (about 2 s window) stays below 1.3 m/s for at least 4 s.
     Slow stretches less than 3 s apart are merged. This cue does not depend on the ball, which is
     missing on about 41% of graded frames.
3. **Possession (`H`/`A`).** A sticky state per half. It changes only after the other team has shown
   control on 18 frames (evidence further than 2 s apart does not accumulate), and the change is
   back-dated to the first of those frames. Frames without a ball hold the state. After a stoppage the
   first control evidence sets the possessor immediately, because a restart has a clear possessor.
4. **Touches are not changes of possession.** An `H`/`A` spell shorter than 3 s that sits directly
   between two spells of the other team is absorbed into them.
5. Frames outside the two graded halves are `X`.

## Where the parameters came from

| parameter | value | basis |
|---|---|---|
| radius, margin | 3 m, 1 m | typical control distance; nearest-player distance in the public window is a median 1.6 m during play |
| `SLOW_SPEED` | 1.3 m/s | mean player speed is 2.1-2.5 m/s in live play and 1.4 m/s over the public dead-ball frames (1.1-2.0 per segment) |
| `SWITCH_FRAMES` | 18 | middle of 12 / 18 / 25, which were within noise of each other on the public window; 12 gave too many short possessions over the full match and 25 too few |
| `MERGE_FRAMES` | 75 | same score for 2, 3 and 4 s on the public window |

The public window has 17 possessions, so it was used to check that the pipeline is correct and to
reject clearly bad settings, not to rank variants (see `EVALUATION.md`).

## Checks that do not need the private ground truth

Over the full match the output has 265 possessions, mean 13.4 s, median 9.5 s (the public window:
14.2 s), 144 stoppages and 42% of graded frames labelled `D`.

`work/crosscheck_tags.py` compares it with the Smart Tagging, which is independent of the public
window. 55 of 58 set pieces have a `D` run in the 4 s before or inside them. Restart types behave
as expected: kick-offs after a goal are 93% `D`, goals 80%. Against 52% for a random 5 s window,
"contains any `D`" is 80-100% for every restart type.

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
match is called dead ball. On the public window the score falls monotonically as it rises (1.2:
0.509, 1.3: 0.482, 1.4: 0.411, 1.5: 0.378), but that window is 20% `D` and is the fastest part of
the match, so it is not evidence about the right value for the whole match. 1.3 m/s is kept because
it follows from the player-speed statistics, not because it scored best.

## Known weaknesses

- **Player speed is a proxy for a dead ball.** It was calibrated on the first five minutes, the
  fastest part of the match. A slow live passage can be labelled `D` and a quick restart can be
  missed. Using an adaptive threshold lowered the full-match `D` share to 29%, which is implausibly
  low, so a fixed threshold was kept.
- **Team after a restart.** For corners the first possessor after the stoppage agrees with the
  Smart Tagging team only 33% of the time (12 cases), against 78% for goal kicks (9 cases). Too few
  cases to act on, and the tag may not name the attacking side. The Smart Tagging team is
  deliberately not used.
- **Long ball gaps.** When the ball is missing for many seconds inside a possession the state is
  simply held, so a real hand-over during the gap is detected late or not at all.
- **Nearest player is not control.** Contested balls and aerial duels can still produce a wrong
  switch; the 18-frame rule and the absorption of short spells reduce this but do not remove it.
