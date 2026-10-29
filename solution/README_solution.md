# INVICTA: rule-based possession from tracking geometry

More than a Hack 2027, Phase 1. Team **INVICTA**.
Repository: https://github.com/RedSky09/more-than-a-hack-2027

Rule-based, no training, no hand labels. Every label is produced by the rules below from the
tactical ATD and the published match constants alone. The private ground truth is not used. The
public 5-minute ground truth was used to check the pipeline and, to a limited and documented extent,
to choose parameters (see "Where the parameters came from").

## Reproduction

Tested with Python 3.12.0 on Windows. Put the data bundle in `data/` (or set `MTAH_DATA_DIR`), then:

```bash
pip install -r solution/requirements.txt
python solution/make_submission.py --out submission/INVICTA.csv   # about a minute
python evaluation/validate_submission.py submission/INVICTA.csv
```

`submission/` is gitignored, so the file cannot be committed by accident. Without `--out` it goes
to `evaluation/results/submission.csv`. The output is deterministic
and written with `\n` line endings on every platform. Expected result:

- label counts `H 51502, A 36037, D 71318, X 40462` (every `X` is outside the graded halves)
- SHA-256 `035d2478e46b6d4d21b8f7000a7e13630c683ee01153137057b5961a2b2cd5d7`

The match data is gitignored and is not in this repository.

The analysis scripts in `work/` are not needed for the submission. They read a cache built by
`python work/cache_atd.py`; the Smart Tagging cross-checks also need `python work/explore.py` to have
written `work/smart_tagging.csv`.

## Assets used, and not used

| asset | use |
|---|---|
| tactical ATD (players, ball, team of each track slot) | **the only tracking input to the labels** |
| `games.py`, `FAQ.md` (half boundaries, pitch size 105.3 x 68 m, kick-off frames 11185 and 113742) | used |
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
   - the ball is tracked more than 0.5 m outside the pitch. A gap of up to 7 s between such a
     stoppage and the next one is closed: a ball that has just left the pitch is not back in play
     until someone restarts it, and the tracker often finds it again a few metres inside the line
     while it is still being retrieved;
   - the mean speed of all visible players (about 2 s window) stays below 1.3 m/s for at least 4 s.
     Slow stretches less than 3 s apart are merged. This cue does not depend on the ball, which is
     missing on about 41% of graded frames.
3. **Kick-offs.** Everything in a half before its kick-off is `D`. The second half is graded from
   frame 110808 but kicks off at 113742 (FAQ.md), so this covers 117 s. A slow-player stoppage that
   runs into a kick-off is cut there, because the ball is in play from that frame.
4. **Loose and contested balls.** A single touch, a deflection or a duel does not change
   possession. The state changes only after the other team has shown control on 18 frames
   (evidence further than 2 s apart does not accumulate); the change is back-dated to the first of
   those frames. Until then the last possessor keeps the ball, as the challenge definition says.
5. **Touches without control.** An `H`/`A` spell shorter than 3 s sitting directly between two
   spells of the other team is absorbed into them.
6. **Restarts.** After a stoppage the first control evidence sets the possessor immediately
   (and back-fills to the end of the stoppage), because a restart has a clear possessor.
7. **Missing ball.** Frames without a ball never change the state; it is held.
8. **No abstentions inside a half.** A stretch between two stoppages with no control evidence at
   all has no possessor. Rather than `X`, it gets the label the scorer would substitute for `X`
   anyway (the previous label in the half; in practice always `D`). spellF1 is unchanged by this,
   and macroF1 no longer counts those frames (2,897 of them) as certain misses.
9. Frames outside the two graded halves are `X`.

## Where the parameters came from

| parameter | value | basis |
|---|---|---|
| radius, margin | 3 m, 1 m | typical control distance; median nearest-player distance in the public window is 1.6 m in live play. Not tuned. |
| `SLOW_SPEED` | 1.3 m/s | mean player speed is 2.1-2.5 m/s in live play and 1.4 m/s over the public dead-ball frames. **It was also the value with the best dead-ball F1 on the public window** among 1.3, 1.5, 1.7 and 1.9, so the public ground truth did influence it. In the final pipeline 1.2 scores higher there (spellF1 0.581 vs 0.550) and is not adopted: the gap is within noise and chasing it would be tuning on 17 possessions. |
| `SWITCH_FRAMES` | 18 | tried 12, 18 and 25 (and 50 in an early version without stoppage detection, which produced far too few possessions). In an intermediate version 18 scored best of the three on the public window (0.442 vs 0.419 and 0.411), within noise; in the final pipeline 18 and 25 tie (0.550) and 12 is 0.538. Over the full match 12 gives more short possessions and 25 fewer. |
| `MERGE_FRAMES` | 75 frames (3 s) | same public score for 2, 3 and 4 s. 6 s scored higher there (0.508 vs 0.482 in an intermediate version) and was not chosen: over the full match it produced 5 possessions longer than 60 s, against 1 at 3 s. |
| `OFF_BRIDGE` | 175 frames (7 s) | added after inspecting the public window, where one stoppage was split by the ball being briefly re-tracked just inside the line. 100 frames changed nothing there; 175 and 250 gave the same result, so 175 is the smaller of the two. Small effect over the full match (about 1,200 frames). Based on essentially one public segment, so weakly supported. |
| `MIN_STOP`, `BRIDGE` | 100, 75 frames | picked from a handful of public-window runs; the full-match output is insensitive to them (below). |

The public window has 17 possessions and one missed possession moves spellF1 by about 0.06, so these
choices are weakly supported. The bootstrap interval on the public window is wider than any gap
between our variants.

Score on the public window (not a ranking signal): macroF1 0.851, spellF1 0.550, against 0.621 and
0.326 for the published-floor baseline `baseline_smooth50`.

## Checks that do not need the private ground truth

Over the full match the output has 254 possessions, mean 13.8 s, median 9.8 s (public ground truth:
14.2 s over 17), and 45% of graded frames labelled `D`.

`work/crosscheck_tags.py` compares it with the Smart Tagging, which is independent of the public
window. 55 of 58 set pieces have a `D` run in the 4 s before or inside them. Goals are 89% `D` and
kick-offs after a goal 93%. A random 5 s window contains any `D` 53% of the time, against 80-100%
for every restart type. This shows the stoppage detector finds restarts; it does not show that the
boundaries of the stoppages are right.

## Sensitivity

`work/sensitivity.py` moves one parameter at a time and scores the result against the base
configuration with the official scorer (`--gt <base.csv>`). This measures **consistency, not
accuracy**: there is no full-match ground truth to measure the latter against.

| parameter | tried | spellF1 vs base | `D` share of graded frames |
|---|---|---|---|
| `BRIDGE`, `MIN_STOP` | 50/125, 75/150 | 0.96-0.99 | 44-47% |
| `SWITCH_FRAMES` | 12 / 25 | 0.95 / 0.89 | 45% |
| `MARGIN`, `MERGE_FRAMES` | 0.5/1.5, 50/100 | 0.92-0.93 | 45% |
| `RADIUS` | 2 / 4 m | 0.85 / 0.95 | 45% |
| `SLOW_SPEED` | 1.1 / 1.2 / 1.4 / 1.5 m/s | 0.87 / 0.94 / 0.92 / 0.84 | 37 / 41 / 50 / 53% |

The team-assignment parameters barely matter: possession counts stay between 225 and 290 and mean
duration between 12 and 16 s. The result depends mostly on `SLOW_SPEED`, which sets how much of the
match is called dead ball. On the public window lower values score better (1.1: 0.563, 1.2: 0.581,
1.3: 0.550, 1.4: 0.471, 1.5: 0.434), but that window is 20% `D` and is the fastest part of the
match, so it is weak evidence about the right value for the whole match.

## Limitations

- **Late end of stoppages.** The speed cue lags when play restarts: in the public window
  stoppages end 3-4 s late (e.g. frames 16419-16513 and 18408-18480), because player speed recovers
  slowly after a restart. Likewise they can start late when the ball is dead but players are still
  moving. Not fixed: ball speed alone cannot mark the restart, since the ball also moves fast
  while dead (e.g. frames 17380-17440).
- **Player speed is a proxy for a dead ball.** It was calibrated on the first five minutes, the
  fastest part of the match. A slow live passage can be labelled `D` and a quick restart can be
  missed. This is the largest uncertainty in the result.
- **Only the public window was available to check against.** The full-match score is unknown to us.
- **Team after a restart.** For corners the first possessor after the stoppage agrees with the
  Smart Tagging team only 33% of the time (12 cases), against 67% for goal kicks (9 cases).
  Too few cases to act on, and the tag may not name the side taking the restart, so it is not used.
- **Long ball gaps.** When the ball is missing for many seconds inside a possession the state is
  held, so a real hand-over during the gap is detected late or not at all.
- **Nearest player is not control.** Contested balls and aerial duels can still produce a wrong
  switch; the 18-frame rule and the absorption of short spells reduce this but do not remove it.
- **Cross-platform reproduction** was checked on one machine only. Arithmetic is done in float64 to
  make platform-dependent rounding very unlikely to flip a threshold, but it was not run on Linux or
  macOS; the SHA-256 above is there so a mismatch is easy to spot.
- **With more time:** learn the stoppage and hand-over decisions from other matches' tracking with
  corrected labels (allowed by the rules), detect touches from changes in ball velocity (the ATD has
  no ball height), and use the video to resolve long ball gaps.
