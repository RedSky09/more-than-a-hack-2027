"""Explain every possession the public ground truth has and a submission does not.

This reads only committed files -- `evaluation/gt/*_public.csv` and `evaluation/results/*.csv` -- so
it needs no match data. For each ground-truth possession it prints the reference span, what the
submission predicted inside it, and the best overlapping same-team spell, which together separate
the three ways a possession gets missed: too little evidence to confirm it, a dead-ball stretch
declared inside it, or a detected possession with the wrong boundary.

    python evaluation/diagnose_misses.py                              # the submitted system
    python evaluation/diagnose_misses.py evaluation/results/x.csv
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import possession_metrics as pm  # noqa: E402

FPS = 25.0


def read_labels(path):
    return pm.label_codes(pm.read_label_csv(path))


def spelling(codes, start, stop):
    """'H0-40 A40-100' -- the submission's labels inside [start, stop), with offsets in frames."""
    window = codes[start:stop]
    breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(window)]))
    return " ".join("{}{}-{}".format(pm.LABELS[window[bounds[i]]], bounds[i], bounds[i + 1])
                    for i in range(len(bounds) - 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission", nargs="?", default=os.path.join(
        HERE, "results", "possession_rules_v5.csv"),
        help="results CSV to diagnose (default: the submitted system)")
    parser.add_argument("--gt", default=None, help="ground truth CSV (default: the public one)")
    parser.add_argument("--iou", type=float, default=pm.SPELL_IOU_HEADLINE)
    args = parser.parse_args()

    gt_path = args.gt or (glob.glob(os.path.join(HERE, "gt", "*_gt_public.csv")) or [None])[0]
    if not gt_path:
        raise SystemExit("no public ground truth in {}".format(os.path.join(HERE, "gt")))

    gt_codes = read_labels(gt_path)
    pred_codes = read_labels(args.submission)
    runs = pm.valid_runs(gt_codes)
    filled = pm.fill_abstentions(pred_codes, runs)
    gt_spells = pm.possession_spells(gt_codes, runs)
    pred_spells = pm.possession_spells(filled, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    matched = pm.matched_gt_flags(overlaps, len(gt_spells), args.iou)

    print("{} vs {}".format(os.path.basename(args.submission), os.path.basename(gt_path)))
    print("graded runs {}, possessions gt {} pred {}, matched at IoU {}: {}\n".format(
        runs, len(gt_spells), len(pred_spells), args.iou, int(matched.sum())))

    counts = {"unconfirmed": 0, "dead": 0, "boundary": 0}
    for index, (start, stop, code) in enumerate(gt_spells):
        team = pm.LABELS[code]
        other_code = pm.A if team == "H" else pm.H
        window = filled[start:stop]
        same, opp, dead = (float((window == code).mean()), float((window == other_code).mean()),
                           float((window == pm.D).mean()))
        inside = [(iou, pred_spells[j]) for iou, i, j in overlaps if i == index]
        verdict = "matched" if matched[index] else "MISSED"
        print("{:<7} #{:<2} {} {:>6}-{:<6} {:>5.1f}s  same {:.2f} opp {:.2f} dead {:.2f}".format(
            verdict, index, team, start, stop, (stop - start) / FPS, same, opp, dead))
        if not matched[index]:
            if dead > 0.4:
                counts["dead"] += 1
                cause = "dead ball declared inside it"
            elif not inside or inside[0][0] < args.iou / 2:
                counts["unconfirmed"] += 1
                cause = "no same-team spell of this possession at all"
            else:
                counts["boundary"] += 1
                cause = "detected, wrong boundary"
            print("        -> {}".format(cause))
        if inside:
            iou, (pred_start, pred_stop, _) = inside[0]
            print("        best {}-spell {} ({} frames) IoU {:.3f}".format(
                team, (pred_start, pred_stop), pred_stop - pred_start, iou))
        print("        pred: {}".format(spelling(filled, start, stop)))

    dead_inside = sum(int((filled[s:e] == pm.D).sum()) for s, e, _ in gt_spells)
    dead_window = sum(int((filled[a:b] == pm.D).sum()) for a, b in runs)
    print("\nmissed possessions by cause: {}".format(counts))
    print("dead-ball frames inside a true possession: {} of {} in the graded window".format(
        dead_inside, dead_window))


if __name__ == "__main__":
    main()
