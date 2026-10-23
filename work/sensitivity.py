"""Sensitivity of the full-match output to each parameter of solution/make_submission.py.

The base configuration is used as a pseudo ground truth; every perturbed run is written as a
submission and scored against it with the official scorer (evaluation/run_evaluation.py --gt <csv>).
A perturbation that barely moves the output relative to the base is a parameter the result does
not depend on; one that moves it a lot is a parameter we are guessing.

usage: sensitivity.py            (writes work/sens/, then prints the table)
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "solution"))
from cache_atd import load  # noqa: E402
import make_submission as M  # noqa: E402

SENS = os.path.join(HERE, "sens")
RESULTS = os.path.join(SENS, "results")
os.makedirs(RESULTS, exist_ok=True)

BASE = {k: getattr(M, k) for k in (
    "RADIUS", "MARGIN", "SWITCH_FRAMES", "SLOW_SPEED", "MIN_STOP", "BRIDGE", "MERGE_FRAMES")}

GRID = [
    ("SLOW_SPEED", (1.1, 1.2, 1.4, 1.5)),
    ("SWITCH_FRAMES", (12, 25)),
    ("RADIUS", (2.0, 4.0)),
    ("MARGIN", (0.5, 1.5)),
    ("MIN_STOP", (75, 150)),
    ("BRIDGE", (50, 125)),
    ("MERGE_FRAMES", (50, 100)),
]

players, ball, team = load()


def run(overrides):
    for k, v in BASE.items():
        setattr(M, k, v)
    for k, v in overrides.items():
        setattr(M, k, v)
    labels = M.predict(players, ball, team)
    for k, v in BASE.items():
        setattr(M, k, v)
    return labels


base = run({})
M.write_submission(os.path.join(SENS, "base.csv"), base)
runs = {}
for param, values in GRID:
    for v in values:
        name = "{}={}".format(param, v)
        labels = run({param: v})
        M.write_submission(os.path.join(RESULTS, name + ".csv"), labels)
        runs[name] = labels

graded = base != "X"


def spells(labels):
    d = []
    for a, b in ((11185, 90651), (110808, 190197)):
        s = labels[a:b + 1]
        edges = np.flatnonzero(s[1:] != s[:-1]) + 1
        bounds = np.concatenate(([0], edges, [len(s)]))
        d += [(bounds[i + 1] - bounds[i]) / 25 for i in range(len(bounds) - 1) if s[bounds[i]] in "HA"]
    return np.array(d)


def row(name, labels):
    d = spells(labels)
    g = np.concatenate([labels[a:b + 1] for a, b in ((11185, 90651), (110808, 190197))])
    agree = np.mean(labels[graded] == base[graded])
    return "{:<20} {:>5.1f}% {:>6d} {:>6.1f} {:>6.1f} {:>6.1f}%".format(
        name, 100 * agree, len(d), d.mean(), np.median(d), 100 * np.mean(g == "D"))


print("{:<20} {:>6} {:>6} {:>6} {:>6} {:>7}".format("run", "agree", "poss", "mean", "median", "D share"))
print(row("base", base))
for name, labels in runs.items():
    print(row(name, labels))

print()
out = subprocess.run(
    [sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py"),
     "--gt", os.path.join(SENS, "base.csv"), "--results", RESULTS, "--out", os.path.join(SENS, "out"),
     "--no-detail"], capture_output=True, text=True)
lines = [l for l in out.stdout.splitlines() if l[:4].strip().isdigit() or l.startswith("rank") or l.startswith("---")]
print("\n".join(lines))
if out.returncode:
    print(out.stderr[-1500:])
