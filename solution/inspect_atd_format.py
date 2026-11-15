"""Report the structure of the ATD metadata and raw file, without printing any match data.

Answers one question: does the provided export declare ball-state channels (BallState,
BallOwningTeam) that the position-only readers ignore?

    $env:MTAH_DATA_DIR = "D:\\path\\to\\bundle"      # PowerShell
    python solution/inspect_atd_format.py

Prints only schema information -- element names, declared channel ids and names, the number of
colon-separated fields per raw record, and how many records have each layout. No coordinates, no
per-frame values, nothing that could reconstruct the tracking.
"""

import collections
import os
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, asset_paths, require  # noqa: E402

FIELD_SEP = ":"  # one group per object: players, then the ball


def declared_channels(meta_path):
    """(channel id, name) pairs in file order, and every element tag in the document."""
    root = ET.parse(meta_path).getroot()
    channels = []
    for channel in root.iter("Channel"):
        name = (channel.findtext("Name") or "").strip()
        channels.append((channel.get("id"), name))
    tags = sorted({element.tag for element in root.iter()})
    return channels, tags


def channel_register(meta_path):
    """StringRegister entries, which some exports use to name the channels instead."""
    root = ET.parse(meta_path).getroot()
    return [(register.get("name"), (register.text or "").strip())
            for register in root.iter("StringRegister")]


def raw_layout(raw_path, limit=None):
    """How many colon-separated fields each record has, and the group counts within them."""
    field_counts = collections.Counter()
    group_counts = collections.Counter()
    ball_groups = collections.Counter()
    n = 0
    with open(raw_path) as handle:
        for line in handle:
            if limit is not None and n >= limit:
                break
            if not line.strip():
                continue
            n += 1
            parts = line.rstrip("\n").split(FIELD_SEP)
            field_counts[len(parts)] += 1
            if len(parts) >= 2:
                group_counts[parts[1].count(";") + 1] += 1
            if len(parts) >= 3:
                # "x,y,s" gives 3 comma-separated values; a 4th would be an extra channel.
                ball_groups[len(parts[2].split(","))] += 1
    return n, field_counts, group_counts, ball_groups


def main(game=GAME, data_dir=None):
    paths = asset_paths(game, data_dir)
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    print("metadata: {}".format(os.path.basename(meta_path)))
    print("raw:      {}".format(os.path.basename(raw_path)))

    channels, tags = declared_channels(meta_path)
    print("\nXML element tags ({} distinct)".format(len(tags)))
    for tag in tags:
        print("  {}".format(tag))

    print("\ndeclared <Channel> entries ({}):".format(len(channels)))
    for channel_id, name in channels:
        print("  {:>3}  {}".format(channel_id or "-", name or "(no name)"))

    register = channel_register(meta_path)
    if register:
        print("\n<StringRegister> entries ({}):".format(len(register)))
        for name, text in register:
            print("  {} = {}".format(name, text))

    n, field_counts, group_counts, ball_groups = raw_layout(raw_path)
    print("\nraw records read: {}".format(n))
    print("colon-separated fields per record:")
    for count, records in sorted(field_counts.items()):
        print("  {} fields: {} records".format(count, records))
    print("player groups per record, most common: {}".format(group_counts.most_common(5)))
    print("comma-separated values in the ball field:")
    for count, records in sorted(ball_groups.items()):
        print("  {} values: {} records".format(count, records))

    print("\ninterpretation key")
    print("  'x,y,speed' in a group = position and speed only")
    print("  a 4th comma value, an extra colon field, or a declared channel named")
    print("  BallState/BallOwningTeam means the export carries ball state beyond positions")


if __name__ == "__main__":
    main()
