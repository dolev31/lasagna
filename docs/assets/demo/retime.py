"""Cut the idle waits out of a recorded asciicast, keeping the scripted pauses.

The recorded run takes about five minutes, most of it embedding the corpus and
reranking. This caps every gap between two output events at ``--max-gap``
seconds and gives each pause that session.sh marked its declared length. It
changes timing only: every byte the commands printed is kept, in order.

    python retime.py raw.cast demo.cast
"""

from __future__ import annotations

import argparse
import json
import re

MARKER = re.compile(r"\x1b\]1337;LasagnaDemoPause=([0-9.]+)\x07")


def retime(lines: list[str], max_gap: float) -> tuple[list[str], int]:
    header = json.loads(lines[0])
    header.pop("idle_time_limit", None)
    out = [json.dumps(header)]
    clock = previous = 0.0
    markers = 0
    for line in lines[1:]:
        if not line.strip():
            continue
        t, kind, data = json.loads(line)
        clock += min(t - previous, max_gap)
        previous = t
        # A marker splits its event: what came before it shows now, what came
        # after it shows once the pause is over.
        pieces = MARKER.split(data)
        for i, piece in enumerate(pieces):
            if i % 2:
                markers += 1
                clock += float(piece)
            elif piece:
                out.append(json.dumps([round(clock, 3), kind, piece]))
    return out, markers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--max-gap", type=float, default=0.7)
    parser.add_argument("--expect-markers", type=int, help="fail unless this many pauses are found")
    args = parser.parse_args()

    with open(args.src, encoding="utf-8") as f:
        lines = f.read().splitlines()
    out, markers = retime(lines, args.max_gap)
    if args.expect_markers is not None and markers != args.expect_markers:
        raise SystemExit(f"found {markers} pause markers, expected {args.expect_markers}")
    with open(args.dst, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
