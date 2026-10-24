#!/usr/bin/env python3
"""Write a campaign's task list: one line per (benchmark, method, shard) run.

    python scripts/make_campaign.py final d24 > campaigns/final_d24.tsv
    python scripts/make_campaign.py final rest > campaigns/final_rest.tsv
    python scripts/make_campaign.py source > campaigns/source.tsv
    python scripts/make_campaign.py empty > campaigns/calibration_empty.tsv

Each line is ``<run name>\\t<bench>\\t<method>\\t<shard>``; scripts/run_task.py
runs one line. "final" is the proposed model and its variants on every
benchmark, with the calibration study; "source" reruns the source's released
fitters where it saved no result for the exact protocol (and refits its d = 24
models); "empty" repeats the calibration of the three broader cubes with the
empty coalition observed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tnsets.benchmarks import attribution, broader, calibration  # noqa: E402

SHARDS = {
    "d24": ["seed31", "seed43", "seed59"],
    "d12": ["seed43", "seed59"],
    "d8": ["all"],
    "attribution": list(attribution.CUBES),
    "sportmusic": ["all"],
    "broader": list(broader.CUBES),
    "probe": ["qwen_probe", "uci_har"],
}
SOURCE_RUNS = [("sportmusic", "all"), ("probe", "uci_har")] + [("broader", c) for c in broader.CUBES]
CALIBRATION_METHODS = ["gp", "gp-r1", "gp-stationary"]
MAIN = ["gp"]
ABLATIONS = ["gp-r1", "gp-stationary", "gp-iso", "gp-flat", "gp-r2direct", "gp-tau0.5", "gp-tau2"]


def tasks(campaign: str) -> list[tuple[str, str, str, str]]:
    out = []
    if campaign == "final":
        for method in MAIN + ABLATIONS:
            for bench, shards in SHARDS.items():
                for shard in shards:
                    out.append((f"{bench}-{method}-{shard}", bench, method, shard))
        for method in CALIBRATION_METHODS:
            for shard in calibration.SHARDS:
                out.append((f"calibration-{method}-{shard}", "calibration", method, shard))
    elif campaign == "source":
        for bench, shard in SOURCE_RUNS + [("d24", s) for s in SHARDS["d24"]]:
            out.append((f"{bench}-source-{shard}", bench, "source", shard))
    elif campaign == "empty":
        for cube in broader.CUBES:
            out.append((f"calibration-gp-{cube}-empty", "calibration", "gp", f"{cube}:empty"))
    else:
        raise SystemExit(f"unknown campaign {campaign!r}: final, source or empty")
    return out


if __name__ == "__main__":
    # optional second argument: "d24" keeps only the d = 24 tasks, "rest" the others
    rows = tasks(sys.argv[1] if len(sys.argv) > 1 else "final")
    part = sys.argv[2] if len(sys.argv) > 2 else "all"
    for row in rows:
        if part == "all" or (part == "d24") == (row[1] == "d24"):
            print("\t".join(row))
