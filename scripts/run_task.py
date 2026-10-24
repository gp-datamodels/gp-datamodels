#!/usr/bin/env python3
"""Run one line of a campaign task list (scripts/make_campaign.py) into out/runs/<run name>.

    python scripts/run_task.py campaigns/final_rest.tsv 0     # line 0 of the list

Without the line index, the line is $SLURM_ARRAY_TASK_ID, so that a Slurm job
array runs a whole list, for example

    sbatch --array=0-323 --cpus-per-task=4 --mem=8G \\
        --wrap "python scripts/run_task.py campaigns/final_rest.tsv"
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    tasks = Path(sys.argv[1]).read_text().splitlines()
    index = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ["SLURM_ARRAY_TASK_ID"])
    name, bench, method, shard = tasks[index].split("\t")
    script = Path(__file__).resolve().parent / "run_bench.py"
    command = [sys.executable, str(script), "--bench", bench, "--method", method, "--shard", shard,
               "--run-dir", f"out/runs/{name}"]
    print(" ".join(command), flush=True)
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
