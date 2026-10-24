"""Run directories: config.json, env.json and metrics.json, three files per run.

A run is one (benchmark, method, shard). Its units of work (a seed, a budget,
a repetition) are recorded in ``metrics.json`` under "units", the file being
rewritten atomically after each unit, so a job killed at its wall time loses
at most one unit and a resubmit skips the finished ones. One file per run,
rather than one per unit, keeps campaigns of thousands of units within the
file-count quotas of shared file systems.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import sys
import time
from pathlib import Path


def _git_stamp() -> dict:
    try:
        import subprocess
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
        return {"commit": commit, "dirty": str(dirty).lower()}
    except OSError:
        return {}


def environment() -> dict:
    import numpy
    import torch
    return {"host": socket.gethostname(), "platform": platform.platform(), "python": sys.version.split()[0],
            "numpy": numpy.__version__, "torch": torch.__version__, "threads": torch.get_num_threads(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "git": _git_stamp(), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True, default=float) + "\n")
    tmp.replace(path)


class Run:
    def __init__(self, run_dir: str | Path, config: dict):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.dir / "config.json", config)
        _write_json(self.dir / "env.json", environment())
        self.path = self.dir / "metrics.json"
        self.units = json.loads(self.path.read_text())["units"] if self.path.exists() else {}

    def done(self, unit: str) -> bool:
        return unit in self.units

    def record(self, unit: str, metrics: dict) -> None:
        self.units[unit] = metrics
        _write_json(self.path, {"units": self.units})

    def collect(self) -> dict:
        _write_json(self.path, {"units": self.units})
        return self.units
