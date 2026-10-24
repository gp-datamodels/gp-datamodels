#!/usr/bin/env python3
"""Run one (benchmark, method, shard) and record each unit of work as it finishes.

    python scripts/run_bench.py --bench d24 --method gp --shard seed31 --run-dir out/runs/d24-gp-seed31
    python scripts/run_bench.py --bench attribution --method gp --shard modular_d8_seed31 --run-dir ...
    python scripts/run_bench.py --bench broader --method source --shard atlas_uci_har_clients_d10 --run-dir ...

A resubmitted run skips the units whose metrics.json already exists.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from tnsets.runlog import Run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--shard", default="all")
    parser.add_argument("--units", default="", help="comma-separated subset of units")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    bench = importlib.import_module(f"tnsets.benchmarks.{args.bench}")
    takes_shard = "shard" in inspect.signature(bench.run_unit).parameters
    units = bench.units(args.shard)
    if args.units:
        wanted = set(args.units.split(","))
        units = [u for u in units if u in wanted]
    run = Run(args.run_dir, {"bench": args.bench, "method": args.method, "shard": args.shard,
                             "units": units, "argv": sys.argv})
    failures = 0
    for unit in units:
        if run.done(unit):
            continue
        start = time.time()
        try:
            kwargs = {"shard": args.shard} if takes_shard else {}
            metrics = bench.run_unit(unit, args.method, **kwargs)
        except Exception:  # record and continue: one bad unit must not cost the others
            failures += 1
            traceback.print_exc()
            continue
        metrics["unit_seconds"] = time.time() - start
        run.record(unit, metrics)
        print(f"{unit}: {metrics['unit_seconds']:.1f}s", flush=True)
    run.collect()
    print(f"done: {len(units) - failures}/{len(units)} units", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
