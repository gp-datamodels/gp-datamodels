"""Benchmark protocols of the source paper, reproduced on its released cubes.

Each module exposes ``units(shard)`` and ``run_unit(unit, method)``. A method
is a preset name from :mod:`tnsets.methods`, or ``"source"`` for the source's
own released fitters where no saved result exists for the exact protocol.
"""
from pathlib import Path

TNDM = Path(__file__).resolve().parents[3] / "third_party" / "TNDM"

BENCHMARKS = ("d24", "d8", "attribution", "sportmusic", "broader", "d12", "probe")
