#!/usr/bin/env python3
"""Local and CI entry point for standardized multi-model evaluations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
from runner.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
