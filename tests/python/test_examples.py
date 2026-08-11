# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Every documentation example must run, and its self-checks must pass.

The examples under ``docs/sphinx/examples/python/`` are self-verifying:
each checks its output against an independent reference and exits nonzero
on any discrepancy. Executing them here makes CI enforce that — an example
cannot rot silently once it is rendered into the documentation.

The examples run on the fp64 CPU simulator (several pin it themselves;
the environment covers the rest — fp32 defaults would fail their
tolerances). ``df_encoding.py`` is excluded: it is the bring-your-own
encoding *module*, imported by its sibling demo and by
``test_df_encoding.py``, not a standalone script. Dependencies: two
examples need ``pyscf`` and two need ``qsppack``; both are installed by
every CI lane, and the examples fail with a one-line install hint rather
than a traceback if they are missing.
"""

import os
import pathlib
import subprocess
import sys

import pytest

_EXAMPLES_DIR = (pathlib.Path(__file__).resolve().parents[2] / "docs" /
                 "sphinx" / "examples" / "python")
_NOT_STANDALONE = {"df_encoding.py"}

EXAMPLES = sorted(path.name for path in _EXAMPLES_DIR.glob("*.py")
                  if path.name not in _NOT_STANDALONE)


def test_examples_discovered():
    # The glob must find the examples: an empty parametrization would
    # silently pass while covering nothing (e.g. after a directory move).
    assert len(EXAMPLES) >= 15


@pytest.mark.parametrize("name", EXAMPLES)
def test_example_runs_and_self_verifies(name):
    env = dict(os.environ)
    env.setdefault("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu")
    result = subprocess.run(
        [sys.executable, str(_EXAMPLES_DIR / name)],
        capture_output=True,
        text=True,
        timeout=600,
        env=env)
    assert result.returncode == 0, (
        f"{name} exited {result.returncode}\n"
        f"--- stdout (tail) ---\n{result.stdout[-2000:]}\n"
        f"--- stderr (tail) ---\n{result.stderr[-2000:]}")
