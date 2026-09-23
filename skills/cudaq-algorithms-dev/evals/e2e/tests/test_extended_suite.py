# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The coverage completion suite is isolated from historical experiments."""
import importlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_completion_cli_keeps_historical_registry_and_paired_staging(
        tmp_path, monkeypatch):
    import run
    from test_runner import fixture_source
    suite = importlib.import_module("extended")
    source = fixture_source(tmp_path / "source", "skill")
    monkeypatch.setattr(run, "REPO", source)
    monkeypatch.setattr(run.subprocess, "check_output",
                        lambda *a, **kw: "revision\n")
    monkeypatch.setattr(run, "runtime_metadata", lambda p: {"executable": p})
    before, label = dict(run.CASES), run.SUITE
    destination = tmp_path / "campaign"
    suite.main([
        "prepare", "--output",
        str(destination), "--python", sys.executable, "--repetitions", "1",
        "--cases", "df_spin_diagnostics,psi4_energy"
    ])
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["suite"] == "coverage_completion"
    assert len(manifest["runs"]) == 4
    assert {r["arm"] for r in manifest["runs"]} == {"baseline", "skill"}
    assert {c["id"]
            for c in manifest["cases"]
            } == {"df_spin_diagnostics", "psi4_energy"}
    assert run.CASES == before and run.SUITE == label
    with pytest.raises(RuntimeError, match="suite"):
        run.main(["report", "--output", str(destination)])
    suite.main(["report", "--output", str(destination)])
    assert run.CASES == before and run.SUITE == label


def test_every_uncovered_record_has_a_direct_scientific_contract():
    suite = importlib.import_module("extended")
    root = Path(__file__).resolve().parents[3]
    registry = json.loads((root / "coverage/features.json").read_text())
    required = {
        symbol
        for _, spec in suite.case_registry().values()
        for symbol in spec["required_public_apis"]
    }
    for feature in registry["features"]:
        has_old_current_evidence = any(e["campaign"].endswith("refactored")
                                       and e["status"] == "scientific_pass"
                                       for e in feature["executions"])
        if not has_old_current_evidence:
            assert required.intersection(feature["symbols"]), feature["id"]
