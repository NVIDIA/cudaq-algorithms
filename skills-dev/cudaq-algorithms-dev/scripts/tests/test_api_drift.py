# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise imports and public API changes in disposable Python packages."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "check_api_drift.py"


@pytest.fixture
def source(tmp_path):
    package = tmp_path / "python/cudaq_algorithms"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("from .primitives import Thing\n")
    (package / "primitives.py").write_text(
        "class Thing:\n    def kernel(self): pass\n\n"
        "def helper(): pass\n"
        "# A dynamically assigned public export must import successfully.\n"
        "dynamic = getattr(Thing, 'kernel')\n")
    (tmp_path / "registry.json").write_text(
        json.dumps({
            "features": [{
                "id":
                "primitive",
                "family":
                "primitives",
                "record":
                "references/primitives.md",
                "symbols": [
                    "cudaq_algorithms.Thing.kernel",
                    "cudaq_algorithms.primitives.dynamic"
                ]
            }]
        }))
    (tmp_path / "contract.json").write_text(
        json.dumps({
            "schema_version":
            1,
            "additional_symbols": ["cudaq_algorithms.primitives.helper"],
            "absent": [{
                "id":
                "local-phase",
                "claim":
                "No apply_local_phase API",
                "sources": ["evals/evals.json#contextual-existing-primitive"],
                "name_patterns": ["apply_local_phase"]
            }]
        }))
    return tmp_path


def check(source, **extra_env):
    result = subprocess.run([
        sys.executable,
        str(SCRIPT), "--source-root",
        str(source / "python"), "--registry",
        str(source / "registry.json"), "--contract",
        str(source / "contract.json")
    ],
                            capture_output=True,
                            text=True,
                            env={
                                **os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                                **extra_env
                            })
    return result, json.loads(result.stdout) if result.stdout else {}


def test_documented_imports_and_actual_absence_pass(source):
    result, report = check(source)
    assert result.returncode == 0, result.stderr
    assert report["passed"] is True
    assert report["supported_symbols_checked"] == 3
    assert report["absent_claims_checked"] == 1
    assert report["package_origin"] == str(
        source / "python/cudaq_algorithms/__init__.py")


def test_documented_method_disappearing_fails_with_contract_location(source):
    module = source / "python/cudaq_algorithms/primitives.py"
    module.write_text(module.read_text().replace(
        "def kernel", "def replacement").replace("'kernel'", "'replacement'"))
    result, report = check(source)
    assert result.returncode == 1
    assert report["passed"] is False
    error = next(row for row in report["errors"]
                 if row["check"] == "missing_public_symbol")
    assert error["symbol"] == "cudaq_algorithms.Thing.kernel"
    assert error["records"] == ["references/primitives.md"]


@pytest.mark.parametrize("placement", ["root", "new_module", "new_package"])
def test_absent_api_becomes_public_requires_claim_update(source, placement):
    package = source / "python/cudaq_algorithms"
    if placement == "root":
        path = package / "__init__.py"
        path.write_text(path.read_text() + "def apply_local_phase(): pass\n")
    elif placement == "new_module":
        (package / "new_api.py").write_text("def apply_local_phase(): pass\n")
    else:
        path = package / "new_api"
        path.mkdir()
        (path / "__init__.py").write_text("def apply_local_phase(): pass\n")
    result, report = check(source)
    assert result.returncode == 1
    error = next(row for row in report["errors"]
                 if row["check"] == "absent_api_appeared")
    assert error["claim_id"] == "local-phase"
    assert "apply_local_phase" in error["matches"][0]
    assert "Update" in error["message"]


def test_private_implementation_name_is_not_a_public_absence_violation(source):
    (source / "python/cudaq_algorithms/_private.py"
     ).write_text("def apply_local_phase(): pass\n")
    result, report = check(source)
    assert result.returncode == 0
    assert report["passed"] is True


def test_import_failure_is_not_mistaken_for_absence(source):
    (source / "python/cudaq_algorithms/new_api.py").write_text(
        "import dependency_that_is_not_installed_for_api_drift_test\n"
        "def apply_local_phase(): pass\n")
    result, report = check(source)
    assert result.returncode == 1
    error = next(row for row in report["errors"]
                 if row["check"] == "import_failed")
    assert error["module"] == "cudaq_algorithms.new_api"
    assert "dependency_that_is_not_installed" in error["message"]


def test_requested_source_takes_precedence_over_old_installed_copy(
        source, tmp_path):
    stale = tmp_path / "stale/cudaq_algorithms"
    stale.mkdir(parents=True)
    (stale / "__init__.py"
     ).write_text("raise RuntimeError('stale package imported')\n")
    result, report = check(source, PYTHONPATH=str(stale.parent))
    assert result.returncode == 0, result.stderr
    assert report["package_origin"].startswith(str(source / "python"))


def test_missing_checkout_does_not_fall_back_to_installed_copy(
        source, tmp_path):
    stale = tmp_path / "installed/cudaq_algorithms"
    stale.mkdir(parents=True)
    for name in ("__init__.py", "primitives.py"):
        (stale / name).write_bytes(
            (source / "python/cudaq_algorithms" / name).read_bytes())
    (source / "python/cudaq_algorithms/__init__.py").unlink()
    result, report = check(source, PYTHONPATH=str(stale.parent))
    assert result.returncode == 1
    assert any(row["check"] == "source_root" for row in report["errors"])


def test_declared_export_missing_from_all_is_drift(source):
    module = source / "python/cudaq_algorithms/primitives.py"
    module.write_text(module.read_text() + "__all__ = ['Thing', 'helper']\n")
    result, report = check(source)
    assert result.returncode == 1
    assert any(row["check"] == "missing_public_symbol"
               and row["symbol"] == "cudaq_algorithms.primitives.dynamic"
               for row in report["errors"])


def test_empty_contract_cannot_silently_pass(source):
    (source /
     "contract.json").write_text('{"schema_version": 1, "absent": []}')
    result, report = check(source)
    assert result.returncode == 1
    assert any(row["check"] == "contract" for row in report["errors"])


def test_malformed_absence_pattern_is_a_structured_contract_error(source):
    path = source / "contract.json"
    contract = json.loads(path.read_text())
    contract["absent"][0]["name_patterns"] = ["["]
    path.write_text(json.dumps(contract))
    result, report = check(source)
    assert result.returncode == 1
    assert any(row["check"] == "contract" for row in report["errors"])


def test_absent_class_method_appearing_requires_claim_update(source):
    path = source / "contract.json"
    contract = json.loads(path.read_text())
    contract["absent"][0].pop("name_patterns")
    contract["absent"][0]["symbols"] = [
        "cudaq_algorithms.Thing.controlled_kernel"
    ]
    path.write_text(json.dumps(contract))
    result, report = check(source)
    assert result.returncode == 0
    module = source / "python/cudaq_algorithms/primitives.py"
    module.write_text(module.read_text() +
                      "Thing.controlled_kernel = lambda self: None\n")
    result, report = check(source)
    assert result.returncode == 1
    assert any(
        row["check"] == "absent_api_appeared"
        and row["matches"] == ["cudaq_algorithms.Thing.controlled_kernel"]
        for row in report["errors"])


@pytest.mark.parametrize("public_name",
                         ["PhaseEstimation", "phase_estimation"])
def test_known_phase_estimation_names_require_claim_update(
        source, public_name):
    production_contract = json.loads(
        (SCRIPT.parents[1] / "coverage/api_contract.json").read_text())
    contract_path = source / "contract.json"
    contract = json.loads(contract_path.read_text())
    contract["absent"] = [
        row for row in production_contract["absent"]
        if row["id"] == "qpe-primitive"
    ]
    contract_path.write_text(json.dumps(contract))
    result, report = check(source)
    assert result.returncode == 0, report

    package = source / "python/cudaq_algorithms"
    if public_name == "PhaseEstimation":
        # A neutral module name isolates the exported class spelling.
        (package / "candidate.py").write_text("class PhaseEstimation: pass\n")
        root = package / "__init__.py"
        root.write_text(root.read_text() +
                        "from .candidate import PhaseEstimation\n")
    else:
        # A neutral export isolates the known public module spelling.
        (package / "phase_estimation.py").write_text("def run(): pass\n")

    result, report = check(source)
    assert result.returncode == 1, report
    error = next(row for row in report["errors"]
                 if row["check"] == "absent_api_appeared")
    assert error["claim_id"] == "qpe-primitive"
    assert f"cudaq_algorithms.{public_name}" in error["matches"]
    assert "Update the skill records/evaluation assertions" in error["message"]
    assert "not a permanent implementation ban" in error["message"]
