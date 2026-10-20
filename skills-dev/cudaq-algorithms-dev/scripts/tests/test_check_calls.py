# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline contract tests for the source-only runtime signature probe.

These tests exercise parsing and argument binding, not numerical correctness.
Neither submitted snippets nor inspected package code may be executed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "skills/cudaq-algorithms/scripts/check_calls.py"


@pytest.fixture(scope="module")
def checker():
    path = SCRIPT
    spec = importlib.util.spec_from_file_location("isolated_signature_probe", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module.check_source
    finally:
        sys.modules.pop(spec.name, None)


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "cudaq_algorithms"
    (root / "stateprep").mkdir(parents=True)
    (root / "__init__.py").write_text(
        "from . import stateprep\n"
        "from .stateprep import hartree_fock as prepare_reference\n"
    )
    (root / "stateprep" / "__init__.py").write_text(
        "from ._kernels import hartree_fock, uccsd, binding, flexible, wrapped, Factory\n"
    )
    (root / "stateprep" / "_kernels.py").write_text(textwrap.dedent("""\
        import cudaq

        @cudaq.kernel
        def hartree_fock(qubits, num_electrons):
            raise AssertionError("Source function must never execute")

        @cudaq.kernel
        def uccsd(qubits, thetas, num_electrons, spin):
            raise AssertionError("Source function must never execute")

        def binding(x, /, y=2, *, required, flag=False):
            raise AssertionError("Source function must never execute")

        def flexible(x, *items, option=True, **extra):
            raise AssertionError("Source function must never execute")

        @signature_changing_decorator
        def wrapped(x):
            raise AssertionError("Source function must never execute")

        class Factory:
            def __init__(self, x):
                raise AssertionError("Source constructor must never execute")
        """))
    return root


def assert_classification(result, line, category):
    """Require one diagnostic for the target call, without fixing prose wording."""
    assert all(isinstance(result[key], list) for key in ("checked", "errors", "unknown"))
    found = [
        (key, entry)
        for key in ("checked", "errors", "unknown")
        for entry in result[key]
        if entry["line"] == line
    ]
    assert len(found) == 1, result
    actual, entry = found[0]
    assert actual == category, result
    assert entry["call"], entry
    assert entry["name"], entry
    assert "evidence" in entry, entry
    if category != "unknown":
        assert entry["evidence"], entry
    if category != "checked":
        assert entry["reason"], entry
    return entry


@pytest.mark.parametrize(
    "call,category",
    [
        ("sp.hartree_fock(q, n)", "checked"),
        ("sp.hartree_fock(qubits=q, num_electrons=n)", "checked"),
        ("sp.hartree_fock(q, n, spin)", "errors"),
        ("sp.hartree_fock(q)", "errors"),
        ("sp.hartree_fock(q, n, spin=0)", "errors"),
        ("sp.hartree_fock(q, n, num_electrons=n)", "errors"),
        ("sp.uccsd(q, theta, n, spin)", "checked"),
        ("sp.uccsd(q, theta, n)", "errors"),
    ],
)
def test_hf_and_uccsd_bind_arguments_from_kernel_source(checker, package, call, category):
    result = checker("from cudaq_algorithms import stateprep as sp\n" + call, package)
    assert_classification(result, 2, category)


@pytest.mark.parametrize(
    "source",
    [
        "import cudaq_algorithms as ca\nca.stateprep.hartree_fock(q, n)",
        "import cudaq_algorithms.stateprep as sp\nsp.hartree_fock(q, n)",
        "import cudaq_algorithms.stateprep\ncudaq_algorithms.stateprep.hartree_fock(q, n)",
        "from cudaq_algorithms import stateprep\nstateprep.hartree_fock(q, n)",
        "from cudaq_algorithms.stateprep import hartree_fock as hf\nhf(q, n)",
        "from cudaq_algorithms import prepare_reference\nprepare_reference(q, n)",
    ],
)
def test_explicit_import_aliases_and_relative_reexports_resolve(checker, package, source):
    assert_classification(checker(source, package), 2, "checked")


@pytest.mark.parametrize(
    "call,category",
    [
        ("sp.binding(1, required=3)", "checked"),
        ("sp.binding(1, 2, required=3, flag=True)", "checked"),
        ("sp.binding(1, y=2, required=3)", "checked"),
        ("sp.binding(1)", "errors"),
        ("sp.binding(x=1, required=3)", "errors"),
        ("sp.binding(1, 2, 3, required=4)", "errors"),
        ("sp.binding(1, 2, y=3, required=4)", "errors"),
        ("sp.binding(1, required=3, nonexistent=True)", "errors"),
        ("sp.flexible(1, 2, 3, option=False, arbitrary=True)", "checked"),
        ("sp.flexible(option=False)", "errors"),
    ],
)
def test_python_argument_binding_rules(checker, package, call, category):
    assert_classification(checker("from cudaq_algorithms import stateprep as sp\n" + call, package), 2, category)


@pytest.mark.parametrize(
    "call",
    [
        "sp.hartree_fock(*args)",
        "sp.hartree_fock(q, **kwargs)",
        "sp.uccsd(*args, **kwargs)",
    ],
)
def test_dynamic_unpacking_is_unknown(checker, package, call):
    assert_classification(checker("from cudaq_algorithms import stateprep as sp\n" + call, package), 2, "unknown")


@pytest.mark.parametrize("call", ["sp.missing_api(q)", "sp.wrapped(1)", "sp.Factory(1)"])
def test_unresolved_decorated_and_constructor_calls_are_unknown(checker, package, call):
    assert_classification(checker("from cudaq_algorithms import stateprep as sp\n" + call, package), 2, "unknown")


@pytest.mark.parametrize(
    "source,line",
    [
        ("from cudaq_algorithms.stateprep import hartree_fock as hf\nhf = replacement\nhf(q, n, spin)", 3),
        ("from cudaq_algorithms import stateprep as sp\nsp = replacement\nsp.hartree_fock(q, n, spin)", 3),
        ("from cudaq_algorithms.stateprep import hartree_fock as hf\ndef caller(hf):\n    hf(q, n, spin)", 3),
        ("from cudaq_algorithms import stateprep as sp\ndef caller(sp):\n    sp.hartree_fock(q, n, spin)", 3),
    ],
)
def test_shadowed_imports_do_not_generate_false_signature_errors(checker, package, source, line):
    assert_classification(checker(source, package), line, "unknown")


def test_valid_outer_alias_remains_resolvable_after_local_shadow(checker, package):
    source = (
        "from cudaq_algorithms.stateprep import hartree_fock as hf\n"
        "def caller(hf):\n"
        "    hf(q, n, spin)\n"
        "hf(q, n)\n"
    )
    result = checker(source, package)
    assert_classification(result, 3, "unknown")
    assert_classification(result, 4, "checked")


def test_calls_inside_submitted_kernel_keep_module_import_resolution(checker, package):
    source = (
        "import cudaq\n"
        "from cudaq_algorithms import stateprep\n"
        "@cudaq.kernel\n"
        "def prepare(qubits):\n"
        "    stateprep.hartree_fock(qubits, NUM_ELECTRONS, SPIN)\n"
        "    stateprep.uccsd(qubits, THETAS, NUM_ELECTRONS, SPIN)\n"
    )
    result = checker(source, package)
    assert_classification(result, 5, "errors")
    assert_classification(result, 6, "checked")


@pytest.mark.parametrize("pattern", ["{'callback': hf}", "[*hf]", "{**hf}"])
def test_match_capture_shadowing_prevents_false_error_and_false_checked(checker, package, pattern):
    source = (
        "from cudaq_algorithms.stateprep import hartree_fock as hf\n"
        "match payload:\n"
        f"    case {pattern}:\n"
        "        hf()\n"
        "        hf(q, n)\n"
    )
    result = checker(source, package)
    assert_classification(result, 4, "unknown")
    assert_classification(result, 5, "unknown")


def test_lambda_comprehension_target_does_not_resolve_to_imported_api(checker, package):
    source = (
        "from cudaq_algorithms.stateprep import hartree_fock as hf\n"
        "callbacks = lambda items: [hf() for hf in items]\n"
        "hf(q, n)\n"
    )
    result = checker(source, package)
    assert_classification(result, 2, "unknown")
    assert_classification(result, 3, "checked")


@pytest.mark.parametrize("parameters", ["value=hf()", "*, value=hf()"])
def test_lambda_default_calls_are_checked_in_enclosing_scope(checker, package, parameters):
    source = (
        "from cudaq_algorithms.stateprep import hartree_fock as hf\n"
        f"callback = lambda {parameters}: value\n"
        "hf(q, n)\n"
    )
    result = checker(source, package)
    assert_classification(result, 2, "errors")
    assert_classification(result, 3, "checked")


@pytest.mark.parametrize("parameters", ["hf=hf()", "*, hf=hf()"])
def test_lambda_default_precedes_parameter_shadowing(checker, package, parameters):
    source = (
        "from cudaq_algorithms.stateprep import hartree_fock as hf\n"
        "callback = (\n"
        f"    lambda {parameters}:\n"
        "    hf()\n"
        ")\n"
        "hf(q, n)\n"
    )
    result = checker(source, package)
    assert_classification(result, 3, "errors")
    assert_classification(result, 4, "unknown")
    assert_classification(result, 6, "checked")


def test_class_decorator_base_and_keyword_calls_use_enclosing_scope(checker, package):
    source = (
        "from cudaq_algorithms.stateprep import hartree_fock as hf\n"
        "@hf()\n"
        "class Example(\n"
        "    hf(),\n"
        "    metaclass=hf(),\n"
        "):\n"
        "    pass\n"
    )
    result = checker(source, package)
    for line in (2, 4, 5):
        assert_classification(result, line, "errors")


@pytest.mark.parametrize(
    "calls,status,counts",
    [
        ("hf(q, n)", 0, {"checked": 1, "errors": 0, "unknown": 0}),
        ("hf(q, n, spin)", 1, {"checked": 0, "errors": 1, "unknown": 0}),
        ("hf(*args)", 2, {"checked": 0, "errors": 0, "unknown": 1}),
        ("", 2, {"checked": 0, "errors": 0, "unknown": 0}),
        ("hf(q, n, spin)\nhf(*args)", 1, {"checked": 0, "errors": 1, "unknown": 1}),
        ("callback = lambda value=hf(): value\nhf(q, n)", 1, {"checked": 1, "errors": 1, "unknown": 0}),
        ("callback = lambda *, value=hf(): value\nhf(q, n)", 1, {"checked": 1, "errors": 1, "unknown": 0}),
    ],
)
def test_cli_distinguishes_errors_unknown_and_no_checked_calls(package, tmp_path, calls, status, counts):
    snippet = tmp_path / "submitted.py"
    snippet.write_text("from cudaq_algorithms.stateprep import hartree_fock as hf\n" + calls)
    process = subprocess.run(
        [sys.executable, "-B", str(SCRIPT),
         "--package-root", str(package), str(snippet)],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    result = json.loads(process.stdout)
    assert {key: len(result[key]) for key in counts} == counts, result
    assert process.returncode == status, (process.stdout, process.stderr)
    assert not process.stderr


@pytest.mark.parametrize(
    "calls,status,counts",
    [
        ("hf(q, n)", 0, {"checked": 1, "errors": 0, "unknown": 0}),
        ("hf(q, n, spin)", 1, {"checked": 0, "errors": 1, "unknown": 0}),
        ("hf(q)\nhf(q, n, spin)\nhf(*args)\nexternal_call()", 1,
         {"checked": 0, "errors": 2, "unknown": 2}),
        ("hf(*args)", 2, {"checked": 0, "errors": 0, "unknown": 1}),
        ("", 2, {"checked": 0, "errors": 0, "unknown": 0}),
        ("hf(*args)\nhf(**kwargs)", 2, {"checked": 0, "errors": 0, "unknown": 2}),
    ],
)
def test_cli_summary_keeps_errors_evidence_unknown_counts_and_status(
    checker, package, tmp_path, calls, status, counts
):
    snippet = tmp_path / "submitted.py"
    code = "from cudaq_algorithms.stateprep import hartree_fock as hf\n" + calls
    snippet.write_text(code)
    command = [sys.executable, "-B", str(SCRIPT), "--package-root", str(package)]
    full_process = subprocess.run(
        command + [str(snippet)], capture_output=True, text=True, check=False, timeout=10
    )
    full = json.loads(full_process.stdout)
    assert full_process.stdout == json.dumps(checker(code, package), indent=2) + "\n"
    assert set(full) == {"checked", "errors", "unknown"}
    assert {key: len(value) for key, value in full.items()} == counts
    process = subprocess.run(
        command + ["--summary", str(snippet)],
        capture_output=True, text=True, check=False, timeout=10,
    )
    assert process.stdout, process.stderr
    summary = json.loads(process.stdout)
    assert summary["counts"] == counts
    assert summary["errors"] == full["errors"]
    for error in summary["errors"]:
        assert error["evidence"]["path"]
        assert error["evidence"]["line"]
        assert error["evidence"]["declaration"]
    expected_reasons = {}
    for item in full["unknown"]:
        reason = item["reason"]
        expected_reasons[reason] = expected_reasons.get(reason, 0) + 1
    assert summary["unknown_reasons"] == expected_reasons
    assert sum(summary["unknown_reasons"].values()) == counts["unknown"]
    assert "--summary" in summary["note"]
    assert "unverified" in summary["note"]
    assert process.returncode == full_process.returncode == status
    assert not process.stderr and not full_process.stderr


def test_source_and_snippet_are_never_imported_or_executed(checker, tmp_path, monkeypatch):
    name = "static_probe_sentinel_package"
    root = tmp_path / name
    root.mkdir()
    imported_marker = tmp_path / "package-was-executed"
    snippet_marker = tmp_path / "snippet-was-executed"
    called_marker = tmp_path / "function-was-executed"
    (root / "__init__.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(imported_marker)!r}).write_text('unsafe import')\n"
        "def entry(x, *, required):\n"
        f"    Path({str(called_marker)!r}).write_text('unsafe call')\n"
    )
    source = (
        "from pathlib import Path\n"
        f"Path({str(snippet_marker)!r}).write_text('unsafe execution')\n"
        f"from {name} import entry\n"
        "entry(1, required=2)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    assert name not in sys.modules
    result = checker(source, root, package_name=name)
    assert_classification(result, 4, "checked")
    assert not imported_marker.exists()
    assert not snippet_marker.exists()
    assert not called_marker.exists()
    assert not any(key == name or key.startswith(name + ".") for key in sys.modules)


def test_repository_hf_and_uccsd_source_catches_observed_arity_error(checker):
    root = ROOT / "python/cudaq_algorithms"
    assert root.is_dir(), "Repository Python source is required for this contract test"
    source = (
        "from cudaq_algorithms import stateprep\n"
        "stateprep.hartree_fock(q, n)\n"
        "stateprep.hartree_fock(q, n, spin)\n"
        "stateprep.uccsd(q, theta, n, spin)\n"
        "stateprep.uccsd(q, theta, n)\n"
    )
    result = checker(source, root)
    assert_classification(result, 2, "checked")
    assert_classification(result, 3, "errors")
    assert_classification(result, 4, "checked")
    assert_classification(result, 5, "errors")
