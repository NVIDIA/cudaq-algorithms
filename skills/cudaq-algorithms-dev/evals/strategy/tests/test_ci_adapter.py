# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E05 binding controls; only evaluator-owned harmless modules execute here."""
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest


def load_adapter(monkeypatch):
    checks = Path(__file__).resolve().parents[1] / "checks"
    path = checks / "ci_adapter.py"
    assert path.is_file(), "explicit E05 adapter is not implemented"
    monkeypatch.syspath_prepend(str(checks))
    spec = importlib.util.spec_from_file_location("ci_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def adapter(monkeypatch):
    return load_adapter(monkeypatch)


def declaration(**changes):
    return {
        "schema_version": 1,
        "module": "e05_binding_fixture",
        "attribute": "prepare",
        "args": ["orbital_basis", "occupations", "coefficients"],
        "kwargs": {},
        "representation": "basis_occupations",
        "coefficient_policy": "reject",
        **changes
    }


def write_module(directory, name, source):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.py"
    path.write_text(source)
    importlib.invalidate_caches()
    return path


def load_fixture(monkeypatch, directory, name):
    monkeypatch.syspath_prepend(str(directory))
    monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(name)


@pytest.fixture
def artifact(monkeypatch, tmp_path):
    root = tmp_path / "project"
    write_module(
        root, "e05_binding_fixture", """
result = object()
seen = None
not_callable = 17

def prepare(basis, occupations, weights):
    global seen
    seen = (basis, occupations, weights)
    return result

def prepare_matrices(matrices, weights):
    global seen
    seen = (matrices, weights)
    return result

def raises(basis, occupations, weights):
    raise ZeroDivisionError("artifact failure")

class PublicFactories:
    @staticmethod
    def prepare(*, basis, determinants, weights):
        return globals()["prepare"](basis, determinants, weights)

class Constructor:
    def __init__(self, basis, occupations, weights):
        raise AssertionError("constructor was invoked")
""")
    return root, load_fixture(monkeypatch, root, "e05_binding_fixture")


@pytest.fixture
def case():
    return {
        role: object()
        for role in ("orbital_basis", "occupations", "orbital_matrices",
                     "coefficients")
    }


@pytest.mark.parametrize("representation,attribute,args,kwargs,roles", [
    ("basis_occupations", "prepare", [
        "orbital_basis", "occupations", "coefficients"
    ], {}, ["orbital_basis", "occupations", "coefficients"]),
    ("basis_occupations", "PublicFactories.prepare", [], {
        "weights": "coefficients",
        "determinants": "occupations",
        "basis": "orbital_basis"
    }, ["orbital_basis", "occupations", "coefficients"]),
    ("occupied_matrices", "prepare_matrices", ["orbital_matrices"], {
        "weights": "coefficients"
    }, ["orbital_matrices", "coefficients"]),
    ("occupied_matrices", "prepare_matrices", [], {
        "weights": "coefficients",
        "matrices": "orbital_matrices"
    }, ["orbital_matrices", "coefficients"]),
])
def test_declared_roles_reach_artifact_unchanged(monkeypatch, artifact, case,
                                                 representation, attribute,
                                                 args, kwargs, roles):
    adapter = load_adapter(monkeypatch)
    root, module = artifact
    invoke = adapter.bind(declaration(representation=representation,
                                      attribute=attribute,
                                      args=args,
                                      kwargs=kwargs),
                          artifact_root=root)

    assert invoke(case) is module.result
    assert all(actual is case[role]
               for actual, role in zip(module.seen, roles))
    assert len(module.seen) == len(roles)


@pytest.mark.parametrize("policy", ["normalize", "reject"])
def test_coefficient_policy_does_not_repair_inputs(adapter, artifact, case,
                                                   policy):
    root, module = artifact
    case["coefficients"] = [2.0, 3.0j, -4.0, 5.0]
    invoke = adapter.bind(declaration(coefficient_policy=policy),
                          artifact_root=root)

    assert invoke(case) is module.result
    assert module.seen[2] is case["coefficients"]
    assert case["coefficients"] == [2.0, 3.0j, -4.0, 5.0]


@pytest.mark.parametrize("change", [
    {
        "schema_version": True
    },
    {
        "schema_version": 2
    },
    {
        "representation": "infer"
    },
    {
        "representation": []
    },
    {
        "coefficient_policy": "repair"
    },
    {
        "coefficient_policy": []
    },
    {
        "module": "../artifact.py"
    },
    {
        "module": "api._private"
    },
    {
        "attribute": "_private"
    },
    {
        "attribute": "PublicFactories._private"
    },
    {
        "args": None
    },
    {
        "args": ["orbital_basis", "coefficients"]
    },
    {
        "args": ["orbital_basis", "occupations", "occupations"]
    },
    {
        "args": ["orbital_basis", "occupations", "coefficients", "num_system"]
    },
    {
        "kwargs": {
            "weights": "coefficients"
        }
    },
    {
        "kwargs": {
            "weights": "abs(coefficients)"
        }
    },
    {
        "args": [],
        "kwargs": {
            "private.weights": "coefficients"
        }
    },
    {
        "args": ["orbital_basis", "occupations", []]
    },
    {
        "result_accessor": "kernel"
    },
    {
        "kind": "method"
    },
])
def test_invalid_declaration_rejected_before_import(adapter, change):
    with pytest.raises(ValueError):
        adapter.bind(
            declaration(module="e05_module_that_does_not_exist", **change
                        ) if "module" not in change else declaration(**change))


@pytest.mark.parametrize("bad", [None, [], {}, {"schema_version": 1}])
def test_incomplete_declaration_is_rejected(adapter, bad):
    with pytest.raises(ValueError):
        adapter.bind(bad)


@pytest.mark.parametrize(
    "attribute", ["missing", "not_callable", "PublicFactories.prepare"])
def test_missing_noncallable_or_wrong_signature_is_unknown(
        adapter, artifact, case, attribute):
    root, module = artifact
    invoke = adapter.bind(declaration(attribute=attribute), artifact_root=root)
    with pytest.raises(adapter.AdapterUnavailable):
        invoke.preflight(case)
    assert module.seen is None


def test_missing_module_is_unknown(adapter, case):
    invoke = adapter.bind(declaration(module="e05_module_that_does_not_exist"))
    with pytest.raises(adapter.AdapterUnavailable):
        invoke.preflight(case)


@pytest.mark.parametrize("bad_case",
                         [None, [], {}, {
                             "coefficients": [1, 0, 0, 0]
                         }])
def test_missing_case_roles_are_unknown(adapter, artifact, bad_case):
    root, module = artifact
    invoke = adapter.bind(declaration(), artifact_root=root)
    with pytest.raises(adapter.AdapterUnavailable):
        invoke(bad_case)
    assert module.seen is None


@pytest.mark.parametrize("attribute", ["raises", "Constructor"])
def test_preflight_does_not_execute_target(adapter, artifact, case, attribute):
    root, _ = artifact
    adapter.bind(declaration(attribute=attribute),
                 artifact_root=root).preflight(case)


def test_artifact_exception_is_preserved(adapter, artifact, case):
    root, _ = artifact
    invoke = adapter.bind(declaration(attribute="raises"), artifact_root=root)
    with pytest.raises(ZeroDivisionError, match="artifact failure"):
        invoke(case)


def test_binding_detaches_callable_argument_and_keyword_selection(
        adapter, artifact, case):
    root, module = artifact
    binding = declaration(args=["orbital_basis"],
                          kwargs={
                              "occupations": "occupations",
                              "weights": "coefficients"
                          })
    invoke = adapter.bind(binding, artifact_root=root)
    binding["attribute"] = "raises"
    binding["args"][0] = "coefficients"
    binding["kwargs"]["weights"] = "orbital_basis"
    binding["representation"] = "occupied_matrices"

    assert invoke(case) is module.result
    assert module.seen == (case["orbital_basis"], case["occupations"],
                           case["coefficients"])


def test_in_root_reexport_is_allowed(adapter, artifact, monkeypatch, case):
    root, module = artifact
    write_module(root, "e05_reexport_api",
                 "from e05_binding_fixture import prepare\n")
    load_fixture(monkeypatch, root, "e05_reexport_api")

    invoke = adapter.bind(declaration(module="e05_reexport_api"),
                          artifact_root=root)
    assert invoke(case) is module.result


def test_outside_declared_module_is_unknown(adapter, artifact, tmp_path, case):
    _, module = artifact
    other = tmp_path / "other"
    other.mkdir()
    invoke = adapter.bind(declaration(), artifact_root=other)
    with pytest.raises(adapter.AdapterUnavailable, match="origin"):
        invoke(case)
    assert module.seen is None


def test_in_root_alias_to_outside_target_is_unknown(adapter, artifact,
                                                    monkeypatch, tmp_path,
                                                    case):
    _, module = artifact
    other = tmp_path / "other"
    write_module(other, "e05_outside_alias",
                 "from e05_binding_fixture import prepare\n")
    load_fixture(monkeypatch, other, "e05_outside_alias")
    invoke = adapter.bind(declaration(module="e05_outside_alias"),
                          artifact_root=other)
    with pytest.raises(adapter.AdapterUnavailable, match="origin"):
        invoke.preflight(case)
    assert module.seen is None


def test_module_symlink_escape_is_unknown(adapter, monkeypatch, tmp_path,
                                          case):
    outside = write_module(tmp_path, "e05_outside",
                           "def prepare(*args):\n    return args\n")
    root = tmp_path / "project"
    root.mkdir()
    (root / "e05_link_api.py").symlink_to(outside)
    load_fixture(monkeypatch, root, "e05_link_api")
    invoke = adapter.bind(declaration(module="e05_link_api"),
                          artifact_root=root)
    with pytest.raises(adapter.AdapterUnavailable, match="origin"):
        invoke.preflight(case)


def test_builtin_without_source_provenance_is_unknown(adapter, monkeypatch,
                                                      tmp_path, case):
    root = tmp_path / "project"
    write_module(root, "e05_builtin_api", "prepare = print\n")
    load_fixture(monkeypatch, root, "e05_builtin_api")
    invoke = adapter.bind(declaration(module="e05_builtin_api"),
                          artifact_root=root)
    with pytest.raises(adapter.AdapterUnavailable, match="origin"):
        invoke.preflight(case)


def test_missing_artifact_root_is_unknown(adapter, tmp_path):
    with pytest.raises(adapter.AdapterUnavailable, match="origin"):
        adapter.bind(declaration(), artifact_root=tmp_path / "missing")


def test_root_symlink_is_resolved_before_invocation(adapter, monkeypatch,
                                                    tmp_path, case):
    first, second = tmp_path / "first", tmp_path / "second"
    source = "def prepare(*args):\n    return args\n"
    write_module(first, "e05_retargeted", source)
    write_module(second, "e05_retargeted", source)
    link = tmp_path / "project"
    link.symlink_to(first, target_is_directory=True)
    monkeypatch.syspath_prepend(str(link))
    monkeypatch.delitem(sys.modules, "e05_retargeted", raising=False)
    invoke = adapter.bind(declaration(module="e05_retargeted"),
                          artifact_root=link)
    link.unlink()
    link.symlink_to(second, target_is_directory=True)
    importlib.invalidate_caches()

    with pytest.raises(adapter.AdapterUnavailable, match="origin"):
        invoke.preflight(case)
