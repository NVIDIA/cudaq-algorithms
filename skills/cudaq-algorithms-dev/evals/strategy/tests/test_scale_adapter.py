# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Binding tests: wrong selection, implicit repair, and malformed declarations."""
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest


def load_adapter():
    path = Path(__file__).resolve().parents[1] / "checks/scale_adapter.py"
    assert path.is_file(), "explicit E02 adapter is not implemented"
    spec = importlib.util.spec_from_file_location("scale_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_binding(**changes):
    return {
        "schema_version": 1,
        "kind": "callable",
        "module": __name__,
        "attribute": "scale_function",
        "args": ["encoding", "factor"],
        "kwargs": {},
        "zero_policy": "encode",
        **changes
    }


def write_module(directory, name, source):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.py"
    path.write_text(source)
    importlib.invalidate_caches()
    return path


def import_module(monkeypatch, directory, name):
    monkeypatch.syspath_prepend(str(directory))
    monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(name)


def scale_function(encoding, factor):
    return encoding, factor


class ScaleConstructor:

    def __init__(self, *, base, multiplier):
        self.base, self.multiplier = base, multiplier


class MethodEncoding:

    def scaled(self, *, multiplier):
        return self, multiplier


@pytest.mark.parametrize("kind", ["function", "constructor", "method"])
def test_explicit_api_shapes_keep_arguments_and_result(kind):
    adapter = load_adapter()
    encoding, factor = MethodEncoding(), object()
    if kind == "function":
        result = adapter.bind(make_binding())(encoding, factor)
        assert result[0] is encoding and result[1] is factor
    elif kind == "constructor":
        result = adapter.bind(
            make_binding(attribute="ScaleConstructor",
                         args=[],
                         kwargs={
                             "base": "encoding",
                             "multiplier": "factor"
                         }))(encoding, factor)
        assert result.base is encoding and result.multiplier is factor
    else:
        declaration = make_binding(kind="method",
                                   attribute="scaled",
                                   args=[],
                                   kwargs={"multiplier": "factor"})
        del declaration["module"]
        result = adapter.bind(declaration)(encoding, factor)
        assert result[0] is encoding and result[1] is factor


@pytest.mark.parametrize("change", [
    {
        "schema_version": True
    },
    {
        "kind": "guess"
    },
    {
        "attribute": "_private"
    },
    {
        "module": "../worker.py"
    },
    {
        "args": ["encoding", "factor", "factor"]
    },
    {
        "args": ["encoding"]
    },
    {
        "kwargs": {
            "scale": "abs(factor)"
        }
    },
    {
        "repair_sign": True
    },
    {
        "zero_policy": "return_reference"
    },
    {
        "kind": []
    },
    {
        "zero_policy": []
    },
    {
        "args": None
    },
])
def test_invalid_binding_rejected_before_import(change):
    adapter = load_adapter()
    with pytest.raises(ValueError):
        adapter.bind(make_binding(**change))


def test_missing_target_is_unavailable_without_fallback():
    adapter = load_adapter()
    with pytest.raises(adapter.AdapterUnavailable):
        adapter.bind(make_binding(attribute="missing_scale"))(object(), 2)


def raising_scale(encoding, factor):
    raise ZeroDivisionError("artifact failure")


def test_artifact_exceptions_are_not_repaired():
    adapter = load_adapter()
    with pytest.raises(ZeroDivisionError, match="artifact failure"):
        adapter.bind(make_binding(attribute="raising_scale"))(object(), 0)


def test_bound_return_object_is_not_unwrapped_or_replaced():
    adapter = load_adapter()
    sentinel = object()
    # Public module attribute selected explicitly, without a mock importer.
    setattr(sys.modules[__name__], "sentinel_scale",
            lambda encoding, factor: sentinel)
    assert adapter.bind(make_binding(attribute="sentinel_scale"))(
        None, -2) is sentinel


def test_wrong_argument_mapping_is_unknown_not_artifact_rejection():
    adapter = load_adapter()
    declaration = make_binding(attribute="ScaleConstructor")
    with pytest.raises(adapter.AdapterUnavailable):
        adapter.bind(declaration)(object(), 1.)


def test_mutating_declaration_does_not_reselect_callable():
    adapter = load_adapter()
    declaration = make_binding()
    invoke = adapter.bind(declaration)
    declaration["attribute"] = "raising_scale"
    declaration["args"].reverse()
    assert invoke("base", 2.) == ("base", 2.)


def test_preflight_does_not_execute_artifact_and_detects_missing_binding():
    adapter = load_adapter()
    adapter.bind(make_binding(attribute="raising_scale")).preflight(object())
    with pytest.raises(adapter.AdapterUnavailable):
        adapter.bind(make_binding(attribute="missing_scale")).preflight(
            object())


@pytest.mark.parametrize("attribute, args, kwargs", [
    ("scale_function", ["encoding", "factor"], {}),
    ("ScaleConstructor", [], {
        "base": "encoding",
        "multiplier": "factor"
    }),
])
def test_artifact_root_accepts_in_root_reexports(monkeypatch, tmp_path,
                                                 attribute, args, kwargs):
    adapter = load_adapter()
    root = tmp_path / "project"
    suffix = attribute.lower()
    implementation = f"e02_reexport_impl_{suffix}"
    api = f"e02_reexport_api_{suffix}"
    write_module(
        root, implementation, """
def scale_function(encoding, factor):
    return encoding, factor

class ScaleConstructor:
    def __init__(self, *, base, multiplier):
        self.base = base
        self.multiplier = multiplier
""")
    write_module(root, api, f"""
from {implementation} import {attribute}
""")
    import_module(monkeypatch, root, api)
    declaration = make_binding(module=api,
                               attribute=attribute,
                               args=args,
                               kwargs=kwargs)

    result = adapter.bind(declaration, artifact_root=root)("base", 2.)

    if attribute == "scale_function":
        assert result == ("base", 2.)
    else:
        assert (result.base, result.multiplier) == ("base", 2.)


def test_artifact_root_accepts_bound_method_defined_in_root(
        monkeypatch, tmp_path):
    adapter = load_adapter()
    root = tmp_path / "project"
    write_module(
        root, "e02_method_impl", """
class MethodEncoding:
    def scaled(self, *, multiplier):
        return self, multiplier
""")
    module = import_module(monkeypatch, root, "e02_method_impl")
    declaration = make_binding(kind="method",
                               attribute="scaled",
                               args=[],
                               kwargs={"multiplier": "factor"})
    del declaration["module"]
    encoding = module.MethodEncoding()

    result = adapter.bind(declaration, artifact_root=root)(encoding, -.7)

    assert result[0] is encoding and result[1] == -.7


def test_artifact_root_rejects_declared_module_outside_root(
        monkeypatch, tmp_path):
    adapter = load_adapter()
    root = tmp_path / "project"
    root.mkdir()
    write_module(
        tmp_path, "e02_outside_module", """
def scale_function(encoding, factor):
    raise AssertionError("outside operation was invoked")
""")
    import_module(monkeypatch, tmp_path, "e02_outside_module")
    declaration = make_binding(module="e02_outside_module")

    with pytest.raises(adapter.AdapterUnavailable, match="origin"):
        adapter.bind(declaration, artifact_root=root)(object(), 2.)


def test_artifact_root_rejects_imported_outside_alias(monkeypatch, tmp_path):
    adapter = load_adapter()
    root = tmp_path / "project"
    write_module(
        tmp_path, "e02_evaluator_impl", """
def scale_function(encoding, factor):
    raise AssertionError("evaluator operation was invoked")
""")
    write_module(root, "e02_alias_api", """
from e02_evaluator_impl import scale_function
""")
    monkeypatch.syspath_prepend(str(tmp_path))
    import_module(monkeypatch, root, "e02_alias_api")
    declaration = make_binding(module="e02_alias_api")

    with pytest.raises(adapter.AdapterUnavailable, match="origin"):
        adapter.bind(declaration, artifact_root=root).preflight(object())


def test_artifact_root_rejects_bound_method_defined_outside_root(
        monkeypatch, tmp_path):
    adapter = load_adapter()
    root = tmp_path / "project"
    root.mkdir()
    write_module(
        tmp_path, "e02_outside_method", """
class MethodEncoding:
    def scaled(self, factor):
        raise AssertionError("outside method was invoked")
""")
    module = import_module(monkeypatch, tmp_path, "e02_outside_method")
    declaration = make_binding(kind="method",
                               attribute="scaled",
                               args=["factor"],
                               kwargs={})
    del declaration["module"]

    with pytest.raises(adapter.AdapterUnavailable, match="origin"):
        adapter.bind(declaration, artifact_root=root)(module.MethodEncoding(),
                                                      2.)


def test_artifact_root_rejects_symlink_escape(monkeypatch, tmp_path):
    adapter = load_adapter()
    root = tmp_path / "project"
    outside = write_module(
        tmp_path, "e02_symlink_target", """
def scale_function(encoding, factor):
    return encoding, factor
""")
    root.mkdir()
    (root / "e02_symlink_api.py").symlink_to(outside)
    import_module(monkeypatch, root, "e02_symlink_api")
    declaration = make_binding(module="e02_symlink_api")

    with pytest.raises(adapter.AdapterUnavailable, match="origin"):
        adapter.bind(declaration, artifact_root=root).preflight(object())


def test_artifact_root_rejects_unknown_builtin_provenance(
        monkeypatch, tmp_path):
    adapter = load_adapter()
    root = tmp_path / "project"
    write_module(root, "e02_builtin_alias", "scale_function = len\n")
    import_module(monkeypatch, root, "e02_builtin_alias")
    declaration = make_binding(module="e02_builtin_alias",
                               args=["encoding"],
                               kwargs={"factor": "factor"})

    with pytest.raises(adapter.AdapterUnavailable, match="origin"):
        adapter.bind(declaration, artifact_root=root).preflight([])


def test_artifact_root_is_captured_before_resolution(monkeypatch, tmp_path):
    adapter = load_adapter()
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    write_module(
        first_root, "e02_retargeted_root", """
def scale_function(encoding, factor):
    return encoding, factor
""")
    write_module(
        second_root, "e02_retargeted_root", """
def scale_function(encoding, factor):
    raise AssertionError("retargeted operation was invoked")
""")
    root_link = tmp_path / "project"
    root_link.symlink_to(first_root, target_is_directory=True)
    monkeypatch.syspath_prepend(str(root_link))
    invoke = adapter.bind(make_binding(module="e02_retargeted_root"),
                          artifact_root=root_link)
    root_link.unlink()
    root_link.symlink_to(second_root, target_is_directory=True)
    importlib.invalidate_caches()

    with pytest.raises(adapter.AdapterUnavailable, match="origin"):
        invoke.preflight(object())


def test_verify_origin_requires_the_expected_in_root_file(
        monkeypatch, tmp_path):
    adapter = load_adapter()
    root = tmp_path / "project"
    expected = write_module(root, "e02_expected_consumer", """
class Consumer:
    pass
""")
    other = write_module(root, "e02_other_consumer", """
class Consumer:
    pass
""")
    expected_module = import_module(monkeypatch, root, "e02_expected_consumer")
    other_module = import_module(monkeypatch, root, "e02_other_consumer")

    adapter.verify_origin(expected_module.Consumer, root,
                          expected.relative_to(root))
    with pytest.raises(adapter.AdapterUnavailable, match="origin"):
        adapter.verify_origin(other_module.Consumer, root,
                              expected.relative_to(root))


def test_artifact_root_preflight_does_not_execute_constructor(
        monkeypatch, tmp_path):
    adapter = load_adapter()
    root = tmp_path / "project"
    write_module(
        root, "e02_exploding_constructor", """
class ExplodingConstructor:
    def __init__(self, encoding, factor):
        raise AssertionError("constructor was invoked")
""")
    import_module(monkeypatch, root, "e02_exploding_constructor")
    declaration = make_binding(module="e02_exploding_constructor",
                               attribute="ExplodingConstructor")

    adapter.bind(declaration, artifact_root=root).preflight(object())
