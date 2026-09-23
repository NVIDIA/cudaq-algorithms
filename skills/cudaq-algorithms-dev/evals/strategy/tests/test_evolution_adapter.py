# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E08 explicit binding controls using only evaluator-owned fixtures."""
from collections import Counter, UserDict, defaultdict
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest


class EqualToRealLinear:

    def __eq__(self, other):
        return other == "real_linear"


class RaisingEquality:

    def __eq__(self, other):
        raise RuntimeError("schema validation invoked user equality")


def load_adapter(monkeypatch):
    checks = Path(__file__).resolve().parents[1] / "checks"
    path = checks / "evolution_adapter.py"
    assert path.is_file(), "explicit E08 adapter is not implemented"
    monkeypatch.syspath_prepend(str(checks))
    spec = importlib.util.spec_from_file_location("evolution_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def adapter(monkeypatch):
    return load_adapter(monkeypatch)


def callable_stage(attribute, args, kwargs=None, module="e08_binding_fixture"):
    return {
        "kind": "callable",
        "module": module,
        "attribute": attribute,
        "args": args,
        "kwargs": {} if kwargs is None else kwargs
    }


def method_stage(attribute, args, kwargs=None):
    return {
        "kind": "method",
        "attribute": attribute,
        "args": args,
        "kwargs": {} if kwargs is None else kwargs
    }


def declaration(**changes):
    value = {
        "schema_version": 1,
        "build": callable_stage("build", ["terms", "time", "state_prep"]),
        "recover": callable_stage("recover", ["cos_state", "sin_state"]),
        "validate": callable_stage("validate",
                                   ["actual_state", "initial_state"]),
        "kernels": {
            "kind": "mapping",
            "cosine": "cosine",
            "sine": "sine"
        },
        "domain": {
            "hamiltonian": "complex",
            "initial_state": "complex"
        },
        "phase_policy": "exact",
        "recovery_semantics": "real_linear",
        "validation_mode": "raises",
    }
    value.update(changes)
    return value


def write_module(directory, name, source):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.py"
    path.write_text(source)
    importlib.invalidate_caches()
    return path


def import_fixture(monkeypatch, directory, name):
    monkeypatch.syspath_prepend(str(directory))
    monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(name)


@pytest.fixture
def artifact(monkeypatch, tmp_path):
    root = tmp_path / "project"
    write_module(
        root, "e08_binding_fixture", '''
seen = {}
build_result = object()
recover_result = object()
validate_result = object()

def build(terms, time, prep, *, initial=None):
    seen["build"] = (terms, time, prep, initial)
    return build_result

def recover(cosine, sine, *, phases=None):
    seen["recover"] = (cosine, sine, phases)
    return recover_result

def validate(actual, initial, *, tolerance=None):
    seen["validate"] = (actual, initial, tolerance)
    return validate_result

def raises_system_exit(terms, time, prep):
    raise SystemExit("artifact exit")

def exploding(terms, time, prep):
    raise AssertionError("artifact was invoked")

not_callable = 17

class Bundle:
    cosine_kernel = object()
    sine_kernel = object()

    def recover_method(self, cosine, *, sine, original_bundle):
        seen["method_recover"] = (self, cosine, sine, original_bundle)
        return recover_result

    def validate_method(self, actual, initial):
        seen["method_validate"] = (self, actual, initial)
        return validate_result

    def exploding_method(self, cosine, sine):
        raise AssertionError("artifact method was invoked")
''')
    return root, import_fixture(monkeypatch, root, "e08_binding_fixture")


@pytest.fixture
def data(artifact):
    _, module = artifact
    return {
        role: object()
        for role in ("terms", "time", "state_prep", "initial_state", "degree",
                     "tolerance", "cos_state", "sin_state", "cos_phases",
                     "sin_phases", "actual_state")
    } | {
        "bundle": module.Bundle()
    }


def test_positional_keyword_and_method_mappings_preserve_identity(
        adapter, artifact, data):
    root, module = artifact
    binding = declaration(
        build=callable_stage("build", ["terms", "time", "state_prep"],
                             {"initial": "initial_state"}),
        recover=method_stage("recover_method", ["cos_state"], {
            "sine": "sin_state",
            "original_bundle": "bundle"
        }),
        validate=callable_stage("validate", ["actual_state", "initial_state"],
                                {"tolerance": "tolerance"}),
    )
    bound = adapter.bind(binding, artifact_root=root)

    assert bound.build(data) is module.build_result
    assert bound.recover(data) is module.recover_result
    assert bound.validate(data) is module.validate_result
    assert module.seen["build"] == (data["terms"], data["time"],
                                    data["state_prep"], data["initial_state"])
    assert module.seen["method_recover"] == (data["bundle"], data["cos_state"],
                                             data["sin_state"], data["bundle"])
    assert module.seen["validate"] == (data["actual_state"],
                                       data["initial_state"],
                                       data["tolerance"])


def test_artifact_exception_identity_is_preserved(adapter, artifact, data):
    root, _ = artifact
    bound = adapter.bind(declaration(build=callable_stage(
        "raises_system_exit", ["terms", "time", "state_prep"])),
                         artifact_root=root)
    with pytest.raises(SystemExit) as caught:
        bound.build(data)
    assert caught.value.code == "artifact exit"


def test_preflight_resolves_each_stage_without_invoking_it(
        adapter, artifact, data):
    root, module = artifact
    binding = declaration(
        build=callable_stage("exploding", ["terms", "time", "state_prep"]),
        recover=method_stage("exploding_method", ["cos_state", "sin_state"]),
        validate=method_stage("validate_method",
                              ["actual_state", "initial_state"]),
    )
    bound = adapter.bind(binding, artifact_root=root)

    for stage in ("build", "recover", "validate"):
        assert bound.preflight(stage, data) is None
    assert module.seen == {}


def test_binding_detaches_all_declarations_and_exposes_read_only_configuration(
        adapter, artifact, data):
    root, module = artifact
    binding = declaration()
    bound = adapter.bind(binding, artifact_root=root)
    binding["build"]["attribute"] = "raises_system_exit"
    binding["build"]["args"].reverse()
    binding["recover"]["kwargs"]["bad"] = "bundle"
    binding["kernels"]["cosine"] = "wrong"
    binding["domain"]["hamiltonian"] = "real"
    binding["phase_policy"] = "ray"

    assert bound.build(data) is module.build_result
    assert bound.kernels({"cosine": 1, "sine": 2}) == (1, 2)
    assert dict(bound.domain) == {
        "hamiltonian": "complex",
        "initial_state": "complex"
    }
    assert bound.phase_policy == "exact"
    assert bound.recovery_semantics == "real_linear"
    assert bound.validation_mode == "raises"
    with pytest.raises(TypeError):
        bound.domain["hamiltonian"] = "real"
    with pytest.raises(AttributeError):
        bound.phase_policy = "ray"


@pytest.mark.parametrize("kind,bundle", [
    ("mapping", UserDict({
        "cosine": object(),
        "sine": object()
    })),
    ("sequence", [object(), object()]),
    ("sequence", (object(), object())),
])
def test_mapping_and_sequence_kernel_selectors_preserve_objects(
        adapter, kind, bundle):
    selectors = ({
        "kind": "mapping",
        "cosine": "cosine",
        "sine": "sine"
    } if kind == "mapping" else {
        "kind": "sequence",
        "cosine": 1,
        "sine": 0
    })
    bound = adapter.bind(declaration(kernels=selectors))

    cosine, sine = bound.kernels(bundle)
    if kind == "mapping":
        assert cosine is bundle["cosine"] and sine is bundle["sine"]
    else:
        assert cosine is bundle[1] and sine is bundle[0]


def test_attribute_kernel_selectors_do_not_invoke_selected_methods(
        adapter, artifact):
    _, module = artifact
    bundle = module.Bundle()
    bound = adapter.bind(
        declaration(
            kernels={
                "kind": "attributes",
                "cosine": "validate_method",
                "sine": "recover_method"
            }))

    cosine, sine = bound.kernels(bundle)
    assert cosine.__self__ is bundle and cosine.__func__ is module.Bundle.validate_method
    assert sine.__self__ is bundle and sine.__func__ is module.Bundle.recover_method


@pytest.mark.parametrize("kernels,bundle", [
    ({
        "kind": "mapping",
        "cosine": "cosine",
        "sine": "sine"
    }, {}),
    ({
        "kind": "mapping",
        "cosine": "cosine",
        "sine": "sine"
    }, []),
    ({
        "kind": "attributes",
        "cosine": "cosine",
        "sine": "sine"
    }, object()),
    ({
        "kind": "sequence",
        "cosine": 0,
        "sine": 1
    }, "ab"),
    ({
        "kind": "sequence",
        "cosine": 0,
        "sine": 2
    }, [object()]),
])
def test_missing_or_unsupported_kernel_results_are_unknown(
        adapter, kernels, bundle):
    bound = adapter.bind(declaration(kernels=kernels))
    with pytest.raises(adapter.AdapterUnavailable):
        bound.kernels(bundle)


def test_mapping_selector_does_not_synthesize_missing_counter_keys(adapter):
    bundle = Counter({"cosine": 3})
    original = bundle.copy()
    bound = adapter.bind(declaration())

    with pytest.raises(adapter.AdapterUnavailable):
        bound.kernels(bundle)
    assert bundle == original


def test_mapping_selector_does_not_mutate_defaultdict_for_missing_keys(
        adapter):
    bundle = defaultdict(list, {"cosine": [object()]})
    original = dict(bundle)
    bound = adapter.bind(declaration())

    with pytest.raises(adapter.AdapterUnavailable):
        bound.kernels(bundle)
    assert dict(bundle) == original


@pytest.mark.parametrize("change", [
    {
        "schema_version": True
    },
    {
        "schema_version": 2
    },
    {
        "build": {
            "kind": [],
            "attribute": "build",
            "args": [],
            "kwargs": {}
        }
    },
    {
        "domain": {
            "hamiltonian": "complex"
        }
    },
    {
        "domain": {
            "hamiltonian": "guess",
            "initial_state": "complex"
        }
    },
    {
        "domain": {
            "hamiltonian": [],
            "initial_state": "complex"
        }
    },
    {
        "domain": {
            "hamiltonian": "real",
            "initial_state": "complex",
            "extra": "real"
        }
    },
    {
        "phase_policy": "infer"
    },
    {
        "phase_policy": []
    },
    {
        "recovery_semantics": "complex_linear"
    },
    {
        "recovery_semantics": []
    },
    {
        "validation_mode": "verdict"
    },
    {
        "validation_mode": []
    },
    {
        "extra": True
    },
    {
        "build": method_stage("build", ["terms", "time", "state_prep"])
    },
    {
        "recover":
        callable_stage("recover", ["cos_state", "sin_state"],
                       module="../bad.py")
    },
    {
        "validate": callable_stage("_private",
                                   ["actual_state", "initial_state"])
    },
    {
        "build": callable_stage("build", ["terms", "time"])
    },
    {
        "build": callable_stage("build",
                                ["terms", "time", "state_prep", "terms"])
    },
    {
        "build":
        callable_stage("build", ["terms", "time", "state_prep", "cos_state"])
    },
    {
        "build": callable_stage("build", ["terms", "time", "state_prep", []])
    },
    {
        "recover":
        callable_stage("recover", ["cos_state", "sin_state"],
                       {"x": "sin_state"})
    },
    {
        "validate":
        callable_stage("validate", ["actual_state", "initial_state"],
                       {"x.y": "tolerance"})
    },
    {
        "validate":
        callable_stage("validate", ["actual_state", "initial_state"],
                       {"tolerance": "abs(tolerance)"})
    },
    {
        "kernels": {
            "kind": "mapping",
            "cosine": "_private",
            "sine": "sine"
        }
    },
    {
        "kernels": {
            "kind": "attributes",
            "cosine": "nested.value",
            "sine": "sine"
        }
    },
    {
        "kernels": {
            "kind": "sequence",
            "cosine": True,
            "sine": 1
        }
    },
    {
        "kernels": {
            "kind": "sequence",
            "cosine": -1,
            "sine": 0
        }
    },
    {
        "kernels": {
            "kind": [],
            "cosine": "cosine",
            "sine": "sine"
        }
    },
    {
        "kernels": {
            "kind": "invent",
            "cosine": "cosine",
            "sine": "sine"
        }
    },
])
def test_invalid_declarations_are_rejected_before_import(
        adapter, monkeypatch, change):
    imported = []
    monkeypatch.setattr(importlib, "import_module",
                        lambda name: imported.append(name))
    with pytest.raises(ValueError):
        adapter.bind(declaration(**change))
    assert imported == []


@pytest.mark.parametrize("value", [None, [], {}, {"schema_version": 1}])
def test_incomplete_top_level_declarations_are_rejected(adapter, value):
    with pytest.raises(ValueError):
        adapter.bind(value)


@pytest.mark.parametrize("spoof", [EqualToRealLinear(), RaisingEquality()])
def test_recovery_semantics_rejects_non_strings_without_equality_dispatch(
        adapter, spoof):
    with pytest.raises(ValueError):
        adapter.bind(declaration(recovery_semantics=spoof))


@pytest.mark.parametrize("stage", ["build", "recover", "validate"])
def test_missing_context_is_unknown(adapter, artifact, data, stage):
    root, _ = artifact
    missing = dict(data)
    required = {
        "build": "terms",
        "recover": "cos_state",
        "validate": "actual_state"
    }
    del missing[required[stage]]
    bound = adapter.bind(declaration(), artifact_root=root)
    with pytest.raises(adapter.AdapterUnavailable):
        getattr(bound, stage)(missing)


def test_method_receiver_requires_bundle_independent_of_argument_roles(
        adapter, artifact, data):
    root, _ = artifact
    bound = adapter.bind(
        declaration(recover=method_stage("recover_method", ["cos_state"], {
            "sine": "sin_state",
            "original_bundle": "bundle"
        })),
        artifact_root=root)
    missing = dict(data)
    del missing["bundle"]
    with pytest.raises(adapter.AdapterUnavailable):
        bound.preflight("recover", missing)


@pytest.mark.parametrize("stage,entry", [
    ("build", callable_stage("missing", ["terms", "time", "state_prep"])),
    ("recover", callable_stage("not_callable", ["cos_state", "sin_state"])),
    ("validate", callable_stage("build", ["actual_state", "initial_state"])),
])
def test_missing_noncallable_or_unbindable_entrypoints_are_unknown(
        adapter, artifact, data, stage, entry):
    root, _ = artifact
    bound = adapter.bind(declaration(**{stage: entry}), artifact_root=root)
    with pytest.raises(adapter.AdapterUnavailable):
        bound.preflight(stage, data)


def test_unknown_preflight_stage_is_unavailable(adapter, data):
    bound = adapter.bind(declaration())
    with pytest.raises(adapter.AdapterUnavailable):
        bound.preflight("infer", data)


def test_outside_declared_module_is_unknown(adapter, artifact, tmp_path, data):
    _, module = artifact
    other = tmp_path / "other"
    other.mkdir()
    bound = adapter.bind(declaration(), artifact_root=other)
    with pytest.raises(adapter.AdapterUnavailable, match="origin"):
        bound.preflight("build", data)
    assert module.seen == {}


def test_in_root_module_aliasing_outside_callable_is_unknown(
        adapter, artifact, monkeypatch, tmp_path, data):
    root, _ = artifact
    outside = tmp_path / "outside"
    write_module(
        outside, "e08_outside_callable", '''
def build(terms, time, prep):
    raise AssertionError("outside callable was invoked")
''')
    monkeypatch.syspath_prepend(str(outside))
    write_module(root, "e08_alias_api",
                 "from e08_outside_callable import build\n")
    import_fixture(monkeypatch, root, "e08_alias_api")
    binding = declaration(build=callable_stage(
        "build", ["terms", "time", "state_prep"], module="e08_alias_api"))
    bound = adapter.bind(binding, artifact_root=root)
    with pytest.raises(adapter.AdapterUnavailable, match="origin"):
        bound.preflight("build", data)


def test_bound_method_defined_outside_root_is_unknown(adapter, monkeypatch,
                                                      tmp_path, data):
    root = tmp_path / "project"
    root.mkdir(exist_ok=True)
    outside = tmp_path / "outside"
    write_module(
        outside, "e08_outside_method", '''
class Bundle:
    def recover_method(self, cosine, sine):
        raise AssertionError("outside method was invoked")
''')
    module = import_fixture(monkeypatch, outside, "e08_outside_method")
    data["bundle"] = module.Bundle()
    bound = adapter.bind(declaration(
        recover=method_stage("recover_method", ["cos_state", "sin_state"])),
                         artifact_root=root)
    with pytest.raises(adapter.AdapterUnavailable, match="origin"):
        bound.preflight("recover", data)


def test_module_symlink_escape_is_unknown(adapter, monkeypatch, tmp_path,
                                          data):
    outside = write_module(
        tmp_path, "e08_symlink_target", '''
def build(terms, time, prep):
    raise AssertionError("symlink target was invoked")
''')
    root = tmp_path / "project"
    root.mkdir(exist_ok=True)
    (root / "e08_link_api.py").symlink_to(outside)
    import_fixture(monkeypatch, root, "e08_link_api")
    binding = declaration(build=callable_stage(
        "build", ["terms", "time", "state_prep"], module="e08_link_api"))
    bound = adapter.bind(binding, artifact_root=root)
    with pytest.raises(adapter.AdapterUnavailable, match="origin"):
        bound.preflight("build", data)
