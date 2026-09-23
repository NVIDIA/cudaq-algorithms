# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit evaluator-owned E08 binding; use only in the isolated checker.

Artifact imports and calls are untrusted.  This module binds only declared APIs:
it performs no discovery, synthesis, phase construction, result conversion, or
numerical validation.
"""
from collections import Counter
from collections.abc import Mapping
import importlib
import inspect
from types import MappingProxyType

from scale_adapter import (AdapterUnavailable, _public_path, _require_in_root,
                           _root_path)

_STAGE_ROLES = {
    "build": (
        frozenset({"terms", "time", "state_prep"}),
        frozenset({"initial_state", "degree", "tolerance"}),
    ),
    "recover": (
        frozenset({"cos_state", "sin_state"}),
        frozenset({
            "cos_phases", "sin_phases", "bundle", "initial_state", "terms",
            "time", "tolerance"
        }),
    ),
    "validate": (
        frozenset({"actual_state", "initial_state"}),
        frozenset({"terms", "time", "tolerance", "bundle"}),
    ),
}


def _simple_public_identifier(value):
    return _public_path(value) and "." not in value


def _stage_declaration(stage, declaration):
    if not isinstance(declaration, dict):
        raise ValueError(f"{stage} binding must be an object")
    kind = declaration.get("kind")
    fields = {"kind", "attribute", "args", "kwargs"}
    if kind == "callable":
        fields.add("module")
    if (not isinstance(kind, str) or kind not in {"callable", "method"}
            or set(declaration) != fields
            or (stage == "build" and kind != "callable")
            or not _public_path(declaration.get("attribute")) or
        (kind == "callable" and not _public_path(declaration.get("module")))):
        raise ValueError(f"invalid {stage} binding fields")

    args, kwargs = declaration["args"], declaration["kwargs"]
    if (not isinstance(args, list) or not isinstance(kwargs, Mapping)
            or not all(_simple_public_identifier(key) for key in kwargs)):
        raise ValueError(f"invalid {stage} argument mapping")
    roles = list(args) + list(kwargs.values())
    required, optional = _STAGE_ROLES[stage]
    allowed = required | optional
    if not all(isinstance(role, str) and role in allowed for role in roles):
        raise ValueError(f"invalid {stage} semantic-role mapping")
    counts = Counter(roles)
    if (any(counts[role] != 1 for role in required)
            or any(count > 1 for count in counts.values())):
        raise ValueError(f"invalid {stage} semantic-role mapping")

    # Immutable strings plus detached containers prevent later caller mutation.
    return (kind, declaration.get("module"), declaration["attribute"],
            tuple(args), MappingProxyType(dict(kwargs)), tuple(roles))


def _kernel_declaration(declaration):
    fields = {"kind", "cosine", "sine"}
    if not isinstance(declaration, dict) or set(declaration) != fields:
        raise ValueError("invalid kernel binding fields")
    kind = declaration["kind"]
    cosine, sine = declaration["cosine"], declaration["sine"]
    if isinstance(kind, str) and kind in {"mapping", "attributes"}:
        if not (_simple_public_identifier(cosine)
                and _simple_public_identifier(sine)):
            raise ValueError("invalid kernel selector")
    elif isinstance(kind, str) and kind == "sequence":
        if not (type(cosine) is int and cosine >= 0 and type(sine) is int
                and sine >= 0):
            raise ValueError("invalid kernel selector")
    else:
        raise ValueError("invalid kernel binding kind")
    return kind, cosine, sine


def _domain_declaration(declaration):
    fields = {"hamiltonian", "initial_state"}
    if (not isinstance(declaration, dict) or set(declaration) != fields or any(
            not isinstance(value, str) or value not in {"real", "complex"}
            for value in declaration.values())):
        raise ValueError("invalid domain declaration")
    return MappingProxyType(dict(declaration))


class _BoundEvolution:
    __slots__ = ("_stages", "_kernel_binding", "_root", "_domain",
                 "_phase_policy", "_recovery_semantics", "_validation_mode")

    def __init__(self, stages, kernel_binding, root, domain, phase_policy,
                 recovery_semantics, validation_mode):
        self._stages = MappingProxyType(dict(stages))
        self._kernel_binding = kernel_binding
        self._root = root
        self._domain = domain
        self._phase_policy = phase_policy
        self._recovery_semantics = recovery_semantics
        self._validation_mode = validation_mode

    @property
    def domain(self):
        return self._domain

    @property
    def phase_policy(self):
        return self._phase_policy

    @property
    def recovery_semantics(self):
        return self._recovery_semantics

    @property
    def validation_mode(self):
        return self._validation_mode

    def _resolve(self, stage, data):
        try:
            binding = self._stages[stage]
        except (KeyError, TypeError) as exc:
            raise AdapterUnavailable("declared stage is unavailable") from exc
        if not isinstance(data, Mapping):
            raise AdapterUnavailable("stage data is unavailable")
        kind, module_name, attribute, args, kwargs, roles = binding
        if any(role not in data for role in roles):
            raise AdapterUnavailable(
                "required stage-data roles are unavailable")
        if kind == "method" and "bundle" not in data:
            raise AdapterUnavailable("method receiver bundle is unavailable")

        try:
            target = (importlib.import_module(module_name)
                      if kind == "callable" else data["bundle"])
            if self._root is not None and kind == "callable":
                _require_in_root(target, self._root)
            for segment in attribute.split("."):
                target = getattr(target, segment)
        except (ImportError, AttributeError) as exc:
            raise AdapterUnavailable(
                "declared entrypoint is unavailable") from exc
        if not callable(target):
            raise AdapterUnavailable("declared entrypoint is not callable")
        if self._root is not None:
            _require_in_root(target, self._root)

        positional = [data[role] for role in args]
        named = {parameter: data[role] for parameter, role in kwargs.items()}
        try:
            inspect.signature(target).bind(*positional, **named)
        except (TypeError, ValueError) as exc:
            raise AdapterUnavailable(
                "declared argument mapping cannot bind") from exc
        return target, positional, named

    def _invoke(self, stage, data):
        target, positional, named = self._resolve(stage, data)
        return target(*positional, **named)

    def build(self, data):
        return self._invoke("build", data)

    def recover(self, data):
        return self._invoke("recover", data)

    def validate(self, data):
        return self._invoke("validate", data)

    def preflight(self, stage, data):
        self._resolve(stage, data)

    def kernels(self, bundle):
        kind, cosine, sine = self._kernel_binding
        if kind == "mapping":
            if not isinstance(bundle, Mapping):
                raise AdapterUnavailable(
                    "declared mapping kernel result is unavailable")
            if cosine not in bundle or sine not in bundle:
                raise AdapterUnavailable(
                    "declared mapping kernels are unavailable")
            try:
                return bundle[cosine], bundle[sine]
            except KeyError as exc:
                raise AdapterUnavailable(
                    "declared mapping kernels are unavailable") from exc
        if kind == "attributes":
            try:
                return getattr(bundle, cosine), getattr(bundle, sine)
            except AttributeError as exc:
                raise AdapterUnavailable(
                    "declared attribute kernels are unavailable") from exc
        if not isinstance(bundle, (list, tuple)):
            raise AdapterUnavailable(
                "declared sequence kernel result is unavailable")
        try:
            return bundle[cosine], bundle[sine]
        except IndexError as exc:
            raise AdapterUnavailable(
                "declared sequence kernels are unavailable") from exc


def bind(declaration, *, artifact_root=None):
    """Bind an exact E08 declaration without invoking artifact code."""
    fields = {
        "schema_version", "build", "recover", "validate", "kernels", "domain",
        "phase_policy", "recovery_semantics", "validation_mode"
    }
    if not isinstance(declaration, dict) or set(declaration) != fields:
        raise ValueError("invalid binding fields")
    if type(declaration["schema_version"]
            ) is not int or declaration["schema_version"] != 1:
        raise ValueError("unsupported binding schema version")
    phase_policy = declaration["phase_policy"]
    recovery_semantics = declaration["recovery_semantics"]
    validation_mode = declaration["validation_mode"]
    if not isinstance(phase_policy, str) or phase_policy not in {
            "exact", "ray"
    }:
        raise ValueError("invalid phase policy")
    if (not isinstance(recovery_semantics, str)
            or recovery_semantics != "real_linear"):
        raise ValueError("invalid recovery semantics")
    if (not isinstance(validation_mode, str)
            or validation_mode not in {"raises", "boolean"}):
        raise ValueError("invalid validation mode")

    # Validate and detach every nested declaration before artifact resolution.
    stages = {
        stage: _stage_declaration(stage, declaration[stage])
        for stage in ("build", "recover", "validate")
    }
    kernel_binding = _kernel_declaration(declaration["kernels"])
    domain = _domain_declaration(declaration["domain"])
    root = _root_path(artifact_root) if artifact_root is not None else None
    return _BoundEvolution(stages, kernel_binding, root, domain, phase_policy,
                           recovery_semantics, validation_mode)
