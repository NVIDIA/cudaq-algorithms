# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit evaluator-owned E05 binding; execute only in the isolated checker.

Inputs and results pass through unchanged. There is no API discovery, synthesis,
normalization, phase repair, or output conversion. Artifact imports are untrusted;
the origin checks reused from E02 are provenance guards, not a security sandbox.
"""
from collections.abc import Mapping
import importlib
import inspect

from scale_adapter import (AdapterUnavailable, _public_path, _require_in_root,
                           _root_path)

_ROLES = {
    "basis_occupations": ("orbital_basis", "occupations", "coefficients"),
    "occupied_matrices": ("orbital_matrices", "coefficients"),
}


def bind(declaration, *, artifact_root=None):
    """Return ``invoke(case_data)`` and ``preflight`` (no factory invocation).

    Preflight imports and introspects artifact code, so it too must run only
    inside the isolated checker; it is not a non-executing safety check.

    The evaluator supplies all case data, including already-sliced occupied
    orbital matrices when that representation is selected. Only the selected
    representation's roles are required; unrelated case-data fields are ignored.
    ``coefficient_policy`` records the artifact's documented policy for the
    oracle. This adapter never implements either normalization or rejection.
    """
    fields = {
        "schema_version", "module", "attribute", "args", "kwargs",
        "representation", "coefficient_policy"
    }
    if not isinstance(declaration, dict) or set(declaration) != fields:
        raise ValueError("invalid binding fields")
    representation = declaration["representation"]
    policy = declaration["coefficient_policy"]
    if (type(declaration["schema_version"]) is not int
            or declaration["schema_version"] != 1
            or not isinstance(representation, str)
            or representation not in _ROLES or not isinstance(policy, str)
            or policy not in {"normalize", "reject"}
            or not _public_path(declaration["module"])
            or not _public_path(declaration["attribute"])):
        raise ValueError("invalid binding declaration")
    args, kwargs = declaration["args"], declaration["kwargs"]
    if (not isinstance(args, list) or not isinstance(kwargs, dict)
            or not all(_public_path(key) and "." not in key
                       for key in kwargs)):
        raise ValueError("invalid argument mapping")
    required = _ROLES[representation]
    roles = args + list(kwargs.values())
    if (not all(isinstance(role, str) for role in roles)
            or sorted(roles) != sorted(required)):
        raise ValueError(
            "map each required role exactly once; no constants or expressions")

    # Detach the declaration before any artifact import or invocation.
    args, kwargs = tuple(args), dict(kwargs)
    module_name, attribute = declaration["module"], declaration["attribute"]
    root = _root_path(artifact_root) if artifact_root is not None else None

    def resolve(case_data):
        if not isinstance(case_data, Mapping) or any(role not in case_data
                                                     for role in required):
            raise AdapterUnavailable(
                "required case-data roles are unavailable")
        try:
            target = importlib.import_module(module_name)
            if root is not None:
                _require_in_root(target, root)
            for segment in attribute.split("."):
                target = getattr(target, segment)
        except (ImportError, AttributeError) as exc:
            raise AdapterUnavailable(
                "declared entrypoint is unavailable") from exc
        if not callable(target):
            raise AdapterUnavailable("declared entrypoint is not callable")
        if root is not None:
            _require_in_root(target, root)
        positional = [case_data[role] for role in args]
        named = {key: case_data[role] for key, role in kwargs.items()}
        try:
            inspect.signature(target).bind(*positional, **named)
        except (TypeError, ValueError) as exc:
            raise AdapterUnavailable(
                "declared argument mapping cannot bind") from exc
        return target, positional, named

    def invoke(case_data):
        target, positional, named = resolve(case_data)
        return target(*positional, **named)

    def preflight(case_data):
        resolve(case_data)

    invoke.preflight = preflight
    return invoke
