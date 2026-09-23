# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit evaluator-owned API binding; execute only in the isolated checker.

No source discovery, expression evaluation, output conversion, or algorithm
fallback is permitted. Imported artifact code is untrusted, not sandboxed here.
"""
import importlib
import inspect
import re
from pathlib import Path


class AdapterUnavailable(RuntimeError):
    """The declared artifact entrypoint cannot be bound: acceptance is unknown."""


def _public_path(value):
    return (isinstance(value, str) and bool(
        re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*",
                     value)))


def _root_path(artifact_root):
    try:
        root = Path(artifact_root).resolve(strict=True)
    except (TypeError, ValueError, OSError) as exc:
        raise AdapterUnavailable(
            "artifact root origin is unavailable") from exc
    if not root.is_dir():
        raise AdapterUnavailable("artifact root origin is unavailable")
    return root


def _source_path(target):
    try:
        source = target.__file__ if inspect.ismodule(
            target) else inspect.getsourcefile(target)
        if source is None:
            raise TypeError("no source file")
        return Path(source).resolve(strict=True)
    except (AttributeError, TypeError, ValueError, OSError) as exc:
        raise AdapterUnavailable("declared origin is unavailable") from exc


def _require_in_root(target, root):
    source = _source_path(target)
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise AdapterUnavailable(
            "declared origin is outside artifact root") from exc
    return source


def verify_origin(target, artifact_root, relative_file):
    """Require target's canonical source to be one exact artifact-root file."""
    root = _root_path(artifact_root)
    try:
        expected = (root / relative_file).resolve(strict=True)
        expected.relative_to(root)
    except (TypeError, ValueError, OSError) as exc:
        raise AdapterUnavailable("expected origin is unavailable") from exc
    if _require_in_root(target, root) != expected:
        raise AdapterUnavailable(
            "declared origin does not match expected artifact file")


def bind(declaration, *, artifact_root=None):
    """Return (encoding, factor) -> artifact result, with no transformation."""
    if not isinstance(declaration, dict):
        raise ValueError("binding must be an object")
    kind = declaration.get("kind")
    fields = {
        "schema_version", "kind", "attribute", "args", "kwargs", "zero_policy"
    }
    if kind == "callable":
        fields.add("module")
    if (not isinstance(kind, str) or kind not in {"callable", "method"}
            or set(declaration) != fields
            or type(declaration["schema_version"]) is not int
            or declaration["schema_version"] != 1
            or not _public_path(declaration["attribute"])
            or declaration["zero_policy"] not in ("encode", "reject")):
        raise ValueError("invalid binding fields")
    if kind == "callable" and not _public_path(declaration["module"]):
        raise ValueError("module must be an explicit dotted import name")
    args, kwargs = declaration["args"], declaration["kwargs"]
    if (not isinstance(args, list) or not isinstance(kwargs, dict)
            or not all(_public_path(key) and "." not in key
                       for key in kwargs)):
        raise ValueError("invalid argument mapping")
    roles = list(args) + list(kwargs.values())
    required = ["encoding", "factor"] if kind == "callable" else ["factor"]
    if not all(isinstance(role, str)
               for role in roles) or sorted(roles) != sorted(required):
        raise ValueError(
            "map each required role exactly once; no constants or expressions")
    # Copy the validated declaration so callers cannot change selection later.
    args, kwargs = tuple(args), dict(kwargs)
    attribute, module_name = declaration["attribute"], declaration.get(
        "module")
    root = _root_path(artifact_root) if artifact_root is not None else None

    def resolve(encoding, factor):
        values = {"encoding": encoding, "factor": factor}
        try:
            target = importlib.import_module(
                module_name) if kind == "callable" else encoding
            if root is not None and kind == "callable":
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
        positional = [values[role] for role in args]
        named = {key: values[role] for key, role in kwargs.items()}
        try:
            inspect.signature(target).bind(*positional, **named)
        except (TypeError, ValueError) as exc:
            raise AdapterUnavailable(
                "declared argument mapping cannot bind") from exc
        return target, positional, named

    def invoke(encoding, factor):
        target, positional, named = resolve(encoding, factor)
        # Exceptions and return values from artifact execution stay untouched.
        return target(*positional, **named)

    invoke.preflight = lambda encoding: resolve(encoding, object()) and None
    return invoke
