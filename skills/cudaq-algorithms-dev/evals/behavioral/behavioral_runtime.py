# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Clean-interpreter isolation adapter for the behavioral evaluator.

This module narrows the shared end-to-end runtime without changing that
runtime's globals or the controller process environment.
"""
from __future__ import annotations

from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path
import re
import site
import subprocess
import sys

HERE = Path(__file__).resolve().parent
E2E_RUNTIME = HERE.parent / "e2e/runtime.py"


def _load_e2e_runtime():
    spec = importlib.util.spec_from_file_location("_behavioral_e2e_runtime",
                                                  E2E_RUNTIME)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load e2e runtime helper: {E2E_RUNTIME}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_RUNTIME = _load_e2e_runtime()
PROFILE = _RUNTIME.PROFILE
_FILESYSTEM_PREFIX = f"permissions.{PROFILE}.filesystem="


@lru_cache(maxsize=8)
def _clean_runtime(python_name: str) -> dict:
    """Validate and describe a genuinely isolated, stdlib-only venv."""
    python = Path(python_name).absolute()
    prefix = python.parent.parent
    config = prefix / "pyvenv.cfg"
    if (not python.is_file() or not os.access(python, os.X_OK)
            or not config.is_file() or config.is_symlink()):
        raise ValueError(
            "behavioral worker interpreter must belong to a real virtual environment"
        )
    settings = {}
    for line in config.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            settings[key.strip().lower()] = value.strip().lower()
    if settings.get("include-system-site-packages") != "false":
        raise ValueError(
            "behavioral worker virtual environment must exclude system site packages"
        )

    script = (
        "import importlib.metadata as m,importlib.util,json,site,sys\n"
        "names=('cudaq','cudaq_algorithms','pip')\n"
        "print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,"
        "'version':[sys.version_info.major,sys.version_info.minor],"
        "'sites':site.getsitepackages(),"
        "'discoverable':{n:importlib.util.find_spec(n) is not None for n in names},"
        "'distributions':sorted(d.metadata.get('Name') or '' for d in m.distributions())}))\n"
    )
    completed = subprocess.run([str(python), "-I", "-B", "-c", script],
                               capture_output=True,
                               text=True,
                               timeout=30)
    if completed.returncode != 0:
        raise ValueError("behavioral worker virtual environment probe failed")
    try:
        metadata = json.loads(completed.stdout)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "behavioral worker virtual environment returned invalid metadata"
        ) from exc
    if (Path(metadata["prefix"]).resolve() != prefix.resolve()
            or Path(metadata["base_prefix"]).resolve() == prefix.resolve()):
        raise ValueError(
            "behavioral worker interpreter is not running in its virtual environment"
        )
    if metadata.get("distributions"):
        raise ValueError(
            "behavioral worker must be clean: installed distributions found")
    if any(metadata.get("discoverable", {}).values()):
        raise ValueError(
            "behavioral worker must be clean: cudaq, cudaq_algorithms, and pip must be absent"
        )
    metadata["prefix"] = str(prefix.resolve())
    metadata["bin"] = str(python.parent)
    return metadata


def _decode_permissions(encoded: str) -> dict[str, str]:
    if not encoded.startswith("{") or not encoded.endswith("}"):
        raise ValueError("unexpected native filesystem profile encoding")
    body = encoded[1:-1]
    matches = re.findall(r'("(?:\\.|[^"\\])*")=("(?:\\.|[^"\\])*")', body)
    permissions = {
        json.loads(key): json.loads(value)
        for key, value in matches
    }
    rebuilt = ",".join(key + "=" + value for key, value in matches)
    if rebuilt != body:
        raise ValueError("unexpected native filesystem profile encoding")
    return permissions


def _encode_permissions(permissions: dict[str, str]) -> str:
    return "{" + ",".join(
        json.dumps(key) + "=" + json.dumps(value)
        for key, value in permissions.items()) + "}"


def _controller_package_roots() -> set[Path]:
    roots = {Path(value) for value in site.getsitepackages()}
    roots.update(
        Path(value) for value in sys.path
        if value and Path(value).name in {"site-packages", "dist-packages"})
    for name in ("cudaq", "cudaq_algorithms"):
        spec = importlib.util.find_spec(name)
        if spec is not None and spec.origin:
            roots.add(Path(spec.origin).resolve().parent.parent)
    return roots


def _denied_package_roots(metadata: dict) -> set[Path]:
    major, minor = metadata["version"]
    version = f"python{major}.{minor}"
    roots = _controller_package_roots()
    roots.update({
        Path("/usr/local/lib") / version / "dist-packages",
        Path("/usr/local/lib") / version / "site-packages",
        Path("/usr/lib") / version / "dist-packages",
        Path("/usr/lib") / version / "site-packages",
        Path("/usr/lib/python3/dist-packages"),
    })
    clean = Path(metadata["prefix"])
    result = set()
    for root in roots:
        resolved = root.absolute().resolve()
        if not resolved.is_relative_to(clean):
            result.add(resolved)
    return result


def _adapt_profile(arguments: list[str], workspace: Path,
                   python: str) -> list[str]:
    metadata = _clean_runtime(str(Path(python).absolute()))
    workspace = Path(workspace).resolve()
    adapted = []
    found = False
    for argument in arguments:
        if argument.startswith(_FILESYSTEM_PREFIX):
            permissions = _decode_permissions(
                argument[len(_FILESYSTEM_PREFIX):])
            permissions[str(workspace.parent)] = "deny"
            permissions[str(workspace)] = "write"
            permissions[metadata["prefix"]] = "read"
            for root in sorted(_denied_package_roots(metadata), key=str):
                permissions[str(root)] = "deny"
            input_root = workspace / "input"
            if input_root.exists():
                permissions[str(input_root.resolve())] = "read"
            argument = _FILESYSTEM_PREFIX + _encode_permissions(permissions)
            found = True
        adapted.append(argument)
    if not found:
        raise ValueError("native filesystem profile was not present")
    return adapted


def behavioral_configuration(workspace, python) -> list[str]:
    """Return the shared native configuration with behavioral-only denials."""
    return _adapt_profile(_RUNTIME.configuration(workspace, python),
                          Path(workspace), python)


def native_command(command, workspace, python) -> list[str]:
    """Run a native command with clean-venv PATH selection, without mutation."""
    metadata = _clean_runtime(str(Path(python).absolute()))
    command = list(map(str, command))
    if not command:
        raise ValueError("native command must not be empty")
    inherited_path = _RUNTIME.environment(workspace).get("PATH", os.defpath)
    clean_path = metadata["bin"] + os.pathsep + inherited_path
    return [
        "/usr/bin/env", "PATH=" + clean_path, "PYTHONNOUSERSITE=1", *command
    ]


def behavioral_sandbox_command(workspace, python, args) -> list[str]:
    """Return a strict sandbox probe command under the clean worker PATH."""
    command = _adapt_profile(
        _RUNTIME.sandbox_command(workspace, python, list(args)),
        Path(workspace), python)
    return native_command(command, workspace, python)


__all__ = [
    "behavioral_configuration", "behavioral_sandbox_command", "native_command"
]
