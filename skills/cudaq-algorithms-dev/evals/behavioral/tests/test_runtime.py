# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral-only runtime isolation and clean-interpreter regressions."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import site
import socket
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
BEHAVIORAL = HERE.parent
E2E = BEHAVIORAL.parent / "e2e"
CONTROLLER_SITE = Path(site.getsitepackages()[0]).resolve()
GLOBAL_CUDAQ = Path(
    f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}"
    "/dist-packages/cudaq/__init__.py")
CONTROLLER_CUDAQ = CONTROLLER_SITE / "cudaq/__init__.py"


def _adapter():
    """Fail as an assertion, rather than collection error, during TDD RED."""
    path = BEHAVIORAL / "behavioral_runtime.py"
    assert path.is_file(), "behavioral runtime adapter is not implemented"
    spec = importlib.util.spec_from_file_location(
        "behavioral_runtime_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _worker_python() -> Path:
    value = os.environ.get("BEHAVIORAL_WORKER_PYTHON")
    if not value:
        pytest.skip(
            "set BEHAVIORAL_WORKER_PYTHON to the clean stdlib-only venv")
    return Path(value)


def _runtime_environment(workspace: Path) -> dict[str, str]:
    sys.path.insert(0, str(E2E))
    try:
        from runtime import environment
        return environment(workspace)
    finally:
        sys.path.remove(str(E2E))


def _filesystem_permissions(arguments: list[str]) -> dict[str, str]:
    prefix = "permissions.cudaq_e2e.filesystem="
    encoded = next(arg[len(prefix):] for arg in arguments
                   if arg.startswith(prefix))
    return dict(
        re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"="([^"\\]*(?:\\.[^"\\]*)*)"',
                   encoded))


def test_configuration_preserves_clean_prefix_and_narrows_behavioral_roots(
        tmp_path):
    adapter = _adapter()
    python = _worker_python()
    workspace = tmp_path / "campaign/preflight/candidate"
    (workspace / "input").mkdir(parents=True)
    (workspace / ".tmp").mkdir()

    arguments = adapter.behavioral_configuration(workspace, str(python))
    permissions = _filesystem_permissions(arguments)
    clean_prefix = python.absolute().parent.parent.resolve()

    assert permissions[str(clean_prefix)] == "read"
    assert permissions[str(workspace.resolve())] == "write"
    assert permissions[str(workspace.resolve().parent)] == "deny"
    assert permissions[str((workspace / "input").resolve())] == "read"
    assert permissions["/usr/local/lib/python3.12/dist-packages"] == "deny"
    assert permissions[str(CONTROLLER_SITE)] == "deny"
    assert "permissions.cudaq_e2e.network.enabled=false" in arguments


def test_non_venv_and_nonclean_interpreters_are_rejected(tmp_path):
    adapter = _adapter()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="virtual environment"):
        adapter.behavioral_configuration(workspace, "/usr/bin/python3")
    with pytest.raises(ValueError, match="clean|distribution|cudaq"):
        adapter.behavioral_configuration(workspace, sys.executable)


@pytest.mark.parametrize("launcher", ["python", "python3"])
def test_native_command_resolves_path_python_to_clean_runtime_without_parent_mutation(
        tmp_path, launcher):
    adapter = _adapter()
    python = _worker_python()
    workspace = tmp_path / "workspace"
    (workspace / ".tmp").mkdir(parents=True)
    before_path = os.environ.get("PATH")
    command = [
        launcher, "-I", "-B", "-c",
        ("import importlib.util,json,pathlib,sys; "
         "print(json.dumps({'prefix':sys.prefix,'json':pathlib.Path(json.__file__).is_file(),"
         "'cudaq':importlib.util.find_spec('cudaq') is None,"
         "'algorithms':importlib.util.find_spec('cudaq_algorithms') is None}))"
         )
    ]

    wrapped = adapter.native_command(command, workspace, str(python))

    assert command[0] == launcher
    assert wrapped[0] == "/usr/bin/env"
    assert wrapped[1].split("=",
                            1)[1].split(os.pathsep,
                                        1)[0] == str(python.absolute().parent)
    assert wrapped[2] == "PYTHONNOUSERSITE=1"
    assert wrapped[-len(command):] == command
    assert os.environ.get("PATH") == before_path
    completed = subprocess.run(wrapped,
                               cwd=workspace,
                               env=_runtime_environment(workspace),
                               capture_output=True,
                               text=True,
                               timeout=30)
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    assert Path(observed["prefix"]).resolve() == python.absolute(
    ).parent.parent.resolve()
    assert observed["json"]
    assert observed["cudaq"] and observed["algorithms"]


def test_strict_sandbox_denies_global_packages_evaluator_sibling_and_network(
        tmp_path):
    adapter = _adapter()
    python = _worker_python()
    workspace = tmp_path / "campaign/preflight/candidate"
    (workspace / "input").mkdir(parents=True)
    (workspace / "input/evidence.txt").write_text("fixed\n")
    (workspace / ".tmp").mkdir()
    sibling = workspace.parent / "baseline/secret.txt"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("secret\n")
    other_repo = workspace.parent / "otherrepo/private.txt"
    other_repo.parent.mkdir()
    other_repo.write_text("private\n")
    forbidden = [
        GLOBAL_CUDAQ, CONTROLLER_CUDAQ, HERE / "test_runtime.py", sibling,
        other_repo
    ]
    script = """import importlib.util,json,pathlib,socket,subprocess,sys
results={}
results['clean_prefix']=pathlib.Path(sys.prefix).resolve()==pathlib.Path(CLEAN_PREFIX).resolve()
results['stdlib']=pathlib.Path(json.__file__).is_file()
for module in ('cudaq','cudaq_algorithms'):
 results['undiscoverable:'+module]=importlib.util.find_spec(module) is None
 try: __import__(module); results['unimportable:'+module]=False
 except ModuleNotFoundError: results['unimportable:'+module]=True
for launcher in ('python','python3'):
 child=subprocess.run([launcher,'-I','-B','-c',"import importlib.util,json,sys; print(json.dumps({'prefix':sys.prefix,'cudaq':importlib.util.find_spec('cudaq') is None,'algorithms':importlib.util.find_spec('cudaq_algorithms') is None}))"],capture_output=True,text=True)
 try: value=json.loads(child.stdout)
 except Exception: value={}
 results['path:'+launcher]=child.returncode==0 and pathlib.Path(value.get('prefix','/')).resolve()==pathlib.Path(CLEAN_PREFIX).resolve() and value.get('cudaq') is True and value.get('algorithms') is True
for name in FORBIDDEN:
 try: pathlib.Path(name).read_bytes(); results['deny:'+name]=False
 except (PermissionError,FileNotFoundError): results['deny:'+name]=True
p=pathlib.Path('input/evidence.txt'); original=p.read_bytes(); results['input_read']=original==b'fixed\\n'
try: p.write_bytes(b'changed'); results['input_write']=False
except OSError: results['input_write']=p.read_bytes()==original
q=pathlib.Path('.tmp/probe'); q.write_text('ok'); results['scratch']=q.read_text()=='ok'
s=None
try:
 s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1',PROBE_PORT)); results['network']=False
except OSError: results['network']=True
finally:
 if s is not None: s.close()
print(json.dumps(results)); assert all(results.values()),results
"""
    script = script.replace("CLEAN_PREFIX",
                            repr(str(python.absolute().parent.parent)))
    script = script.replace("FORBIDDEN",
                            repr([str(path) for path in forbidden]))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        script = script.replace("PROBE_PORT", str(listener.getsockname()[1]))
        command = adapter.behavioral_sandbox_command(workspace, str(python),
                                                     ["-c", script])
        completed = subprocess.run(command,
                                   cwd=workspace,
                                   env=_runtime_environment(workspace),
                                   capture_output=True,
                                   text=True,
                                   timeout=60)

    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    assert all(observed.values()), observed
