# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fair source staging, restrictive native profiles and runtime metadata."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

_CODEX = os.environ.get("CUDAQ_E2E_CODEX") or shutil.which("codex")
CODEX = Path(_CODEX).resolve() if _CODEX else Path("/nonexistent/codex")
REPO = Path(__file__).resolve().parents[4]
SKILL = REPO / "skills/cudaq-algorithms"
MODEL = "gpt-5.5"
PROFILE = "cudaq_e2e"
EXCLUDE = {
    ".git", ".agents", ".codex", "__pycache__", ".pytest_cache", "_build",
    "build", "AGENTS.md"
}


def inventory(root):
    result = {}
    for path in sorted(Path(root).rglob("*")):
        name = str(path.relative_to(root))
        if path.is_symlink():
            result[name] = "symlink:" + os.readlink(path)
        elif path.is_file():
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def scope_changes(before, after):
    return [
        name for name in sorted(before.keys() | after.keys())
        if before.get(name) != after.get(name) and name not in
        {"app.py", "result.npz"} and not name.startswith(".tmp/")
    ]


def _copy_tree(source, destination):

    def ignore(folder, names):
        return [
            name for name in names if name in EXCLUDE or name.endswith((
                ".pyc", ".pyo")) or (Path(folder) / name).is_symlink()
        ]

    shutil.copytree(source, destination, ignore=ignore)


def stage_workspace(source,
                    workspace,
                    arm,
                    public,
                    *,
                    skill_root=None,
                    rg_path=None):
    if arm not in ("baseline", "previous", "skill"):
        raise ValueError(arm)
    source, workspace = Path(source), Path(workspace)
    workspace.mkdir(parents=True, exist_ok=False)
    for name in ("python", "docs/sphinx", "tests/python"):
        if (source / name).is_dir():
            (workspace / name).parent.mkdir(parents=True, exist_ok=True)
            _copy_tree(source / name, workspace / name)
    for name in ("pyproject.toml", "README.md", "LICENSE", "LICENSE.txt"):
        if (source / name).is_file():
            shutil.copy2(source / name, workspace / name)
    if arm != "baseline":
        origin = Path(
            skill_root
        ) if skill_root is not None else source / "skills/cudaq-algorithms"
        dest = workspace / "skills/cudaq-algorithms"
        dest.mkdir(parents=True)
        shutil.copy2(origin / "SKILL.md", dest / "SKILL.md")
        for part in ("references", "assets"):
            if (origin / part).exists():
                _copy_tree(origin / part, dest / part)
    search = Path(rg_path) if rg_path is not None else source / "tools/rg"
    if search.is_file():
        (workspace / "tools").mkdir()
        shutil.copy2(search, workspace / "tools/rg")
    (workspace / "input.json").write_text(json.dumps(public, indent=2))
    (workspace / ".tmp").mkdir()


def stage_runtime_artifacts(workspace, case_id):
    """Contain fixed provider side files without allowing new root artifacts."""
    if case_id != "psi4_energy":
        return
    workspace = Path(workspace)
    scratch = workspace / ".tmp"
    try:
        info = scratch.lstat()
        workspace_root = workspace.resolve(strict=True)
        scratch_root = scratch.resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            "Psi4 scratch must be an existing workspace directory") from exc
    if scratch.is_symlink() or not stat.S_ISDIR(
            info.st_mode) or scratch_root.parent != workspace_root:
        raise ValueError(
            "Psi4 scratch must be a real directory inside the workspace")
    timer = workspace / "timer.dat"
    if os.path.lexists(timer):
        raise FileExistsError(
            f"refusing to replace existing runtime artifact: {timer}")
    timer.symlink_to(".tmp/timer.dat")


def _is_psi4_conda_runtime(python):
    """Detect Psi4 from inert conda records in the interpreter's allowed prefix."""
    metadata = Path(python).absolute().parent.parent / "conda-meta"
    if metadata.is_symlink() or not metadata.is_dir():
        return False
    for record in metadata.glob("psi4-*.json"):
        if record.is_symlink() or not record.is_file():
            continue
        try:
            payload = json.loads(record.read_text())
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(payload, dict) and payload.get("name") == "psi4":
            return True
    return False


def profile_args(workspace, python):
    workspace = Path(workspace).resolve()
    environment = Path(python).absolute().parent.parent
    permissions = {
        ":root": "deny",
        ":minimal": "read",
        ":workspace_roots": "write",
        str(CODEX): "read",
        str(REPO): "deny",
        str(environment): "read",
        str(Path.home() / ".codex/skills"): "deny",
        str(Path.home() / ".codex/plugins"): "deny"
    }
    if _is_psi4_conda_runtime(python):
        # oneMKL locates its loader through /proc/self/exe. This also exposes
        # the already-sanitized environment of this worker, never other PIDs.
        permissions["/proc/self"] = "read"
    for name in ("python", "docs", "tests", "skills", "tools", "input.json",
                 "pyproject.toml", "README.md", "LICENSE", "LICENSE.txt",
                 "_trace_app.py"):
        if (workspace / name).exists():
            permissions[str(workspace / name)] = "read"
    table = "{" + ",".join(
        json.dumps(k) + "=" + json.dumps(v)
        for k, v in permissions.items()) + "}"
    return [
        "-c", f"permissions.{PROFILE}.filesystem=" + table, "-c",
        f"permissions.{PROFILE}.network.enabled=false"
    ]


def disabled_skills():
    paths = {str(SKILL / "SKILL.md")}
    for root in (Path.home() / ".codex/skills",
                 Path.home() / ".codex/plugins/cache", Path("/opt/skills")):
        if Path(root).exists():
            paths.update(str(p) for p in Path(root).rglob("SKILL.md"))
    return "skills.config=[" + ",".join(
        "{path=" + json.dumps(p) + ",enabled=false}"
        for p in sorted(paths)) + "]"


def configuration(workspace, python):
    return [
        "--ignore-user-config", "-c", f'default_permissions="{PROFILE}"',
        *profile_args(workspace, python), "-c", 'model_reasoning_effort="low"',
        "-c", 'web_search="disabled"', "-c", 'approval_policy="never"', "-c",
        "project_doc_max_bytes=0", "-c", "features.multi_agent=false", "-c",
        "features.apps=false", "-c", "features.plugins=false", "-c",
        "features.hooks=false", "-c", "features.memories=false", "-c",
        "features.browser_use=false", "-c", "features.computer_use=false",
        "-c", "features.image_generation=false", "-c",
        "features.shell_snapshot=false", "-c",
        disabled_skills()
    ]


def environment(workspace):
    # Preserve native login location, not API credentials or arbitrary secrets.
    keep = {
        "PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TERM",
        "CODEX_HOME", "LD_LIBRARY_PATH"
    }
    env = {key: value for key, value in os.environ.items() if key in keep}
    env.update({
        "PYTHONPATH":
        str(Path(workspace) / "python"),
        "PYTHONDONTWRITEBYTECODE":
        "1",
        "PYTHONPYCACHEPREFIX":
        str(Path(workspace) / ".tmp/pycache"),
        "PATH":
        str(Path(workspace) / "tools") + os.pathsep +
        env.get("PATH", os.defpath),
        "OMP_NUM_THREADS":
        "1",
        "OPENBLAS_NUM_THREADS":
        "1",
        "MKL_NUM_THREADS":
        "1",
        "TMPDIR":
        str(Path(workspace) / ".tmp"),
        "PSI_SCRATCH":
        str(Path(workspace) / ".tmp"),
        "CUDAQ_DEFAULT_SIMULATOR":
        "qpp-cpu"
    })
    return env


def search_tool_metadata(path=None):
    """Identify an actual installed ripgrep; never synthesize a replacement."""
    selected = path or shutil.which("rg")
    if selected is None:
        return None
    selected = Path(selected).resolve(strict=True)
    result = subprocess.run([str(selected), "--version"],
                            capture_output=True,
                            text=True,
                            timeout=10)
    result.check_returncode()
    if not result.stdout.startswith("ripgrep "):
        raise ValueError("search tool must be a real ripgrep executable")
    return {
        "source_path": str(selected),
        "worker_path": "tools/rg",
        "version": result.stdout.strip(),
        "sha256": hashlib.sha256(selected.read_bytes()).hexdigest()
    }


def sandbox_command(workspace, python, args):
    return [
        str(CODEX), "sandbox", "-P", PROFILE, *profile_args(workspace, python),
        "-C",
        str(workspace), "--",
        str(python), "-B", *map(str, args)
    ]


def runtime_metadata(python):
    script = (
        'import importlib.metadata as m,json,sys; '
        'print(json.dumps({"python":sys.version,"executable":sys.executable,'
        '"distributions":{d.metadata["Name"]:d.version for d in m.distributions()}}))'
    )
    result = subprocess.run([str(python), "-c", script],
                            capture_output=True,
                            text=True,
                            timeout=30)
    result.check_returncode()
    metadata = json.loads(result.stdout)
    conda_records = Path(python).absolute().parent.parent / "conda-meta"
    if conda_records.is_dir():
        metadata["conda_packages"] = {}
        for path in sorted(conda_records.glob("*.json")):
            record = json.loads(path.read_text())
            metadata["conda_packages"][record["name"]] = {
                "version": record["version"],
                "build": record["build"]
            }
    return metadata


def isolation_probe(workspace, python, forbidden):
    script = '''import errno, json, pathlib, socket
results = {}
for name in FORBIDDEN:
    try:
        pathlib.Path(name).read_bytes()
        results[name] = False
    except (PermissionError, FileNotFoundError):
        results[name] = True
for name in ["input.json", "python/cudaq_algorithms/__init__.py"]:
    path = pathlib.Path(name)
    assert path.read_bytes()
    try:
        with path.open("ab") as handle:
            handle.write(b"")
        results["readonly:"+name] = False
    except OSError as exc:
        results["readonly:"+name] = exc.errno in (errno.EACCES, errno.EPERM, errno.EROFS)
path = pathlib.Path(".tmp/probe")
path.write_text("ok")
assert path.read_text() == "ok"
path.unlink()
sock = None
try:
    sock = socket.socket()
    sock.settimeout(1)
    sock.connect(("127.0.0.1", PROBE_PORT))
    results["network_blocked"] = False
except OSError:
    results["network_blocked"] = True
finally:
    if sock is not None:
        sock.close()
print(json.dumps(results))
assert all(results.values()), results
'''.replace("FORBIDDEN", repr(list(map(str, forbidden))))
    import socket
    # A real listening parent endpoint makes connection-refused insufficient
    # for a false pass. It never receives model or authentication traffic.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        script = script.replace("PROBE_PORT", str(listener.getsockname()[1]))
        command = sandbox_command(workspace, python, ["-c", script])
        result = subprocess.run(command,
                                cwd=workspace,
                                env=environment(workspace),
                                capture_output=True,
                                text=True,
                                timeout=45)
    return {
        "passed": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": command
    }
