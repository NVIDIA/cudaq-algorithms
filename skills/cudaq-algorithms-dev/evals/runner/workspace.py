"""Task-local inputs and fail-closed Linux command isolation.

The model client runs outside this sandbox. The command process sees only a
minimal operating system, its workspace, its optional read-only skill and the
configured Python runtime. ``trusted`` is explicitly unisolated and intended
only for offline tests; it is never a fallback for a missing sandbox.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import stat
import subprocess
import tempfile
import time
import uuid


class IsolationUnavailable(RuntimeError):
    """The requested command sandbox could not be established."""


_EXCLUDED = {
    ".git",
    ".agents",
    ".codex",
    ".claude",
    ".ssh",
    ".env",
    "AGENTS.md",
    "CLAUDE.md",
    "SKILL.md",
    "skills",
    "evals",
    "__pycache__",
    ".pytest_cache",
    "credentials",
    "secrets",
}
_SOURCE_PATHS = ("python", "tests/python", "docs/sphinx", "pyproject.toml",
                 "README.md", "LICENSE", "LICENSE.txt", ".cudaq_version")


def _real_path(path: Path, *, directory: bool = False) -> Path:
    path = Path(path).absolute()
    if ".." in path.parts:
        raise ValueError(f"path traversal is forbidden: {path}")
    for current in (*reversed(path.parents), path):
        if current.is_symlink():
            raise ValueError(f"symlink is forbidden: {current}")
    if directory and not path.is_dir():
        raise ValueError(f"expected an existing directory: {path}")
    return path


def _relative(name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("fixture names must be nonempty relative POSIX paths")
    parsed = PurePosixPath(name)
    if parsed.is_absolute() or ".." in parsed.parts or not parsed.parts:
        raise ValueError(f"unsafe fixture path: {name}")
    return Path(*parsed.parts)


def _copy_public(source: Path, destination: Path) -> None:
    source = _real_path(source)
    info = source.stat()
    if stat.S_ISDIR(info.st_mode):
        destination.mkdir(parents=True, exist_ok=True)
        for child in sorted(source.iterdir()):
            if child.name in _EXCLUDED or child.name.startswith(".env."):
                continue
            if child.suffix in (".pyc", ".pyo"):
                continue
            _copy_public(child, destination / child.name)
    elif stat.S_ISREG(info.st_mode):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    else:
        raise ValueError(
            f"only regular files and directories may be staged: {source}")


def stage_workspace(source: Path, skill: Path, fixtures_root: Path, case: dict,
                    attempt: Path, arm: str) -> dict:
    """Stage original declared inputs; grading fields are never serialized.

    Host ``attempt/workspace`` appears as ``/workspace`` to commands, and the
    skill arm's ``attempt/skill`` appears as ``/skill``. Fixture relative paths
    are preserved below ``/workspace/.eval-inputs`` to avoid basename collisions.
    """
    if arm not in ("baseline", "skill"):
        raise ValueError("arm must be baseline or skill")
    source = _real_path(source, directory=True)
    fixtures_root = _real_path(fixtures_root, directory=True)
    attempt = _real_path(attempt)
    files = case.get("files", [])
    if not isinstance(files, list):
        raise ValueError("case files must be a list")
    declared = []
    seen = set()
    for name in files:
        relative = _relative(name)
        if relative in seen:
            raise ValueError(f"duplicate fixture path: {name}")
        seen.add(relative)
        origin = _real_path(fixtures_root / relative)
        if not origin.is_file() or not stat.S_ISREG(origin.stat().st_mode):
            raise ValueError(f"fixture must be a regular file: {name}")
        declared.append((name, relative, origin))
    if arm == "skill":
        skill = _real_path(skill, directory=True)
        _real_path(skill / "SKILL.md")
        if not (skill / "SKILL.md").is_file():
            raise ValueError("skill root must contain SKILL.md")
    for name in ("workspace", "skill"):
        if os.path.lexists(attempt / name):
            raise FileExistsError(
                f"refusing to replace staged attempt content: {attempt / name}"
            )
    attempt.mkdir(parents=True, exist_ok=True)
    workspace = attempt / "workspace"
    workspace.mkdir()
    for name in _SOURCE_PATHS:
        origin = source / name
        if origin.exists() or origin.is_symlink():
            _copy_public(origin, workspace / name)
    (workspace / ".tmp").mkdir()
    (workspace / ".eval-inputs").mkdir()
    fixture_paths = {}
    for name, relative, origin in declared:
        target = workspace / ".eval-inputs" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target)
        target.chmod(0o444)
        fixture_paths[name] = "/workspace/.eval-inputs/" + relative.as_posix()
    staged_skill = None
    if arm == "skill":
        staged_skill = attempt / "skill"
        staged_skill.mkdir()
        shutil.copy2(skill / "SKILL.md", staged_skill / "SKILL.md")
        for name in ("references", "assets", "scripts"):
            origin = skill / name
            if origin.exists() or origin.is_symlink():
                _copy_public(origin, staged_skill / name)
    return {
        "workspace_path": str(workspace),
        "skill_path": str(staged_skill) if staged_skill else None,
        "fixture_paths": fixture_paths
    }


# Every interpolated value below is a positional argument, never shell source.
# Mounts exist only in the newly created private namespace. The coordinator and
# its credentials never enter the namespace. The chrooted command has no Linux
# capabilities, no external network interface and no other processes' /proc.
_NAMESPACE_SCRIPT = r'''
set -euo pipefail
jail="$1"; workspace="$2"; runtime="$3"; command="$4"
mount -t tmpfs -o nosuid,nodev,size=32m tmpfs "$jail"
mkdir -p "$jail/usr" "$jail/bin" "$jail/sbin" "$jail/lib" "$jail/lib64" "$jail/etc" "$jail/dev" "$jail/proc" "$jail/tmp" "$jail/workspace"
readonly_bind() {
    mount --bind "$1" "$2"
    mount -o remount,bind,ro,nosuid,nodev "$2"
}
for name in usr bin sbin lib lib64; do
    if [ -d "/$name" ]; then readonly_bind "/$name" "$jail/$name"; fi
done
for name in ld.so.cache ld.so.conf localtime; do
    if [ -f "/etc/$name" ]; then touch "$jail/etc/$name"; readonly_bind "/etc/$name" "$jail/etc/$name"; fi
done
for name in null zero random urandom; do
    touch "$jail/dev/$name"
    mount --bind "/dev/$name" "$jail/dev/$name"
done
mount -t proc -o nosuid,nodev,noexec proc "$jail/proc"
mount --bind "$workspace" "$jail/workspace"
mount --bind "$workspace/.tmp" "$jail/tmp"
if [ -d "$workspace/.eval-inputs" ]; then
    readonly_bind "$workspace/.eval-inputs" "$jail/workspace/.eval-inputs"
fi
if [ -d "${workspace%/*}/skill" ]; then
    mkdir "$jail/skill"
    readonly_bind "${workspace%/*}/skill" "$jail/skill"
fi
if [ -n "$runtime" ]; then
    case "$runtime" in /usr|/usr/*|/bin|/bin/*|/lib|/lib/*|/lib64|/lib64/*) ;;
    *) mkdir -p "$jail$runtime"; readonly_bind "$runtime" "$jail$runtime" ;;
    esac
fi
mount -o remount,ro,nosuid,nodev "$jail"
cd "$jail/workspace"
exec chroot "$jail" /usr/bin/setpriv --bounding-set=-all --no-new-privs /bin/bash --noprofile --norc -c 'cd /workspace; exec /bin/bash --noprofile --norc -c "$1"' cudaq-worker "$command"
'''


def _runtime(runtime_python: Path | None) -> tuple[str, str]:
    if runtime_python is None:
        return "", "/usr/bin:/bin:/usr/sbin:/sbin"
    # Python itself may be the usual venv symlink; its prefix must be real.
    executable = Path(runtime_python).absolute()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("runtime_python must be an executable interpreter")
    if str(executable.parent) in ("/usr/bin", "/usr/local/bin", "/bin"):
        if not executable.resolve().is_relative_to("/usr"):
            raise ValueError("system runtime must resolve inside /usr")
        return "", str(executable.parent) + ":/usr/bin:/bin:/usr/sbin:/sbin"
    prefix = _real_path(executable.parent.parent, directory=True)
    if not (prefix / "pyvenv.cfg").is_file() and not (prefix /
                                                      "conda-meta").is_dir():
        raise ValueError(
            "custom runtime must be a dedicated virtualenv or conda prefix")
    for marker in (prefix / "pyvenv.cfg", prefix / "conda-meta"):
        _real_path(marker)
    return str(prefix), str(
        executable.parent) + ":/usr/bin:/bin:/usr/sbin:/sbin"


def _environment(workspace: Path, path: str, isolated: bool) -> dict:
    root = "/workspace" if isolated else str(workspace)
    return {
        "PATH": path,
        "HOME": root + "/.tmp",
        "LANG": "C.UTF-8",
        "TERM": "dumb",
        "PYTHONPATH": root + "/python",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPYCACHEPREFIX": root + "/.tmp/pycache",
        "TMPDIR": root + "/.tmp",
        "XDG_CACHE_HOME": root + "/.tmp/cache",
        "CUDAQ_DEFAULT_SIMULATOR": "qpp-cpu",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1"
    }


def run_command(command: str,
                workspace: Path,
                *,
                timeout: float,
                runtime_python: Path | None = None,
                isolation: str = "unshare",
                log_dir: Path | None = None,
                max_output_bytes: int | None = None) -> dict:
    """Run one shell command with full output and a process-tree deadline.

    ``unshare`` needs Linux mount/PID/network namespace permission, mount,
    chroot and setpriv. An unavailable sandbox raises before the command runs.
    Pass ``log_dir`` to retain full stdout/stderr files outside the workspace.
    """
    if isinstance(timeout, bool) or not isinstance(
            timeout,
        (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite positive number")
    if isolation not in ("unshare", "trusted"):
        raise ValueError("isolation must be unshare or trusted")
    if max_output_bytes is not None and (type(max_output_bytes) is not int
                                         or max_output_bytes <= 0):
        raise ValueError('max_output_bytes must be a positive integer')
    if not isinstance(command, str) or "\x00" in command:
        raise ValueError("command must be a string without NUL bytes")
    workspace = _real_path(workspace, directory=True)
    for path in (workspace / ".tmp", workspace / ".eval-inputs",
                 workspace.parent / "skill"):
        _real_path(path)
    (workspace / ".tmp").mkdir(exist_ok=True)
    prefix, path = _runtime(runtime_python)
    if isolation == "unshare":
        missing = [
            name for name in ("unshare", "mount", "chroot", "setpriv", "bash")
            if not shutil.which(name)
        ]
        if missing:
            raise IsolationUnavailable("missing namespace tools: " +
                                       ", ".join(missing))
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="cudaq-command-") as temporary:
        temporary = Path(temporary)
        jail = temporary / "root"
        jail.mkdir()
        output = _real_path(log_dir) if log_dir else temporary
        output.mkdir(parents=True, exist_ok=True)
        identity = uuid.uuid4().hex
        stdout_path, stderr_path = (output / (identity + suffix)
                                    for suffix in (".stdout.log",
                                                   ".stderr.log"))
        argv = ["/bin/bash", "--noprofile", "--norc", "-c", command]
        if isolation == "unshare":
            argv = [
                shutil.which("unshare"), "--mount", "--propagation", "private",
                "--pid", "--fork", "--kill-child=KILL", "--net", "/bin/bash",
                "--noprofile", "--norc", "-c", _NAMESPACE_SCRIPT,
                "cudaq-sandbox",
                str(jail),
                str(workspace), prefix, command
            ]
        timed_out = False
        output_limit_exceeded = False
        with stdout_path.open("xb") as stdout, stderr_path.open(
                "xb") as stderr:
            process = subprocess.Popen(argv,
                                       cwd=workspace,
                                       env=_environment(
                                           workspace, path,
                                           isolation == "unshare"),
                                       stdin=subprocess.DEVNULL,
                                       stdout=stdout,
                                       stderr=stderr,
                                       start_new_session=True)
            try:
                if max_output_bytes is None:
                    process.wait(timeout=timeout)
                else:
                    deadline = time.monotonic() + timeout
                    while process.poll() is None:
                        if stdout_path.stat().st_size + stderr_path.stat(
                        ).st_size > max_output_bytes:
                            output_limit_exceeded = True
                            break
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            timed_out = True
                            break
                        try:
                            process.wait(timeout=min(.05, remaining))
                        except subprocess.TimeoutExpired:
                            pass
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                # Also discard background children after a normally exited shell.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        output_limit_exceeded = output_limit_exceeded or (
            max_output_bytes is not None and stdout_path.stat().st_size +
            stderr_path.stat().st_size > max_output_bytes)

        def read_output(path):
            with path.open('rb') as handle:
                return handle.read(max_output_bytes
                                   or -1).decode(errors='replace')

        result = {
            "exit_code": process.returncode,
            "stdout": read_output(stdout_path),
            "stderr": read_output(stderr_path),
            "wall_seconds": time.monotonic() - started,
            "timed_out": timed_out,
            "isolated": isolation == "unshare",
            "isolation": isolation
        }
        if max_output_bytes is not None:
            result['output_limit_exceeded'] = output_limit_exceeded
        if log_dir:
            result.update(stdout_log=str(stdout_path),
                          stderr_log=str(stderr_path))
        if isolation == "unshare" and process.returncode and not timed_out and (
                result["stderr"].startswith("unshare:")
                or "mount: " in result["stderr"]
                and "permission denied" in result["stderr"]):
            raise IsolationUnavailable("namespace setup failed: " +
                                       result["stderr"].strip())
        return result


def preflight(isolation: str = "unshare",
              runtime_python: Path | None = None) -> dict:
    """Require a real CPU, fp64 Bell-state execution inside the sandbox."""
    with tempfile.TemporaryDirectory(
            prefix="cudaq-sandbox-preflight-") as temporary:
        workspace = Path(temporary) / "workspace"
        workspace.mkdir()
        python = str(runtime_python) if runtime_python else "/usr/bin/python3"
        import shlex
        (workspace / "probe.py").write_text('''import json, sys
import cudaq
import numpy as np
@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])
state = np.asarray(cudaq.get_state(bell))
assert cudaq.get_target().name == 'qpp-cpu'
assert state.dtype == np.dtype('complex128')
assert np.allclose(state, [2**-.5, 0, 0, 2**-.5], atol=1e-12, rtol=1e-12)
print(json.dumps({'runtime': sys.version, 'cudaq': cudaq.__version__,
                  'target': cudaq.get_target().name, 'precision': 'fp64',
                  'science_verified': True}))
''')
        result = run_command(shlex.quote(python) + " probe.py",
                             workspace,
                             timeout=45,
                             runtime_python=runtime_python,
                             isolation=isolation)
        if result["exit_code"] != 0 or result["timed_out"]:
            raise IsolationUnavailable("runtime sandbox probe failed: " +
                                       result["stderr"])
        measured = json.loads(result["stdout"])
        return {
            "available": True,
            "isolated": result["isolated"],
            "isolation": isolation,
            **measured
        }
