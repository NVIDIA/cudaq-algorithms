"""Workspace leakage, immutable inputs, and real command lifecycle contracts."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import pytest


def test_verifier_output_limit_preserves_logs_but_bounds_loaded_output(
        subject, tmp_path):
    worker = tmp_path / 'workspace'
    worker.mkdir()
    result = subject.run_command("python3 -c 'print(\"x\"*100000)'",
                                 worker,
                                 timeout=5,
                                 isolation='trusted',
                                 log_dir=tmp_path / 'logs',
                                 max_output_bytes=1024)
    assert result['output_limit_exceeded']
    assert len(result['stdout']) <= 1024
    assert Path(result['stdout_log']).stat().st_size > 1024


@pytest.fixture
def subject():
    path = Path(__file__).parents[1] / "workspace.py"
    assert path.is_file(
    ), "isolated workspace executor has not been implemented"
    spec = importlib.util.spec_from_file_location("eval_workspace", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs(tmp_path):
    source, skill, fixtures = (tmp_path / name
                               for name in ("source", "skill", "fixtures"))
    for name, text in {
            "python/cudaq_algorithms/library.py": "PUBLIC_LIBRARY = 1\n",
            "tests/python/test_public.py": "PUBLIC_TEST = True\n",
            "docs/sphinx/example.py": "PUBLIC_EXAMPLE = 1\n",
            "pyproject.toml": "[project]\nname='example'\n",
            "skills/other/SKILL.md": "PRIVATE SKILL",
            "python/.env": "PRIVATE CREDENTIAL",
            "python/evals/oracle.py": "PRIVATE ORACLE",
            ".git/config": "PRIVATE HISTORY",
    }.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text("name: cudaq-algorithms\nPUBLIC SKILL\n")
    (skill / "references/usage.md").write_text("PUBLIC REFERENCE\n")
    (skill / "evals").mkdir()
    (skill / "evals/answers.json").write_text("PRIVATE ANSWERS")
    (fixtures / "files").mkdir(parents=True)
    (fixtures / "files/input.txt").write_text("PUBLIC INPUT\n")
    return source, skill, fixtures


def test_staging_changes_only_skill_and_never_copies_grading_material(
        subject, inputs, tmp_path):
    case = {
        "prompt": "ORIGINAL PROMPT",
        "files": ["files/input.txt"],
        "assertions": ["PRIVATE ASSERTION"],
        "expected_output": "PRIVATE EXPECTATION"
    }
    stages = [
        subject.stage_workspace(*inputs, case, tmp_path / "attempts" / arm,
                                arm) for arm in ("baseline", "skill")
    ]
    inventories = []
    for stage in stages:
        workspace = Path(stage["workspace_path"])
        inventories.append({
            str(p.relative_to(workspace)): p.read_bytes()
            for p in workspace.rglob("*") if p.is_file()
        })
        assert (workspace / "python/cudaq_algorithms/library.py").is_file()
        assert (workspace / "tests/python/test_public.py").is_file()
        assert (workspace / "docs/sphinx/example.py").is_file()
        assert (workspace /
                ".eval-inputs/files/input.txt").read_text() == "PUBLIC INPUT\n"
        assert b"PRIVATE" not in b"".join(inventories[-1].values())
    assert inventories[0] == inventories[1]
    assert stages[0]["skill_path"] is None
    assert (Path(stages[1]["skill_path"]) / "SKILL.md").is_file()
    assert not (Path(stages[1]["skill_path"]) / "evals").exists()


@pytest.mark.parametrize("filename",
                         ["../secret", "/etc/passwd", "files/../../secret"])
def test_fixture_traversal_is_rejected_before_staging(subject, inputs,
                                                      tmp_path, filename):
    with pytest.raises(ValueError):
        subject.stage_workspace(*inputs, {"files": [filename]},
                                tmp_path / "attempt", "baseline")


def test_source_and_fixture_symlinks_are_rejected(subject, inputs, tmp_path):
    source, skill, fixtures = inputs
    (source / "python/leak.py").symlink_to(fixtures / "files/input.txt")
    with pytest.raises(ValueError, match="symlink"):
        subject.stage_workspace(*inputs, {"files": []}, tmp_path / "a",
                                "baseline")
    (source / "python/leak.py").unlink()
    (fixtures / "files/leak").symlink_to(fixtures / "files/input.txt")
    with pytest.raises(ValueError, match="symlink"):
        subject.stage_workspace(*inputs, {"files": ["files/leak"]},
                                tmp_path / "b", "baseline")


def test_source_parent_symlink_cannot_bypass_the_public_copy(
        subject, inputs, tmp_path):
    source, skill, fixtures = inputs
    (source / "tests/python/test_public.py").unlink()
    (source / "tests/python").rmdir()
    (source / "tests").rmdir()
    (fixtures / "python").mkdir()
    (fixtures / "python/private.py").write_text("PRIVATE")
    (source / "tests").symlink_to(fixtures, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        subject.stage_workspace(*inputs, {"files": []}, tmp_path / "attempt",
                                "baseline")


def test_existing_attempt_is_never_reused(subject, inputs, tmp_path):
    destination = tmp_path / "attempt"
    destination.mkdir()
    (destination / "workspace").mkdir()
    (destination / "evidence").write_text("preserve me")
    with pytest.raises(FileExistsError):
        subject.stage_workspace(*inputs, {"files": []}, destination,
                                "baseline")
    assert (destination / "evidence").read_text() == "preserve me"


def test_reserved_parent_can_be_staged_without_replacing_evidence(
        subject, inputs, tmp_path):
    destination = tmp_path / "attempt"
    destination.mkdir()
    (destination / "started.json").write_text('{"started": true}')
    result = subject.stage_workspace(*inputs, {"files": []}, destination,
                                     "baseline")
    assert Path(result["workspace_path"]).is_dir()
    assert (destination / "started.json").read_text() == '{"started": true}'


def test_trusted_mode_is_explicit_and_retains_full_output(subject, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    result = subject.run_command(
        "printf 'hello'; printf 'problem' >&2; exit 3",
        workspace,
        timeout=5,
        isolation="trusted")
    assert result["stdout"] == "hello"
    assert result["stderr"] == "problem"
    assert result["exit_code"] == 3
    assert result["timed_out"] is False
    assert result["isolated"] is False
    result = subject.run_command("python3 -c 'print(\"a\" * 25000)'",
                                 workspace,
                                 timeout=5,
                                 isolation="trusted")
    assert len(result["stdout"]) == 25001


def test_deadline_kills_descendants_even_after_stdout_closes(
        subject, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = "(sleep 1; echo leaked > late.txt) >/dev/null 2>&1 & exec sleep 20"
    started = time.monotonic()
    result = subject.run_command(command,
                                 workspace,
                                 timeout=0.1,
                                 isolation="trusted")
    assert result["timed_out"] is True
    assert time.monotonic() - started < 3
    time.sleep(1.1)
    assert not (workspace / "late.txt").exists()


def test_invalid_timeout_or_mode_never_executes(subject, tmp_path):
    for timeout in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            subject.run_command("touch executed",
                                tmp_path,
                                timeout=timeout,
                                isolation="trusted")
    with pytest.raises(ValueError):
        subject.run_command("touch executed",
                            tmp_path,
                            timeout=1,
                            isolation="unknown")
    assert not (tmp_path / "executed").exists()


def test_unavailable_isolation_never_falls_back_to_host(
        subject, tmp_path, monkeypatch):
    monkeypatch.setattr(subject.shutil, "which", lambda _: None)
    with pytest.raises(subject.IsolationUnavailable):
        subject.run_command("touch leaked", tmp_path, timeout=2)
    assert not (tmp_path / "leaked").exists()


def test_arbitrary_interpreter_parent_cannot_be_mounted_as_a_runtime(
        subject, tmp_path):
    (tmp_path / "bin").mkdir()
    interpreter = tmp_path / "bin/python"
    interpreter.symlink_to(sys.executable)
    with pytest.raises(ValueError, match="runtime"):
        subject.run_command("true",
                            tmp_path,
                            timeout=1,
                            runtime_python=interpreter,
                            isolation="trusted")


@pytest.mark.skipif(os.environ.get("CUDAQ_RUNNER_TEST_ISOLATION") != "1",
                    reason="requires approved Linux namespace capability")
def test_real_namespace_hides_host_and_blocks_network_and_input_writes(
        subject, inputs, tmp_path):
    stage = subject.stage_workspace(*inputs, {"files": ["files/input.txt"]},
                                    tmp_path / "attempt", "skill")
    workspace = Path(stage["workspace_path"])
    (tmp_path / "host-secret").write_text("PRIVATE")
    probe = '''import json, pathlib, socket
result = {}
result['skill'] = pathlib.Path('/skill/SKILL.md').read_text()
result['fixture'] = pathlib.Path('/workspace/.eval-inputs/files/input.txt').read_text()
result['host_hidden'] = not pathlib.Path(HOST_SECRET).exists()
result['homes_hidden'] = not pathlib.Path('/home').exists() and not pathlib.Path('/root').exists()
for name in ('/skill/SKILL.md', '/workspace/.eval-inputs/files/input.txt'):
    try:
        pathlib.Path(name).write_text('tamper')
        result[name] = False
    except OSError:
        result[name] = True
try:
    s = socket.socket(); s.settimeout(.1); s.connect(('1.1.1.1', 80))
    result['network_denied'] = False
except OSError:
    result['network_denied'] = True
pathlib.Path('output.txt').write_text('allowed')
print(json.dumps(result))
'''.replace("HOST_SECRET", repr(str(tmp_path / "host-secret")))
    (workspace / "probe.py").write_text(probe)
    result = subject.run_command("python3 probe.py", workspace, timeout=15)
    assert result["exit_code"] == 0, result["stderr"]
    data = json.loads(result["stdout"])
    assert data["host_hidden"] and data["homes_hidden"] and data[
        "network_denied"]
    assert data["/skill/SKILL.md"] and data[
        "/workspace/.eval-inputs/files/input.txt"]
    assert (workspace / "output.txt").read_text() == "allowed"
    assert result["isolated"] is True


@pytest.mark.skipif(
    not os.environ.get("CUDAQ_RUNNER_TEST_PYTHON"),
    reason="requires approved namespaces and a scientific runtime")
def test_runtime_preflight_executes_a_real_quantum_state_check(subject):
    result = subject.preflight(
        runtime_python=Path(os.environ["CUDAQ_RUNNER_TEST_PYTHON"]))
    assert result["available"] and result["isolated"]
    assert result["science_verified"] is True
    assert result["target"] == "qpp-cpu"
