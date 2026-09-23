# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private numerical checks run only in bounded, credential-free containers."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

import pytest


@pytest.fixture
def numerical(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "numerical.py"
    assert path.is_file(), "secret-free numerical runner is not implemented"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("strategy_numerical", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_command_is_isolated_and_uses_frozen_checks(numerical, tmp_path):
    repo, checker = (tmp_path / name for name in ("repo", "checker"))
    tests = tmp_path / "frozen/tests/python"
    for path in (repo, tests, checker):
        path.mkdir(parents=True)
    command = numerical.command(repo,
                                tests,
                                checker,
                                kind="targeted",
                                name="cudaq-check-test")
    assert command[:3] == ["docker", "run", "--rm"]
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--user") + 1] == "65534:65534"
    assert "--read-only" in command
    assert "ALL" in command and "no-new-privileges" in command
    assert "--env-file" not in command
    assert not any("KEY" in token or "TOKEN" in token for token in command)
    mounts = [
        command[i + 1] for i, token in enumerate(command) if token == "--mount"
    ]
    assert len(mounts) == 3 and all(m.endswith(",readonly") for m in mounts)
    assert "/checks/test_protocol_action.py" in command[-1]
    assert "PYTHONPATH=/workspace/project/python" in command
    assert "noexec" not in " ".join(
        command)  # CUDA-Q JIT requires executable tmpfs.


def test_regression_uses_original_tests_not_worker_tests(numerical, tmp_path):
    tests = tmp_path / "frozen/tests/python"
    tests.mkdir(parents=True)
    command = numerical.command(tmp_path,
                                tests,
                                tmp_path,
                                kind="regression",
                                name="cudaq-check-test")
    assert "/frozen-source/tests/python" in command[-1]
    assert "/workspace/project/tests" not in command[-1]
    assert "--confcutdir=/frozen-source/tests/python" in command[-1]
    mounts = [
        command[i + 1] for i, token in enumerate(command) if token == "--mount"
    ]
    assert f"type=bind,source={tmp_path / 'frozen'},target=/frozen-source,readonly" in mounts
    # Original tests locate docs via __file__.parents[2]; preserve that tree.
    assert Path("/frozen-source/tests/python/test_examples.py"
                ).parents[2] == Path("/frozen-source")


def test_untrusted_arguments_and_symlink_mounts_rejected(numerical, tmp_path):
    for kwargs in ({
            "kind": "unknown"
    }, {
            "name": "bad; command"
    }, {
            "image_id": "repo:latest"
    }):
        defaults = dict(kind="targeted", name="cudaq-check-test")
        defaults.update(kwargs)
        with pytest.raises(ValueError):
            numerical.command(tmp_path, tmp_path, tmp_path, **defaults)
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        numerical.command(alias,
                          tmp_path,
                          tmp_path,
                          kind="targeted",
                          name="cudaq-check-test")


def test_outcome_does_not_equate_exit_zero_with_scientific_pass(numerical):
    assert numerical.classify(0,
                              b"no tests ran\n")["status"] == "invalid_output"
    assert numerical.classify(0, b"12 passed in 1.2s\n")["status"] == "passed"
    assert numerical.classify(
        1, b"1 failed, 11 passed in 1.2s\n")["status"] == "failed"
    assert numerical.classify(124, b"")["status"] == "timeout"
    assert numerical.classify(
        0, b"x" * (numerical.MAX_OUTPUT + 1))["status"] == "output_over_limit"


def test_judge_gate_requires_frozen_test_coverage_and_maps_failures(numerical):
    complete = numerical.classify(0, b"307 passed, 4 skipped in 500s\n")
    assert numerical.judge_check(complete,
                                 expected_collected=311,
                                 allowed_skips=4) == "pass"
    skipped = numerical.classify(0, b"1 passed, 310 skipped in 2s\n")
    assert numerical.judge_check(skipped,
                                 expected_collected=311,
                                 allowed_skips=4) == "unknown"
    partial = numerical.classify(0, b"12 passed in 1s\n")
    assert numerical.judge_check(partial, expected_collected=32) == "unknown"
    failed = numerical.classify(1, b"16 failed, 16 passed in 2s\n")
    assert numerical.judge_check(failed, expected_collected=32) == "fail"


@pytest.fixture
def restricted_inputs(tmp_path):
    source, repository, checks = (tmp_path / name
                                  for name in ('source', 'captured', 'checks'))
    files = {
        'tests/python/conftest.py': '# Frozen public pytest configuration.\n',
        'tests/python/test_layout.py': '# Frozen public regression test.\n',
        'docs/sphinx/examples/python/example.py': '# Frozen public example.\n',
        'python/cudaq_algorithms/__init__.py': 'ORIGIN = "reference"\n',
        'pyproject.toml': '[project]\nname = "fixture"\nversion = "0"\n',
    }
    for relative, text in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        path.chmod(0o600)
    for path in [source, *(p for p in source.rglob('*') if p.is_dir())]:
        path.chmod(0o700)
    package = repository / 'python/cudaq_algorithms'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('ORIGIN = "captured"\n')
    for path in (repository, repository / 'python', package):
        path.chmod(0o755)
    (package / '__init__.py').chmod(0o644)
    checks.mkdir()
    (checks /
     'test_protocol_action.py').write_text('# Frozen targeted oracle.\n')
    return source, repository, checks, tmp_path / 'result'


def tree_state(root):
    return {
        str(path.relative_to(root)):
        (stat.S_IMODE(path.stat().st_mode),
         path.read_bytes() if path.is_file() else None)
        for path in (root, *sorted(root.rglob('*')))
    }


def mounts(argv):
    return {
        parts['target']: Path(parts['source'])
        for parts in (dict(
            field.split('=', 1) for field in argv[index + 1].split(',')
            if '=' in field) for index, value in enumerate(argv)
                      if value == '--mount')
    }


def test_run_checks_projects_restricted_source_without_changing_it(
        numerical, restricted_inputs, monkeypatch):
    source, repository, checks, output = restricted_inputs
    before = tree_state(source)
    observed = []

    def docker(argv, **kwargs):
        observed.append(argv)
        selected = mounts(argv)
        projected = selected['/frozen-source']
        assert projected != source, 'UID 65534 cannot traverse the original 0700 frozen source'
        assert projected == output / 'frozen-source'
        assert stat.S_IMODE(output.stat().st_mode) == 0o700
        for path in (projected, *projected.rglob('*')):
            assert stat.S_IMODE(
                path.stat().st_mode) == (0o755 if path.is_dir() else 0o644)
        assert {
            name: data
            for name, (_, data) in tree_state(projected).items()
            if data is not None
        } == {
            name: data
            for name, (_, data) in before.items() if data is not None
        }
        assert selected['/workspace/project'] == repository
        assert 'PYTHONPATH=/workspace/project/python' in argv
        assert (projected /
                'tests/python/test_layout.py').parents[2] == projected
        assert (projected / 'docs/sphinx/examples/python/example.py').is_file()
        freeze = json.loads(
            (output / 'frozen-source-projection.json').read_text())
        assert freeze['source_files'] == {
            name: hashlib.sha256(data).hexdigest()
            for name, (_, data) in before.items() if data is not None
        }
        assert freeze['source_inventory_sha256'] == freeze[
            'projection_inventory_sha256']
        return subprocess.CompletedProcess(argv, 0,
                                           b'307 passed, 4 skipped in 0.1s\n',
                                           b'')

    monkeypatch.setattr(numerical.subprocess, 'run', docker)
    previous_umask = os.umask(0o077)
    try:
        result = numerical.run_checks(repository,
                                      source / 'tests/python',
                                      checks,
                                      output,
                                      kind='regression')
    finally:
        os.umask(previous_umask)
    assert len(observed) == 1
    assert tree_state(source) == before
    projection = result['frozen_source_projection']
    assert projection['source_unchanged'] and projection['projection_unchanged']
    assert projection['manifest_sha256'] == hashlib.sha256(
        (output / 'frozen-source-projection.json').read_bytes()).hexdigest()
    assert numerical.judge_check(result,
                                 expected_collected=311,
                                 allowed_skips=4) == 'pass'


@pytest.mark.skipif(os.geteuid() != 0,
                    reason='actual UID 65534 traversal probe requires root')
def test_unprivileged_checker_reads_projection_but_imports_captured_package(
        numerical, restricted_inputs, monkeypatch):
    source, repository, checks, output = restricted_inputs
    actual_run = subprocess.run

    def docker(argv, **kwargs):
        selected = mounts(argv)
        source_fd = os.open(selected['/frozen-source'],
                            os.O_RDONLY | os.O_DIRECTORY)
        original_fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY)
        worker_fd = os.open(selected['/workspace/project'],
                            os.O_RDONLY | os.O_DIRECTORY)
        try:
            # The inherited directory descriptor emulates traversal beginning at
            # a bind-mount root, without weakening private host ancestors.
            code = (
                'import os,pathlib,sys; os.fchdir(int(sys.argv[1])); '
                'assert pathlib.Path("tests/python/conftest.py").read_text(); '
                'assert pathlib.Path("docs/sphinx/examples/python/example.py").read_text(); '
                'sys.path.insert(0,"/proc/self/fd/"+sys.argv[2]+"/python"); '
                'import cudaq_algorithms; assert cudaq_algorithms.ORIGIN == "captured"'
            )
            denied = actual_run([
                '/usr/bin/python3', '-B', '-c', code,
                str(original_fd),
                str(worker_fd)
            ],
                                cwd='/',
                                user=65534,
                                group=65534,
                                extra_groups=[],
                                pass_fds=(original_fd, worker_fd),
                                capture_output=True,
                                timeout=10)
            assert denied.returncode != 0 and b'PermissionError' in denied.stderr
            probe = actual_run([
                '/usr/bin/python3', '-B', '-c', code,
                str(source_fd),
                str(worker_fd)
            ],
                               cwd='/',
                               user=65534,
                               group=65534,
                               extra_groups=[],
                               pass_fds=(source_fd, worker_fd),
                               capture_output=True,
                               timeout=10)
            assert probe.returncode == 0, probe.stderr.decode()
        finally:
            os.close(source_fd)
            os.close(original_fd)
            os.close(worker_fd)
        return subprocess.CompletedProcess(argv, 0,
                                           b'307 passed, 4 skipped in 0.1s\n',
                                           b'')

    monkeypatch.setattr(numerical.subprocess, 'run', docker)
    numerical.run_checks(repository,
                         source / 'tests/python',
                         checks,
                         output,
                         kind='regression')


@pytest.mark.parametrize('unsafe', [
    'file_symlink', 'directory_symlink', 'fifo', 'hardlink', 'evals', 'skill'
])
def test_projection_rejects_unsafe_or_private_source_without_running_docker(
        numerical, restricted_inputs, monkeypatch, unsafe):
    source, repository, checks, output = restricted_inputs
    entry = source / 'unsafe'
    if unsafe == 'file_symlink':
        entry.symlink_to(source / 'tests/python/conftest.py')
    elif unsafe == 'directory_symlink':
        entry.symlink_to(source / 'tests', target_is_directory=True)
    elif unsafe == 'fifo':
        os.mkfifo(entry)
    elif unsafe == 'hardlink':
        os.link(source / 'tests/python/conftest.py', entry)
    elif unsafe == 'evals':
        (source / 'evals').mkdir()
        (source / 'evals/private.json').write_text('{}')
    else:
        (source / 'SKILL.md').write_text('private treatment')

    def unexpected(*args, **kwargs):
        pytest.fail('Unsafe projection must not invoke Docker')

    monkeypatch.setattr(numerical.subprocess, 'run', unexpected)
    with pytest.raises((ValueError, OSError)):
        numerical.run_checks(repository,
                             source / 'tests/python',
                             checks,
                             output,
                             kind='regression')


def test_existing_output_refused_without_changing_inputs_or_saved_files(
        numerical, restricted_inputs, monkeypatch):
    source, repository, checks, output = restricted_inputs
    output.mkdir(mode=0o700)
    (output / 'keep.json').write_text('old result')
    before, saved = tree_state(source), tree_state(output)

    def unexpected(*args, **kwargs):
        pytest.fail('Existing output must not invoke Docker')

    monkeypatch.setattr(numerical.subprocess, 'run', unexpected)
    with pytest.raises(FileExistsError):
        numerical.run_checks(repository,
                             source / 'tests/python',
                             checks,
                             output,
                             kind='regression')
    assert tree_state(source) == before and tree_state(output) == saved


def test_git_and_python_caches_never_enter_readable_projection(
        numerical, restricted_inputs, monkeypatch):
    source, repository, checks, output = restricted_inputs
    (source / '.git').mkdir()
    (source / '.git/config').write_text('private metadata not copied')
    (source / '__pycache__').mkdir()
    (source / '__pycache__/cache.pyc').write_bytes(b'cache')

    def docker(argv, **kwargs):
        projected = mounts(argv)['/frozen-source']
        assert not (projected / '.git').exists()
        assert not (projected / '__pycache__').exists()
        return subprocess.CompletedProcess(argv, 0, b'32 passed in 0.1s\n',
                                           b'')

    monkeypatch.setattr(numerical.subprocess, 'run', docker)
    result = numerical.run_checks(repository,
                                  source / 'tests/python',
                                  checks,
                                  output,
                                  kind='targeted')
    assert {
        item['path']
        for item in result['frozen_source_projection']['exclusions']
    } == {'.git', '__pycache__'}


def test_input_content_change_cannot_return_a_scientific_pass(
        numerical, restricted_inputs, monkeypatch):
    source, repository, checks, output = restricted_inputs

    def docker(argv, **kwargs):
        (source / 'tests/python/test_layout.py'
         ).write_text('# changed during the run\n')
        return subprocess.CompletedProcess(argv, 0,
                                           b'307 passed, 4 skipped in 0.1s\n',
                                           b'')

    monkeypatch.setattr(numerical.subprocess, 'run', docker)
    result = numerical.run_checks(repository,
                                  source / 'tests/python',
                                  checks,
                                  output,
                                  kind='regression')
    assert result['status'] == 'input_changed'
    assert result['frozen_source_projection']['source_unchanged'] is False
    assert numerical.judge_check(result,
                                 expected_collected=311,
                                 allowed_skips=4) == 'unknown'


@pytest.fixture
def scale_inputs(restricted_inputs, tmp_path):
    source, repository, checks, output = restricted_inputs
    for name in ('qubitization', 'qsvt', 'common_kernels', 'sim_utils',
                 'pauli_lcu', 'block_encoding'):
        for root in (source, repository):
            (root / f'python/cudaq_algorithms/{name}.py'
             ).write_text('# Frozen consumer.\n')
    (checks / 'test_scale_encoding.py').write_text('# Trusted E02 oracle.\n')
    binding = tmp_path / 'private/binding.json'
    binding.parent.mkdir()
    binding.write_text(
        json.dumps({
            'schema_version': 1,
            'kind': 'callable',
            'module': 'cudaq_algorithms.composition',
            'attribute': 'scale',
            'args': ['encoding', 'factor'],
            'kwargs': {},
            'zero_policy': 'encode'
        }))
    (binding.parent / 'credential').write_text('must never enter checker')
    return source, repository, checks, output, binding


def test_e02_selects_sixty_checks_and_mounts_only_frozen_binding(
        numerical, scale_inputs, monkeypatch):
    source, repository, checks, output, binding = scale_inputs
    before = binding.read_bytes()

    def docker(argv, **kwargs):
        selected = mounts(argv)
        assert '/checks/test_scale_encoding.py' in argv[-1]
        assert '/checks/test_protocol_action.py' not in argv[-1]
        assert 'E02_ARTIFACT_ROOT=/workspace/project' in argv
        assert 'E02_SCALE_ADAPTER=/e02-input/binding.json' in argv
        assert selected['/workspace/project'] == repository
        assert set(selected['/e02-input'].iterdir()) == {
            selected['/e02-input'] / 'binding.json'
        }
        assert (selected['/e02-input'] / 'binding.json').read_bytes() == before
        assert selected['/e02-input'] != binding.parent
        assert stat.S_IMODE(
            (selected['/e02-input'] / 'binding.json').stat().st_mode) == 0o644
        frozen = json.loads((output / 'e02-inputs.json').read_text())
        assert frozen['binding_sha256'] == hashlib.sha256(before).hexdigest()
        assert frozen['captured_inventory_sha256']
        assert frozen['consumer_files']['python/cudaq_algorithms/qsvt.py']
        return subprocess.CompletedProcess(argv, 0, b'60 passed in 1s\n', b'')

    monkeypatch.setattr(numerical.subprocess, 'run', docker)
    result = numerical.run_checks(repository,
                                  source / 'tests/python',
                                  checks,
                                  output,
                                  kind='targeted',
                                  case_id='E02',
                                  binding_path=binding)
    assert result['case_id'] == 'E02'
    assert result['e02_inputs']['unchanged'] is True
    assert result['e02_inputs']['manifest_sha256'] == hashlib.sha256(
        (output / 'e02-inputs.json').read_bytes()).hexdigest()
    assert binding.read_bytes() == before
    assert numerical.judge_check(result, expected_collected=60) == 'pass'


@pytest.mark.parametrize('invalid', [
    'missing_binding', 'changed_consumer', 'missing_consumer',
    'binding_symlink'
])
def test_e02_unverified_binding_or_consumer_never_runs(numerical, scale_inputs,
                                                       monkeypatch, invalid):
    source, repository, checks, output, binding = scale_inputs
    if invalid == 'missing_binding':
        binding = None
    elif invalid == 'changed_consumer':
        (repository / 'python/cudaq_algorithms/qsvt.py'
         ).write_text('# Changed worker consumer.\n')
    elif invalid == 'missing_consumer':
        (repository / 'python/cudaq_algorithms/qsvt.py').unlink()
    else:
        alias = binding.parent / 'alias.json'
        alias.symlink_to(binding)
        binding = alias

    def unexpected(*args, **kwargs):
        pytest.fail('Unverified E02 inputs must not execute Docker')

    monkeypatch.setattr(numerical.subprocess, 'run', unexpected)
    with pytest.raises((ValueError, OSError)):
        numerical.run_checks(repository,
                             source / 'tests/python',
                             checks,
                             output,
                             kind='targeted',
                             case_id='E02',
                             binding_path=binding)


@pytest.mark.parametrize(
    'changed', ['binding', 'captured', 'checks', 'staged_binding', 'freeze'])
def test_e02_changed_inputs_cannot_earn_pass(numerical, scale_inputs,
                                             monkeypatch, changed):
    source, repository, checks, output, binding = scale_inputs

    def docker(argv, **kwargs):
        paths = {
            'binding': binding,
            'captured': repository / 'python/cudaq_algorithms/__init__.py',
            'checks': checks / 'test_scale_encoding.py',
            'staged_binding': mounts(argv)['/e02-input'] / 'binding.json',
            'freeze': output / 'e02-inputs.json'
        }
        paths[changed].write_bytes(b'changed after freeze')
        return subprocess.CompletedProcess(argv, 0, b'60 passed in 1s\n', b'')

    monkeypatch.setattr(numerical.subprocess, 'run', docker)
    result = numerical.run_checks(repository,
                                  source / 'tests/python',
                                  checks,
                                  output,
                                  kind='targeted',
                                  case_id='E02',
                                  binding_path=binding)
    assert result['status'] == 'input_changed'
    assert numerical.judge_check(result, expected_collected=60) == 'unknown'


def test_e02_regression_does_not_require_or_mount_adapter(
        numerical, restricted_inputs, monkeypatch):
    source, repository, checks, output = restricted_inputs

    def docker(argv, **kwargs):
        assert '/frozen-source/tests/python' in argv[-1]
        assert not any('E02_' in part for part in argv)
        return subprocess.CompletedProcess(argv, 0,
                                           b'307 passed, 4 skipped in 1s\n',
                                           b'')

    monkeypatch.setattr(numerical.subprocess, 'run', docker)
    result = numerical.run_checks(repository,
                                  source / 'tests/python',
                                  checks,
                                  output,
                                  kind='regression',
                                  case_id='E02')
    assert numerical.judge_check(result,
                                 expected_collected=311,
                                 allowed_skips=4) == 'pass'


@pytest.fixture
def ci_inputs(scale_inputs):
    source, repository, checks, output, binding = scale_inputs
    (checks / 'test_ci_preparation.py'
     ).write_text('# Trusted worker-targeted E05 suite.\n')
    binding.write_text(
        json.dumps({
            'schema_version': 1,
            'module': 'cudaq_algorithms.ci',
            'attribute': 'prepare',
            'representation': 'basis_occupations',
            'coefficient_policy': 'normalize',
            'args': ['orbital_basis', 'occupations', 'coefficients'],
            'kwargs': {}
        }))
    return source, repository, checks, output, binding


def test_e05_freezes_artifact_binding_and_only_required_consumers(
        numerical, ci_inputs, monkeypatch):
    source, repository, checks, output, binding = ci_inputs
    before = binding.read_bytes()
    # Extensions may change exports and add preparation code; those are artifact
    # bytes, not frozen consumers. QSVT is not used by this E05 outcome check.
    (repository /
     'python/cudaq_algorithms/__init__.py').write_text('NEW_EXPORT = 1\n')
    (repository /
     'python/cudaq_algorithms/qsvt.py').write_text('# Unused by E05\n')

    def docker(argv, **kwargs):
        selected = mounts(argv)
        assert '/checks/test_ci_preparation.py' in argv[-1]
        assert 'test_ci_calibration.py' not in argv[-1]
        assert 'E05_CI_ADAPTER=/e05-input/binding.json' in argv
        assert 'E05_ARTIFACT_ROOT=/workspace/project' in argv
        assert not any('E02_' in value for value in argv)
        assert selected['/workspace/project'] == repository
        staged = selected['/e05-input']
        assert staged != binding.parent
        assert {path.name for path in staged.iterdir()} == {'binding.json'}
        assert (staged / 'binding.json').read_bytes() == before
        assert stat.S_IMODE((staged / 'binding.json').stat().st_mode) == 0o644
        frozen = json.loads((output / 'e05-inputs.json').read_text())
        assert frozen['binding_sha256'] == hashlib.sha256(before).hexdigest()
        assert set(frozen['consumer_files']) == {
            f'python/cudaq_algorithms/{name}.py'
            for name in ('qubitization', 'pauli_lcu', 'common_kernels',
                         'block_encoding')
        }
        assert frozen['captured_inventory_sha256'] and frozen[
            'checks_inventory_sha256']
        return subprocess.CompletedProcess(argv, 0, b'1 passed in 1s\n', b'')

    monkeypatch.setattr(numerical.subprocess, 'run', docker)
    result = numerical.run_checks(repository,
                                  source / 'tests/python',
                                  checks,
                                  output,
                                  kind='targeted',
                                  case_id='E05',
                                  binding_path=binding)
    assert result['case_id'] == 'E05'
    assert result['e05_inputs']['unchanged'] is True
    assert result['e05_inputs']['manifest_sha256'] == hashlib.sha256(
        (output / 'e05-inputs.json').read_bytes()).hexdigest()
    assert result['binding_sha256'] == hashlib.sha256(before).hexdigest()
    # An otherwise consistent but partial log cannot become a targeted pass.
    assert numerical.judge_check(result, expected_collected=2) == 'unknown'


@pytest.mark.parametrize('invalid', [
    'missing_binding', 'changed_consumer', 'missing_consumer',
    'binding_symlink'
])
def test_e05_unbound_inputs_never_execute(numerical, ci_inputs, monkeypatch,
                                          invalid):
    source, repository, checks, output, binding = ci_inputs
    if invalid == 'missing_binding':
        binding = None
    elif invalid == 'changed_consumer':
        (repository / 'python/cudaq_algorithms/qubitization.py'
         ).write_text('# Changed consumer\n')
    elif invalid == 'missing_consumer':
        (repository / 'python/cudaq_algorithms/block_encoding.py').unlink()
    else:
        alias = binding.parent / 'alias.json'
        alias.symlink_to(binding)
        binding = alias

    def unexpected(*args, **kwargs):
        pytest.fail('Unverified E05 inputs must not execute Docker')

    monkeypatch.setattr(numerical.subprocess, 'run', unexpected)
    with pytest.raises((ValueError, OSError)):
        numerical.run_checks(repository,
                             source / 'tests/python',
                             checks,
                             output,
                             kind='targeted',
                             case_id='E05',
                             binding_path=binding)
    assert not output.exists()


@pytest.mark.parametrize(
    'changed', ['binding', 'captured', 'checks', 'staged_binding', 'freeze'])
def test_e05_changed_inputs_cannot_earn_pass(numerical, ci_inputs, monkeypatch,
                                             changed):
    source, repository, checks, output, binding = ci_inputs

    def docker(argv, **kwargs):
        paths = {
            'binding': binding,
            'captured': repository / 'python/cudaq_algorithms/__init__.py',
            'checks': checks / 'test_ci_preparation.py',
            'staged_binding': mounts(argv)['/e05-input'] / 'binding.json',
            'freeze': output / 'e05-inputs.json'
        }
        paths[changed].write_bytes(b'changed after freeze')
        return subprocess.CompletedProcess(argv, 0, b'1 passed in 1s\n', b'')

    monkeypatch.setattr(numerical.subprocess, 'run', docker)
    result = numerical.run_checks(repository,
                                  source / 'tests/python',
                                  checks,
                                  output,
                                  kind='targeted',
                                  case_id='E05',
                                  binding_path=binding)
    assert result['status'] == 'input_changed'
    assert numerical.judge_check(result, expected_collected=1) == 'unknown'


def test_e05_original_regression_is_separate_from_binding(
        numerical, restricted_inputs, monkeypatch):
    source, repository, checks, output = restricted_inputs

    def docker(argv, **kwargs):
        assert '/frozen-source/tests/python' in argv[-1]
        assert not any('E05_' in value for value in argv)
        assert '/e05-input' not in mounts(argv)
        return subprocess.CompletedProcess(argv, 0,
                                           b'307 passed, 4 skipped in 1s\n',
                                           b'')

    monkeypatch.setattr(numerical.subprocess, 'run', docker)
    result = numerical.run_checks(repository,
                                  source / 'tests/python',
                                  checks,
                                  output,
                                  kind='regression',
                                  case_id='E05')
    assert numerical.judge_check(result,
                                 expected_collected=311,
                                 allowed_skips=4) == 'pass'


def test_e05_unsupported_layout_skip_is_unknown_but_numerical_failure_fails(
        numerical):
    skipped = numerical.classify(0, b'1 passed, 1 skipped in 1s\n')
    assert numerical.judge_check(skipped, expected_collected=2) == 'unknown'
    failed = numerical.classify(1, b'1 failed, 1 passed in 1s\n')
    assert numerical.judge_check(failed, expected_collected=2) == 'fail'


@pytest.fixture
def evolution_inputs(scale_inputs):
    source, repository, checks, output, binding = scale_inputs
    (checks / 'test_evolution_example.py'
     ).write_text('# Trusted worker-targeted E08 suite.\n')
    binding.write_text(
        json.dumps({
            'schema_version': 1,
            'build': {
                'kind': 'callable',
                'module': 'cudaq_algorithms.evolution',
                'attribute': 'build',
                'args': ['terms', 'time', 'state_prep'],
                'kwargs': {}
            },
            'recover': {
                'kind': 'callable',
                'module': 'cudaq_algorithms.evolution',
                'attribute': 'recover',
                'args': ['cos_state', 'sin_state'],
                'kwargs': {}
            },
            'validate': {
                'kind': 'callable',
                'module': 'cudaq_algorithms.evolution',
                'attribute': 'validate',
                'args': ['actual_state', 'initial_state'],
                'kwargs': {}
            },
            'kernels': {
                'kind': 'mapping',
                'cosine': 'cosine',
                'sine': 'sine'
            },
            'domain': {
                'hamiltonian': 'complex',
                'initial_state': 'complex'
            },
            'phase_policy': 'exact',
            'recovery_semantics': 'real_linear',
            'validation_mode': 'raises',
        }))
    return source, repository, checks, output, binding


def test_e08_registers_twelve_checks_and_freezes_exact_binding_inputs(
        numerical, evolution_inputs, monkeypatch):
    source, repository, checks, output, binding = evolution_inputs
    before = binding.read_bytes()
    expected_consumers = [
        f'python/cudaq_algorithms/{name}.py'
        for name in ('qubitization', 'qsvt', 'common_kernels', 'sim_utils',
                     'pauli_lcu', 'block_encoding')
    ]
    # Worker-authored exports and example code remain part of the captured
    # artifact; only the established public consumers are frozen unchanged.
    (repository / 'python/cudaq_algorithms/__init__.py'
     ).write_text('EVOLUTION_EXPORT = 1\n')

    def docker(argv, **kwargs):
        selected = mounts(argv)
        assert '/checks/test_evolution_example.py' in argv[-1]
        assert 'test_evolution_runtime.py' not in argv[-1]
        assert 'E08_EVOLUTION_ADAPTER=/e08-input/binding.json' in argv
        assert 'E08_ARTIFACT_ROOT=/workspace/project' in argv
        assert not any('E02_' in value or 'E05_' in value for value in argv)
        assert selected['/workspace/project'] == repository
        staged = selected['/e08-input']
        assert staged != binding.parent
        assert list(staged.iterdir()) == [staged / 'binding.json']
        assert (staged / 'binding.json').read_bytes() == before
        assert stat.S_IMODE((staged / 'binding.json').stat().st_mode) == 0o644
        frozen = json.loads((output / 'e08-inputs.json').read_text())
        assert frozen['binding_sha256'] == hashlib.sha256(before).hexdigest()
        # Canonical JSON sorts mapping keys; the registration tuple below
        # preserves the required consumer order before serialization.
        assert set(frozen['consumer_files']) == set(expected_consumers)
        assert frozen['captured_inventory_sha256']
        assert frozen['checks_inventory_sha256']
        return subprocess.CompletedProcess(argv, 0, b'12 passed in 1s\n', b'')

    monkeypatch.setattr(numerical.subprocess, 'run', docker)
    result = numerical.run_checks(repository,
                                  source / 'tests/python',
                                  checks,
                                  output,
                                  kind='targeted',
                                  case_id='E08',
                                  binding_path=binding)

    assert numerical.CHECK_SUITES['E08'] == {
        'targeted': (12, 0),
        'regression': (311, 4)
    }
    assert numerical.E08_CONSUMERS == tuple(expected_consumers)
    assert result['case_id'] == 'E08'
    assert result['e08_inputs']['unchanged'] is True
    assert result['e08_inputs']['manifest_sha256'] == hashlib.sha256(
        (output / 'e08-inputs.json').read_bytes()).hexdigest()
    assert result['binding_sha256'] == hashlib.sha256(before).hexdigest()
    assert binding.read_bytes() == before
    assert numerical.judge_check(result, expected_collected=12) == 'pass'


def test_e08_rereads_staged_binding_before_checker_execution(
        numerical, evolution_inputs, monkeypatch):
    source, repository, checks, output, binding = evolution_inputs
    real_write = numerical.capture._write

    def corrupt_staged_binding(directory, name, content):
        real_write(directory, name, content)
        if name == 'binding.json':
            descriptor = os.open(name,
                                 os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW,
                                 dir_fd=directory)
            try:
                os.write(descriptor, b'changed while staging')
            finally:
                os.close(descriptor)

    def unexpected(*args, **kwargs):
        pytest.fail('Changed staged E08 binding must not execute Docker')

    monkeypatch.setattr(numerical.capture, '_write', corrupt_staged_binding)
    monkeypatch.setattr(numerical.subprocess, 'run', unexpected)
    with pytest.raises(ValueError, match='staged binding'):
        numerical.run_checks(repository,
                             source / 'tests/python',
                             checks,
                             output,
                             kind='targeted',
                             case_id='E08',
                             binding_path=binding)


@pytest.mark.parametrize('invalid', [
    'missing_binding', 'changed_consumer', 'missing_consumer',
    'binding_symlink'
])
def test_e08_unbound_or_changed_sources_never_execute(numerical,
                                                      evolution_inputs,
                                                      monkeypatch, invalid):
    source, repository, checks, output, binding = evolution_inputs
    if invalid == 'missing_binding':
        binding = None
    elif invalid == 'changed_consumer':
        (repository / 'python/cudaq_algorithms/sim_utils.py'
         ).write_text('# Changed E08 consumer.\n')
    elif invalid == 'missing_consumer':
        (repository / 'python/cudaq_algorithms/qsvt.py').unlink()
    else:
        alias = binding.parent / 'alias.json'
        alias.symlink_to(binding)
        binding = alias

    def unexpected(*args, **kwargs):
        pytest.fail('Unverified E08 inputs must not execute Docker')

    monkeypatch.setattr(numerical.subprocess, 'run', unexpected)
    with pytest.raises((ValueError, OSError)):
        numerical.run_checks(repository,
                             source / 'tests/python',
                             checks,
                             output,
                             kind='targeted',
                             case_id='E08',
                             binding_path=binding)
    assert not output.exists()


@pytest.mark.parametrize(
    'changed', ['binding', 'captured', 'checks', 'staged_binding', 'freeze'])
def test_e08_post_execution_mutation_cannot_earn_pass(numerical,
                                                      evolution_inputs,
                                                      monkeypatch, changed):
    source, repository, checks, output, binding = evolution_inputs

    def docker(argv, **kwargs):
        paths = {
            'binding': binding,
            'captured': repository / 'python/cudaq_algorithms/qsvt.py',
            'checks': checks / 'test_evolution_example.py',
            'staged_binding': mounts(argv)['/e08-input'] / 'binding.json',
            'freeze': output / 'e08-inputs.json',
        }
        paths[changed].write_bytes(b'changed after freeze')
        return subprocess.CompletedProcess(argv, 0, b'12 passed in 1s\n', b'')

    monkeypatch.setattr(numerical.subprocess, 'run', docker)
    result = numerical.run_checks(repository,
                                  source / 'tests/python',
                                  checks,
                                  output,
                                  kind='targeted',
                                  case_id='E08',
                                  binding_path=binding)
    assert result['status'] == 'input_changed'
    assert numerical.judge_check(result, expected_collected=12) == 'unknown'


def test_e08_regression_forbids_binding_and_uses_original_suite(
        numerical, evolution_inputs, monkeypatch, tmp_path):
    source, repository, checks, output, binding = evolution_inputs

    def docker(argv, **kwargs):
        assert '/frozen-source/tests/python' in argv[-1]
        assert not any('E08_' in value for value in argv)
        assert '/e08-input' not in mounts(argv)
        return subprocess.CompletedProcess(argv, 0,
                                           b'307 passed, 4 skipped in 1s\n',
                                           b'')

    monkeypatch.setattr(numerical.subprocess, 'run', docker)
    with pytest.raises(ValueError, match='forbid'):
        numerical.run_checks(repository,
                             source / 'tests/python',
                             checks,
                             tmp_path / 'refused',
                             kind='regression',
                             case_id='E08',
                             binding_path=binding)
    assert not (tmp_path / 'refused').exists()

    result = numerical.run_checks(repository,
                                  source / 'tests/python',
                                  checks,
                                  output,
                                  kind='regression',
                                  case_id='E08')
    assert numerical.judge_check(result,
                                 expected_collected=311,
                                 allowed_skips=4) == 'pass'


def test_e08_skip_is_unknown_and_numerical_failure_is_fail(numerical):
    skipped = numerical.classify(0, b'8 passed, 4 skipped in 1s\n')
    assert numerical.judge_check(skipped, expected_collected=12) == 'unknown'
    failed = numerical.classify(1, b'1 failed, 11 passed in 1s\n')
    assert numerical.judge_check(failed, expected_collected=12) == 'fail'
