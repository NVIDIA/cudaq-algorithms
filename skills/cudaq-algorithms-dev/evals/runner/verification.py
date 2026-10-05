"""Execute artifacts in isolation, then compare outputs with private references.

The candidate never imports a reference checker. A separate clean subprocess
runs the trusted numerical code without the candidate's Python search path.
"""

import json
import hashlib
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid

from . import checkers, workspace

MAX_PAYLOAD = 16 * 1024 * 1024


def preflight(runtime_python):
    """Check private-reference dependencies before consuming provider requests."""
    python = str(runtime_python or sys.executable)
    process = subprocess.run([
        python, '-I', '-c',
        'import numpy,scipy,pyscf; from pyscf import gto,scf,fci; print("reference dependencies available")'
    ],
                             capture_output=True,
                             text=True,
                             timeout=30)
    if process.returncode:
        raise ValueError(
            'Private references require numpy, scipy and pyscf in the configured runtime'
        )


def output_payload(stdout):
    lines = stdout.strip().splitlines()
    if not lines or len(lines[-1].encode()) > MAX_PAYLOAD:
        raise ValueError('Missing or oversized final JSON output')

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Duplicate JSON field')
            value[key] = item
        return value

    def invalid(value):
        raise ValueError('Nonfinite JSON number')

    try:
        value = json.loads(lines[-1],
                           object_pairs_hook=unique,
                           parse_constant=invalid)
    except RecursionError as exc:
        raise ValueError('Artifact JSON nesting is excessive') from exc
    if not isinstance(value, dict):
        raise ValueError('Artifact output must be a JSON object')
    return value


def file_inventory(root):
    """Hash authored files without following symlinks or reading generated caches."""
    result = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [
            d for d in dirs
            if d not in ('.tmp', '__pycache__', '.pytest_cache')
        ]
        for name in dirs + names:
            path = Path(directory) / name
            relative = str(path.relative_to(root))
            if path.is_symlink():
                result[relative] = 'symlink'
            elif path.is_file():
                digest = hashlib.sha256()
                with path.open('rb') as handle:
                    for block in iter(lambda: handle.read(65536), b''):
                        digest.update(block)
                result[relative] = digest.hexdigest()
    return result


def check_scope(case_id, before, after):
    allowed = {
        'source-version-drift': 'qsvt_client.py',
        'implementation-scope-overreach': 'pauli_lcu_example.py'
    }
    if case_id not in allowed:
        return
    for path in set(before) | set(after):
        if before.get(path) == after.get(path) or path == allowed[case_id]:
            continue
        # The drift prompt explicitly permits unchanged copies of the fixtures.
        if (case_id == 'source-version-drift' and path not in before and
                after.get(path) == before.get('.eval-inputs/files/' + path)):
            continue
        raise ValueError(f'Change outside authorized scope: {path}')


def instructions(case_id):
    contract = checkers.public_contract(case_id)
    if contract is None:
        return ''
    header = (
        ' Executable verification: preserve a runnable artifact at the '
        'workspace root before your final answer. The harness executes it '
        'after you finish, within the same task budget. ')
    if case_id.startswith('science-'):
        header += (
            'Save evaluation.py; running python evaluation.py must reproduce '
            'your calculation and print one JSON object on its last stdout '
            'line with the following output fields. Serialize complex '
            'numbers as [real, imaginary]; arrays follow the stated shapes. '
            'Statevectors use q0 as the least-significant bit unless the '
            'contract explicitly permits another declared ordering. '
            'Use enough numerical precision to support your accuracy claims. '
            'These artifact fields are for verification of the requested '
            'calculation; explain your method and conclusions in the final answer. '
        )
    return header + json.dumps(contract, ensure_ascii=False)


def _oracle(case_id, payload, runtime_python, timeout):
    source = str(Path(__file__).resolve().parents[1])
    program = '''import sys,json
sys.path.insert(0, sys.argv[1])
from runner.checkers import check
try:
    check(sys.argv[2],json.load(sys.stdin))
except Exception as exc:
    print(json.dumps({'passed':False,'error':type(exc).__name__,'message':str(exc)}))
    raise SystemExit(1)
print(json.dumps({'passed':True}))
'''
    environment = {
        'PATH': '/usr/bin:/bin',
        'LANG': 'C.UTF-8',
        'OMP_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'PYTHONDONTWRITEBYTECODE': '1'
    }
    process = subprocess.Popen([
        str(runtime_python or sys.executable), '-I', '-c', program, source,
        case_id
    ],
                               cwd=source,
                               env=environment,
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True,
                               start_new_session=True)
    try:
        stdout, stderr = process.communicate(json.dumps(payload,
                                                        allow_nan=False),
                                             timeout=timeout)
    except BaseException:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise
    result = output_payload(stdout)
    result.update(exit_code=process.returncode, stderr=stderr)
    return result


def execute(case_id,
            worker,
            attempt,
            *,
            runtime_python,
            isolation,
            deadline,
            initial_files=None):
    """Run once at finalization; no retrospective completion time inference."""
    started = time.monotonic()
    detail = {
        'case_id': case_id,
        'policy': 'after_final_answer_v1',
        'target': 'qpp-cpu',
        'precision': 'fp64'
    }
    status = 'not_run'
    if deadline <= started:
        detail[
            'reason'] = 'Task budget exhausted before independent verification'
    else:
        try:
            if initial_files is not None:
                check_scope(case_id, initial_files, file_inventory(worker))
            module = checkers.module_for(case_id)
            if module is None:
                raise ValueError('Missing preregistered checker')
            filenames = {
                'repository-implementation-third-moment': 'starter_app.py',
                'implementation-scope-overreach': 'pauli_lcu_example.py',
                'source-version-drift': 'qsvt_client.py'
            }
            artifact = worker / filenames.get(case_id, 'evaluation.py')
            workspace._real_path(artifact)
            if not artifact.is_file():
                raise ValueError(f'Missing runnable artifact: {artifact.name}')
            python = shlex.quote(str(runtime_python or '/usr/bin/python3'))
            if case_id.startswith('science-'):
                command = python + ' ' + shlex.quote(artifact.name)
            else:
                from .checkers.regression import probe
                temporary = workspace._real_path(worker / '.tmp')
                temporary.mkdir(exist_ok=True)
                probe_path = temporary / ('verification-' + uuid.uuid4().hex +
                                          '.py')
                with probe_path.open('x') as handle:
                    handle.write(probe(case_id))
                command = python + ' ' + shlex.quote(
                    str(probe_path.relative_to(worker)))
            detail['artifact'] = artifact.name
            detail['artifact_sha256'] = hashlib.sha256(
                artifact.read_bytes()).hexdigest()
            detail['artifact_execution'] = workspace.run_command(
                command,
                worker,
                timeout=max(.001, deadline - time.monotonic()),
                runtime_python=runtime_python,
                isolation=isolation,
                log_dir=attempt / 'verification-logs',
                max_output_bytes=MAX_PAYLOAD)
            observed = detail['artifact_execution']
            if observed.get('output_limit_exceeded'):
                raise ValueError(
                    'Artifact output exceeded verification size limit')
            if observed['timed_out']:
                raise ValueError(
                    'Artifact execution exceeded remaining task budget')
            if observed['exit_code']:
                raise ValueError('Artifact execution failed')
            if initial_files is not None:
                check_scope(case_id, initial_files, file_inventory(worker))
            payload = output_payload(observed['stdout'])
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                detail[
                    'reason'] = 'No budget remains for private reference comparison'
            else:
                detail['oracle'] = _oracle(case_id, payload, runtime_python,
                                           remaining)
                status = 'passed' if detail['oracle']['passed'] else 'failed'
                # Missing trusted dependencies or broken checker code are infrastructure
                # failures, not evidence that a scientific answer is incorrect.
                if detail['oracle'].get('error') not in (None, 'CheckFailure'):
                    status = 'not_run'
        except subprocess.TimeoutExpired:
            detail[
                'reason'] = 'Private reference comparison exceeded task budget'
        except workspace.IsolationUnavailable as exc:
            detail['reason'] = str(exc)
        except (ValueError, OSError, KeyError) as exc:
            detail['reason'] = str(exc)
            status = 'failed'
    detail.update(status=status, seconds=time.monotonic() - started)
    name = 'verification-observation-' + uuid.uuid4().hex + '.json'
    (attempt /
     name).write_text(json.dumps(detail, indent=2, allow_nan=False) + '\n')
    return {
        'status': status,
        'evidence_file': name,
        'seconds': detail['seconds']
    }
