# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real process cleanup and namespace script wiring, without model providers."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runner import workspace


def running(pid):
    try:
        # A killed orphan can remain a zombie until the container init reaps it;
        # it cannot run code or perform the delayed write.
        return Path(f'/proc/{pid}/stat').read_text().rsplit(
            ')', 1)[1].split()[0] not in ('Z', 'X')
    except FileNotFoundError:
        return False


def wait_until(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() >= deadline:
            return False
        time.sleep(.02)
    return True


def cleanup_pids(worker):
    for path in worker.glob('*.pid'):
        pid = int(path.read_text())
        if running(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_continuing_output_producer_is_terminated_and_descendants_cannot_write(
        tmp_path):
    worker = tmp_path / 'workspace'
    worker.mkdir()
    script = '''import os, pathlib, time
pathlib.Path('producer.pid').write_text(str(os.getpid()))
child = os.fork()
if child == 0:
    pathlib.Path('child.pid').write_text(str(os.getpid()))
    os.close(1); os.close(2)
    time.sleep(20)
    pathlib.Path('late.txt').write_text('leaked')
    os._exit(0)
while not pathlib.Path('child.pid').exists():
    time.sleep(.01)
while True:
    os.write(1, b'x' * 4096)
    os.write(2, b'y' * 4096)
    time.sleep(.001)
'''
    (worker / 'producer.py').write_text(script)
    try:
        result = workspace.run_command('python3 producer.py',
                                       worker,
                                       timeout=15,
                                       isolation='trusted',
                                       log_dir=tmp_path / 'logs',
                                       max_output_bytes=1024)
        assert result['output_limit_exceeded']
        assert result['termination_reason'] == 'output_limit'
        assert not result['timed_out']
        pids = [
            int((worker / name).read_text())
            for name in ('producer.pid', 'child.pid')
        ]
        assert wait_until(lambda: all(not running(pid) for pid in pids))
        assert not (worker / 'late.txt').exists()
        assert len(result['stdout']) <= 1024 and len(result['stderr']) <= 1024
        assert Path(result['stdout_log']).stat().st_size > 1024
        assert Path(result['stderr_log']).stat().st_size > 1024
    finally:
        cleanup_pids(worker)


@pytest.mark.parametrize('exists,fail_mount', [(True, False), (False, False),
                                               (True, True)])
def test_namespace_alternatives_bind_is_optional_readonly_and_fail_closed(
        tmp_path, exists, fail_mount):
    # Execute the real shell setup, replacing privileged OS commands only.
    # The directory test is controlled independently of the host distribution.
    jail, worker = [tmp_path / name for name in ('jail', 'workspace')]
    for directory in (jail, worker):
        directory.mkdir()
    (worker / '.tmp').mkdir()
    marker = tmp_path / 'entered-worker'
    log = tmp_path / 'mounts'
    harness = '''
exec() { printf '%s\\n' \"$*\" > \"$EXEC_LOG\"; }
function [() {
    if [[ "$1" == '-d' && "$2" == '/etc/alternatives' ]]; then
        return "$ALTERNATIVES_ABSENT"
    fi
    builtin [ "$@"
}
mount() {
    printf '%s\\n' "$*" >> "$MOUNT_LOG"
    if [[ "$FAIL_ALTERNATIVES" == 1 && "$*" == *'/etc/alternatives'* ]]; then return 32; fi
}
'''
    result = subprocess.run(
        [
            '/bin/bash', '-c', harness + workspace._NAMESPACE_SCRIPT, 'test',
            str(jail),
            str(worker), '', 'true'
        ],
        env={
            **os.environ, 'EXEC_LOG': str(marker),
            'ALTERNATIVES_ABSENT': str(int(not exists)),
            'FAIL_ALTERNATIVES': str(int(fail_mount)),
            'MOUNT_LOG': str(log)
        },
        capture_output=True,
        text=True,
        timeout=10)
    lines = log.read_text().splitlines()
    if exists:
        assert (jail / 'etc/alternatives').is_dir()
        assert f'--bind /etc/alternatives {jail}/etc/alternatives' in lines
        if not fail_mount:
            assert f'-o remount,bind,ro,nosuid,nodev {jail}/etc/alternatives' in lines
    else:
        assert not (jail / 'etc/alternatives').exists()
        assert not any('/etc/alternatives' in line for line in lines)
    assert (result.returncode != 0) is fail_mount, result.stderr
    assert marker.exists() is not fail_mount


@pytest.mark.skipif(os.environ.get('CUDAQ_RUNNER_TEST_ISOLATION') != '1',
                    reason='requires approved Linux namespace capability')
def test_real_namespace_runs_alternative_backed_awk_readonly(tmp_path):
    if not Path('/etc/alternatives/awk').is_symlink():
        pytest.skip('host does not supply an alternative-backed awk')
    assert '/etc/alternatives' in os.readlink('/usr/bin/awk')
    result = workspace.run_command(
        "awk 'BEGIN { print 6 * 7 }' && "
        "if probe=$(mktemp /etc/alternatives/cudaq-probe.XXXXXXXX 2>/dev/null); "
        "then rm -- \"$probe\"; exit 7; fi",
        tmp_path,
        timeout=15)
    assert result['exit_code'] == 0, result['stderr']
    assert result['stdout'] == '42\n'
    assert result['isolated']
