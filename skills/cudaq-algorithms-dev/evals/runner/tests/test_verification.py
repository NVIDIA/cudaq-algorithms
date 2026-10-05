import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runner import verification, campaign


def test_public_contract_does_not_expose_references_or_change_prompt():
    case = {
        'id': 'repository-implementation-third-moment',
        'prompt': 'ORIGINAL',
        'files': []
    }
    messages = campaign.worker_messages(case, 'baseline', 'listed', '')
    assert messages[-1]['content'] == 'ORIGINAL'
    assert 'starter_app.py' in messages[0]['content']
    assert 'Dense T3' not in messages[0]['content']


def test_artifact_execution_and_private_validation_are_distinct(tmp_path):
    worker = tmp_path / 'workspace'
    worker.mkdir()
    (worker / '.tmp').mkdir()
    (worker / '.eval-inputs').mkdir()
    (worker / 'starter_app.py').write_text('''import numpy as np
def third_moment(ket):
    h=np.array([[.75,-.25],[-.25,-.75]])
    return np.vdot(ket,(4*h@h@h-3*h)@ket).real
''')
    result = verification.execute('repository-implementation-third-moment',
                                  worker,
                                  tmp_path,
                                  runtime_python=Path(sys.executable),
                                  isolation='trusted',
                                  deadline=time.monotonic() + 20)
    assert result['status'] == 'passed'
    evidence = json.loads((tmp_path / result['evidence_file']).read_text())
    assert evidence['artifact_execution']['exit_code'] == 0
    assert evidence['oracle']['passed'] is True
    (worker /
     'starter_app.py').write_text('def third_moment(ket): return 0.0\n')
    result = verification.execute('repository-implementation-third-moment',
                                  worker,
                                  tmp_path,
                                  runtime_python=Path(sys.executable),
                                  isolation='trusted',
                                  deadline=time.monotonic() + 20)
    assert result['status'] == 'failed'


def test_missing_artifact_expired_budget_and_symlink_never_pass(tmp_path):
    worker = tmp_path / 'workspace'
    worker.mkdir()
    kwargs = dict(runtime_python=Path(sys.executable), isolation='trusted')
    result = verification.execute('repository-implementation-third-moment',
                                  worker,
                                  tmp_path,
                                  deadline=time.monotonic() - 1,
                                  **kwargs)
    assert result['status'] == 'not_run'
    result = verification.execute('repository-implementation-third-moment',
                                  worker,
                                  tmp_path,
                                  deadline=time.monotonic() + 10,
                                  **kwargs)
    assert result['status'] == 'failed'
    (worker / 'starter_app.py').symlink_to('/etc/passwd')
    result = verification.execute('repository-implementation-third-moment',
                                  worker,
                                  tmp_path,
                                  deadline=time.monotonic() + 10,
                                  **kwargs)
    assert result['status'] == 'failed'


@pytest.mark.parametrize('text', ['{"x":NaN}', '{"x":1,"x":2}', '[1,2]', ''])
def test_bad_output_is_rejected(text):
    with pytest.raises(ValueError):
        verification.output_payload(text)


def test_deeply_nested_output_is_a_controlled_failure():
    with pytest.raises(ValueError):
        verification.output_payload('{"x":' + '[' * 10000 + '0' + ']' * 10000 +
                                    '}')


def test_scope_validation_rejects_changed_source_but_allows_client_fixture_copies(
        tmp_path):
    (tmp_path / 'python').mkdir()
    (tmp_path / 'python/library.py').write_text('original')
    fixtures = tmp_path / '.eval-inputs/files'
    fixtures.mkdir(parents=True)
    (fixtures / 'qsvt_current_contract.py').write_text('protected fixture')
    before = verification.file_inventory(tmp_path)
    (tmp_path / 'qsvt_client.py').write_text('updated client')
    (tmp_path / 'qsvt_current_contract.py').write_text('protected fixture')
    verification.check_scope('source-version-drift', before,
                             verification.file_inventory(tmp_path))
    (tmp_path / 'python/library.py').write_text('modified library')
    with pytest.raises(ValueError, match='scope'):
        verification.check_scope('source-version-drift', before,
                                 verification.file_inventory(tmp_path))


def test_attempt_records_measured_verification_and_hash_bindings(
        tmp_path, monkeypatch):
    from runner import workspace, providers, assessment
    worker = tmp_path / 'workspace'
    worker.mkdir()
    (worker / '.tmp').mkdir()
    (worker / '.eval-inputs').mkdir()
    (worker / 'starter_app.py').write_text('''import numpy as np
def third_moment(ket):
    h=np.array([[.75,-.25],[-.25,-.75]])
    return np.vdot(ket,(4*h@h@h-3*h)@ket).real
''')
    case_id = 'repository-implementation-third-moment'
    attempt = tmp_path / 'attempts' / 'fake' / case_id / '0' / 'baseline'
    attempt.mkdir(parents=True)
    monkeypatch.setattr(
        workspace, 'stage_workspace', lambda *a: {
            'workspace_path': str(worker),
            'skill_path': None
        })
    monkeypatch.setattr(
        providers, 'complete', lambda *a: {
            'choices': [{
                'message': {
                    'role': 'assistant',
                    'content': 'Result saved.'
                },
                'finish_reason': 'stop'
            }],
            'usage': {
                'total_tokens': 17
            }
        })
    spec = {
        'settings': {
            'runtime_python': sys.executable,
            'isolation': 'trusted'
        },
        'protocol': {
            'budget_seconds': 20,
            'tool_call_limit': 2,
            'skill_exposure': 'listed'
        },
        'case_contracts': {
            case_id: {
                'executable_check': True
            }
        }
    }
    row = {'case_id': case_id, 'seed': 0, 'arm': 'baseline'}
    case = {
        'id': case_id,
        'prompt': 'Original.',
        'files': [],
        'assertions': []
    }
    result = campaign.run_attempt(tmp_path, attempt, case, {'alias': 'fake'},
                                  row, spec)
    check = json.loads((attempt / 'verification.json').read_text())
    assert check['status'] == 'passed'
    assert check['result_sha256'] == campaign.digest(attempt / 'result.json')
    assert check['transcript_sha256'] == campaign.digest(attempt /
                                                         'transcript.jsonl')
    completion = check['verified_completion']
    assert 0 < completion['task_seconds'] <= result['resources']['task_seconds']
    assert completion['tokens'] == 17
    assert completion['failed_attempts'] == 0
    exported = assessment._run(tmp_path, attempt, case, 0, 'baseline',
                               {'executable_check': True})
    assert exported['verification'] == 'passed'
