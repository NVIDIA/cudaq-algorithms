"""Opt-in execution of known-correct artifacts through the actual namespace."""

import json
import os
from pathlib import Path
import sys
import time
import shlex

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runner import campaign, verification, workspace

pytestmark = pytest.mark.skipif(
    os.environ.get('CUDAQ_RUNNER_TEST_ISOLATION') != '1',
    reason='requires actual Linux namespace capabilities and CUDA-Q runtime')

ARTIFACTS = {
    'repository-implementation-third-moment':
    ('starter_app.py', '''from cudaq_algorithms import PauliLCU, Walk
def third_moment(ket):
    return Walk(PauliLCU({'Z':.75,'X':-.25})).moment(ket,3)
'''),
    'source-version-drift':
    ('qsvt_client.py', '''from cudaq_algorithms import PhaseSequence
def build_qsp_kernel(transformer, phases):
    return transformer.kernel(PhaseSequence(phases, convention='qsp'))
'''),
    'implementation-scope-overreach':
    ('pauli_lcu_example.py', '''from cudaq_algorithms import PauliLCU
def build_encoding():
    return PauliLCU({'Z':.75,'X':-.25})
'''),
    'negative-generic-cudaq-kernel': ('evaluation.py', '''import cudaq
@cudaq.kernel
def bell():
    q=cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0],q[1])
'''),
    'science-block-encoding-heralded-observables':
    ('evaluation.py', '''import json
import numpy as np
from cudaq_algorithms import PauliLCU
from cudaq_algorithms.sim_utils import action
terms=[(.4,'IIII')]
for j in range(3):
    for c, letters in [(.25,'XX'),(.25,'YY'),(.175,'ZZ'),(.075,'XY'),(-.075,'YX')]:
        word=['I']*4
        word[j],word[j+1]=letters
        terms.append((c,''.join(word)))
for j in range(4):
    word=['I']*4
    word[j]='Z'
    terms.append((.1,''.join(word)))
encoding=PauliLCU(terms)
neel=np.eye(16,dtype=complex)[10]
spiral=np.ones(1,dtype=complex)
for j in reversed(range(4)):
    spiral=np.kron(spiral,np.array([1,1j**j])/np.sqrt(2))
ms=np.array([sum((-1)**j*(1-2*((b>>j)&1)) for j in range(4))/4 for b in range(16)])
probs=[]
mags=[]
for state in [neel,spiral]:
    branch=action(encoding,state)
    p=float(np.vdot(branch,branch).real)
    probs.append(p)
    mags.append(float(np.vdot(branch,ms*branch).real/p))
print(json.dumps({'lcu_terms':terms,'alpha':encoding.alpha,'probabilities':probs,'magnetizations':mags}))
''')
}


@pytest.mark.parametrize('case_id', list(ARTIFACTS))
def test_real_isolated_artifact_verification(case_id, tmp_path):
    case = next(c for c in campaign.read_json(campaign.EVALS /
                                              'evals.json')['evals']
                if c['id'] == case_id)
    attempt = tmp_path / 'attempt'
    staged = workspace.stage_workspace(
        campaign.EVALS.parents[2],
        campaign.EVALS.parents[1] / 'cudaq-algorithms', campaign.EVALS, case,
        attempt, 'baseline')
    worker = Path(staged['workspace_path'])
    initial = verification.file_inventory(worker)
    name, body = ARTIFACTS[case_id]
    (worker / name).write_text(body)
    result = verification.execute(case_id,
                                  worker,
                                  attempt,
                                  runtime_python=Path(
                                      os.environ['CUDAQ_RUNNER_TEST_PYTHON']),
                                  isolation='unshare',
                                  deadline=time.monotonic() + 90,
                                  initial_files=initial)
    assert result['status'] == 'passed', (attempt /
                                          result['evidence_file']).read_text()
    evidence = json.loads((attempt / result['evidence_file']).read_text())
    assert evidence['artifact_execution']['isolated'] is True


def test_two_arm_controller_executes_real_scientific_artifacts(
        tmp_path, monkeypatch):
    from runner import providers, cli
    root = tmp_path / 'pair'
    case_id = 'science-block-encoding-heralded-observables'
    campaign.prepare(root,
                     campaign.EVALS.parents[2], [{
                         'id': 'offline-fixture',
                         'model': 'offline/fixture'
                     }],
                     pilot=True,
                     case_ids=[case_id],
                     budget=90,
                     runtime_python=Path(
                         os.environ['CUDAQ_RUNNER_TEST_PYTHON']))
    monkeypatch.setattr(providers, 'preflight', lambda *a, **k: {'ok': True})
    calls = []

    def respond(model, messages, tools, timeout):
        calls.append(messages[-1]['role'])
        if model.get('model') == 'offline/judge':
            # Structural grade fixture, not a model-quality observation.
            case = next(c for c in campaign.read_json(root /
                                                      'suite.json')['evals']
                        if c['id'] == case_id)
            evidence = 'Artifact saved for independent execution.'
            item = {
                'value': True,
                'evidence': [evidence],
                'reason': 'Offline grading fixture'
            }
            zero = {
                'value': 0,
                'evidence': [evidence],
                'reason': 'Offline grading fixture'
            }
            grade = {
                'case_id':
                case_id,
                'assertions':
                [dict(item, index=i) for i in range(len(case['assertions']))],
                'expected_output':
                item,
                'claimed_success':
                dict(item, value=False),
                'critical_failures':
                zero,
                'tripwires': {
                    k: zero
                    for k in cli.assessment.TRIPWIRES
                }
            }
            content = json.dumps(grade)
            message = {'role': 'assistant', 'content': content}
        elif not any(m['role'] == 'tool' for m in messages):
            body = ARTIFACTS[case_id][1]
            command = 'python3 -c ' + shlex.quote(
                'from pathlib import Path; Path("evaluation.py").write_text(' +
                repr(body) + ')')
            message = {
                'role':
                'assistant',
                'content':
                None,
                'tool_calls': [{
                    'id': 'write-artifact',
                    'type': 'function',
                    'function': {
                        'name': 'shell',
                        'arguments': json.dumps({'command': command})
                    }
                }]
            }
        else:
            message = {
                'role': 'assistant',
                'content': 'Artifact saved for independent execution.'
            }
        return {
            'choices': [{
                'message': message,
                'finish_reason': 'stop'
            }],
            'usage': {
                'total_tokens': 11
            }
        }

    monkeypatch.setattr(providers, 'complete', respond)
    status = campaign.run(root, emit=lambda _: None)
    assert status['offline-fixture']['answered'] == 2
    assert cli.grade(root, {'model': 'offline/judge'},
                     10)['new_assessments'] == 2
    for path in (root / 'attempts').glob('*/*/*/*'):
        check = campaign.read_json(path / 'verification.json')
        assert check['status'] == 'passed'
        assert (path / 'assessment.provenance.json').exists()
        transcript = (path / 'transcript.jsonl').read_text()
        assert 'artifact_execution' in transcript
        assert 'probabilities' in transcript
    count = len(calls)
    campaign.run(root, emit=lambda _: None)
    assert len(calls) == count
