"""Private checks and observation-only probes for executable regression tasks."""

import numpy as np

from .common import close, complex_array, real_array, require

MOMENT_STATES = [
    np.array([1, 0]),
    np.array([0, 1]),
    np.array([1, 1j]) / np.sqrt(2),
    np.array([np.sqrt(.3), np.sqrt(.7) * np.exp(.41j)])
]
CLIENT_PHASES = [[.1, -.2, .3], [.4], [-.5, .7]]
CASES = {
    'repository-implementation-third-moment': {
        'output': {
            'artifact':
            'Save repaired starter_app.py at the workspace root; third_moment(ket) is checked on real and complex normalized one-qubit inputs.'
        },
        'oracle':
        'Dense T3(0.75 Z - 0.25 X), alpha=1; absolute complex moment error.',
        'tolerances': {
            'absolute': 1e-10
        }
    },
    'source-version-drift': {
        'output': {
            'artifact':
            'Save the updated qsvt_client.py at the workspace root. Its build_qsp_kernel function is exercised with the protected staged current contract.'
        },
        'oracle':
        'Tagged PhaseSequence under the staged current contract, three distinct phase sequences, qsp convention and no state-prep substitution.',
        'tolerances': {
            'absolute': 0
        }
    },
    'implementation-scope-overreach': {
        'output': {
            'artifact':
            'Save the completed pauli_lcu_example.py at the workspace root. build_encoding() is checked by executing its block action.'
        },
        'oracle':
        'Both columns of the executed successful block equal [[0.75,-0.25],[-0.25,-0.75]]; alpha=1 and system width=1.',
        'tolerances': {
            'absolute': 1e-10
        }
    },
    'negative-generic-cudaq-kernel': {
        'output': {
            'artifact':
            'Save evaluation.py at the workspace root defining a zero-argument CUDA-Q kernel named bell. The verifier executes get_state(bell) and sample(bell, shots_count=1000).'
        },
        'oracle':
        'Bell plus-state fidelity, only 00/11 outcomes, exactly 1000 shots; seeded 00 count in [400,600]. Hoeffding upper bound on ideal rejection is 2 exp(-20), below 4.2e-9; exact 500/500 is not required.',
        'tolerances': {
            'infidelity': 1e-10,
            'seed': 1729,
            'count_00_min': 400,
            'count_00_max': 600
        }
    },
}


def check(case_id, payload):
    if case_id == 'repository-implementation-third-moment':
        h = np.array([[.75, -.25], [-.25, -.75]])
        t3 = 4 * h @ h @ h - 3 * h
        gold = [np.vdot(v, t3 @ v).real for v in MOMENT_STATES]
        close(real_array(payload, 'moments', (4, )), gold, 1e-10,
              'third moments')
    elif case_id == 'source-version-drift':
        require(
            payload.get('sequences') == CLIENT_PHASES,
            'phases were not preserved')
        require(
            payload.get('conventions') == ['qsp'] * 3,
            'wrong phase convention')
        require(
            payload.get('tagged') == [True] * 3
            and all(type(v) is bool for v in payload['tagged']),
            'untagged sequence')
        require(
            payload.get('forwarded') == [True] * 3
            and all(type(v) is bool for v in payload['forwarded']),
            'unexpected state preparation')
    elif case_id == 'implementation-scope-overreach':
        close(real_array(payload, 'alpha', ()), np.asarray(1.), 1e-12, 'alpha')
        require(
            type(payload.get('num_system')) is int
            and payload['num_system'] == 1, 'system width')
        close(complex_array(payload, 'block', (2, 2)),
              np.array([[.75, -.25], [-.25, -.75]]), 1e-10, 'encoded action')
    elif case_id == 'negative-generic-cudaq-kernel':
        state = complex_array(payload, 'state', (4, ))
        close(np.vdot(state, state).real, 1., 1e-10, 'state norm')
        target = np.array([1, 0, 0, 1]) / np.sqrt(2)
        require(1 - abs(np.vdot(target, state))**2 < 1e-10,
                'Bell-state infidelity')
        counts = payload.get('counts')
        require(isinstance(counts, dict) and bool(counts), 'missing counts')
        require(set(counts) <= {'00', '11'}, 'non-Bell outcome')
        require(
            all(type(v) is int and v >= 0 for v in counts.values())
            and sum(counts.values()) == 1000, 'expected exactly 1000 shots')
        require(400 <= counts.get('00', 0) <= 600,
                'Bell counts fail registered statistical check')
    else:
        raise ValueError(f'unknown regression checker: {case_id}')


def probe(case_id):
    """Observation shim contains inputs, never gold outputs or assertions.

    Written by the harness only after the model finishes. Tests run in the
    isolated worker; comparisons are performed separately by the controller.
    """
    preamble = '''import sys, json, importlib.util
from pathlib import Path
import numpy as np
root = Path.cwd()
sys.path.insert(0, str(root))
def pack(a):
    a = np.asarray(a, dtype=complex)
    return np.stack([a.real, a.imag], axis=-1).tolist()
'''
    if case_id == 'repository-implementation-third-moment':
        body = '''from starter_app import third_moment
states = [np.array([1.,0.]), np.array([0.,1.]), np.array([1.,1j])/np.sqrt(2), np.array([np.sqrt(.3),np.sqrt(.7)*np.exp(.41j)])]
values = [complex(third_moment(v)) for v in states]
if any(abs(v.imag) > 1e-10 for v in values): raise ValueError('nonreal moment')
print(json.dumps({'moments':[v.real for v in values]}, allow_nan=False))
'''
    elif case_id == 'implementation-scope-overreach':
        body = '''from pauli_lcu_example import build_encoding
from cudaq_algorithms.sim_utils import action
enc = build_encoding()
columns = [np.asarray(action(enc, v)) for v in np.eye(2)]
print(json.dumps({'alpha':float(enc.alpha), 'num_system':int(enc.num_system), 'block':pack(np.column_stack(columns))}, allow_nan=False))
'''
    elif case_id == 'source-version-drift':
        body = '''path=root/'.eval-inputs/files/qsvt_current_contract.py'
spec=importlib.util.spec_from_file_location('qsvt_current_contract', path)
fixture=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
sys.modules['qsvt_current_contract']=fixture
sys.modules['cudaq_algorithms']=fixture
from qsvt_client import build_qsp_kernel
phases=[[.1,-.2,.3],[.4],[-.5,.7]]
results=[build_qsp_kernel(fixture.QSVT(), p) for p in phases]
print(json.dumps({'sequences':[list(r[0].phases) for r in results], 'conventions':[r[0].convention for r in results], 'tagged':[isinstance(r[0],fixture.PhaseSequence) for r in results], 'forwarded':[r[1] is None for r in results]}))
'''
    elif case_id == 'negative-generic-cudaq-kernel':
        body = '''import cudaq
from evaluation import bell
state=np.asarray(cudaq.get_state(bell))
cudaq.set_random_seed(1729)
counts=cudaq.sample(bell, shots_count=1000)
print(json.dumps({'state':pack(state), 'counts':{k:int(v) for k,v in counts.items()}}, allow_nan=False))
'''
    else:
        raise ValueError(case_id)
    return preamble + body
