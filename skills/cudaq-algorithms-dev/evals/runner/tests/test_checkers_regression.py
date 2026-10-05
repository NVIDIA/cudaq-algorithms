import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runner.checkers import regression
from runner.checkers.common import CheckFailure


def test_third_moment_complex_states_and_wrong_sign():
    h = np.array([[.75, -.25], [-.25, -.75]])
    polynomial = 4 * np.linalg.matrix_power(h, 3) - 3 * h
    states = regression.MOMENT_STATES
    values = [float(np.vdot(v, polynomial @ v).real) for v in states]
    regression.check('repository-implementation-third-moment',
                     {'moments': values})
    with pytest.raises(CheckFailure):
        regression.check('repository-implementation-third-moment',
                         {'moments': [-x for x in values]})


def test_pauli_lcu_checks_action_not_just_alpha():
    block = np.array([[.75, -.25], [-.25, -.75]], dtype=complex)
    data = {
        'alpha': 1,
        'num_system': 1,
        'block': np.stack([block.real, block.imag], axis=-1).tolist()
    }
    regression.check('implementation-scope-overreach', data)
    data['block'][0][1][0] *= -1
    with pytest.raises(CheckFailure):
        regression.check('implementation-scope-overreach', data)


def test_bell_requires_correct_phase_and_exact_shot_total():
    data = {
        'state': [[2**-.5, 0], [0, 0], [0, 0], [2**-.5, 0]],
        'counts': {
            '00': 491,
            '11': 509
        }
    }
    regression.check('negative-generic-cudaq-kernel', data)
    data['state'][3][0] *= -1
    with pytest.raises(CheckFailure):
        regression.check('negative-generic-cudaq-kernel', data)
    data['state'][3][0] *= -1
    data['counts']['01'] = 1
    with pytest.raises(CheckFailure):
        regression.check('negative-generic-cudaq-kernel', data)
    data['counts'] = {'00': 1000}
    with pytest.raises(CheckFailure):
        regression.check('negative-generic-cudaq-kernel', data)


def test_client_requires_current_phase_contract_on_multiple_inputs():
    data = {
        'sequences': regression.CLIENT_PHASES,
        'conventions': ['qsp'] * 3,
        'tagged': [True] * 3,
        'forwarded': [True] * 3
    }
    regression.check('source-version-drift', data)
    data['tagged'][1] = False
    with pytest.raises(CheckFailure):
        regression.check('source-version-drift', data)
