import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runner.checkers import common


def test_fermion_anticommutation_and_sector_order():
    ops = common.annihilators(3)
    for i, a in enumerate(ops):
        for j, b in enumerate(ops):
            assert np.allclose(a @ b.conj().T + b.conj().T @ a,
                               np.eye(8) * (i == j))
    assert list(common.sector(4, 2, spin=(1, 1))) == [3, 6, 9, 12]
    assert common.pauli(2, [(1, {
        0: 'Z'
    })]).diagonal().tolist() == [1, -1, 1, -1]


@pytest.mark.parametrize('bad', [[True], [float('nan')], ['1'], [1, 2]])
def test_real_array_rejects_invalid_scientific_data(bad):
    with pytest.raises(common.CheckFailure):
        common.real_array({'x': bad}, 'x', (1, ))


def test_complex_and_absolute_phase_error():
    v = common.complex_array({'x': [[0, 1], [1, 0]]}, 'x', (2, ))
    assert np.array_equal(v, [1j, 1])
    with pytest.raises(common.CheckFailure):
        common.close(v, -v, 1e-6, 'phase-sensitive state')
    with pytest.raises(common.CheckFailure):
        common.close([float('nan')], [0], 1e-6, 'finite')
