import numpy as np


def test_jordan_wigner_one_body_smoke():
    import cudaq_algorithms as algorithms

    one_body = np.zeros((2, 2), dtype=np.complex128)
    one_body[0, 0] = 1.0
    op = algorithms.fermion.jordan_wigner(one_body)
    assert op is not None


def test_bravyi_kitaev_one_body_smoke():
    import cudaq_algorithms as algorithms

    one_body = np.zeros((2, 2), dtype=np.complex128)
    one_body[0, 0] = 1.0
    op = algorithms.fermion.bravyi_kitaev(one_body)
    assert op is not None
