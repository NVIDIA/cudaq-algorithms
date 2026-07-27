import itertools

import numpy as np

import cudaq_algorithms as algorithms


def _make_uccsd_operator_pool(num_qubits, num_electrons, spin=0):
    return algorithms.stateprep.make_uccsd_operator_pool(
        num_qubits, num_electrons, spin)


def _assert_max_qubit_count(operators, max_qubits):
    for op in operators:
        assert op.qubit_count <= max_qubits


def test_generate_with_default_config():
    operators = _make_uccsd_operator_pool(num_qubits=4, num_electrons=2)
    assert operators
    assert len(operators) == 2 + 1
    _assert_max_qubit_count(operators, 4)


def test_generate_with_custom_coefficients():
    operators = _make_uccsd_operator_pool(num_qubits=4, num_electrons=2)

    assert operators
    assert len(operators) == 2 + 1

    for op in operators:
        assert op.qubit_count <= 4
        expected_coefficients = [0.5, 0.125]
        for term in op:
            assert abs(
                term.evaluate_coefficient().real) in expected_coefficients


def test_generate_with_odd_electrons():
    operators = _make_uccsd_operator_pool(num_qubits=6,
                                          num_electrons=3,
                                          spin=1)

    assert operators
    assert len(operators) == 2 * 2 + 4
    _assert_max_qubit_count(operators, 6)


def test_generate_with_large_system():
    operators = _make_uccsd_operator_pool(num_qubits=20, num_electrons=10)

    assert operators
    assert len(operators) == 875
    _assert_max_qubit_count(operators, 20)


def test_uccgsd_operator_pool_counts():
    assert len(algorithms.stateprep.make_uccgsd_operator_pool(4)) == 9
    assert len(algorithms.stateprep.make_uccgsd_operator_pool(4, True,
                                                              False)) == 6
    assert len(algorithms.stateprep.make_uccgsd_operator_pool(4, False,
                                                              True)) == 3

    operators = algorithms.stateprep.make_uccgsd_operator_pool(8)
    assert len(operators) == 238
    _assert_max_qubit_count(operators, 8)


def test_upccgsd_operator_pool_counts():
    operators = algorithms.stateprep.make_upccgsd_operator_pool(20)
    doubles = algorithms.stateprep.make_upccgsd_operator_pool(20, True)

    assert len(operators) == 135
    assert len(doubles) == 45
    _assert_max_qubit_count(operators, 20)
    _assert_max_qubit_count(doubles, 20)


def test_ceo_operator_pool_counts():
    two_orbital_operators = algorithms.stateprep.make_ceo_operator_pool(2)
    four_orbital_operators = algorithms.stateprep.make_ceo_operator_pool(4)

    assert len(two_orbital_operators) == 4
    assert len(four_orbital_operators) == 96
    _assert_max_qubit_count(two_orbital_operators, 4)
    _assert_max_qubit_count(four_orbital_operators, 8)


def test_uccsd_operator_pool_correctness():
    pool = _make_uccsd_operator_pool(num_qubits=4, num_electrons=2)

    generated = [[(term.get_pauli_word(4), term.evaluate_coefficient())
                  for term in op] for op in pool]

    expected_operators = [["XZYI", "YZXI"], ["IXZY", "IYZX"],
                          [
                              "YYYX", "YXXX", "XXYX", "YYXY", "XYYY", "XXXY",
                              "YXYY", "XYXX"
                          ]]
    expected_coefficients = [[complex(-0.5, 0),
                              complex(0.5, 0)],
                             [complex(-0.5, 0),
                              complex(0.5, 0)],
                             [
                                 complex(-0.125, 0),
                                 complex(-0.125, 0),
                                 complex(0.125, 0),
                                 complex(-0.125, 0),
                                 complex(0.125, 0),
                                 complex(0.125, 0),
                                 complex(0.125, 0),
                                 complex(-0.125, 0)
                             ]]

    assert len(generated) == len(expected_operators)

    valid_chars = set("IXYZ")
    for i, operator_terms in enumerate(generated):
        for pauli_word, coefficient in operator_terms:
            assert len(pauli_word) <= 4
            assert set(pauli_word).issubset(valid_chars)

            expected_index = expected_operators[i].index(pauli_word)
            assert coefficient == expected_coefficients[i][expected_index]


# ----------------------------------------------------------------------
# Independent content pins for the generalized / CEO pools.
#
# The kernel tests in test_stateprep_kernels.py derive their dense
# reference from the same make_*_operator_pool they exercise, so a
# count-preserving error in a pool's Pauli words, signs, or index
# conventions would pass on both sides. UCCSD is pinned absolutely by
# test_uccsd_operator_pool_correctness; the tests below give the
# generalized and CEO pools an equivalent absolute check.
# ----------------------------------------------------------------------

_I2 = np.eye(2)
_Z2 = np.diag([1.0, -1.0])
_LOWER = np.array([[0.0, 1.0], [0.0, 0.0]])
_PAULI = {
    "I": _I2,
    "X": np.array([[0.0, 1.0], [1.0, 0.0]]),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]]),
    "Z": _Z2,
}


def _annihilator(mode, num_qubits):
    ops = ([_Z2] * mode + [_LOWER] + [_I2] * (num_qubits - mode - 1))[::-1]
    out = np.array([[1.0]])
    for op in ops:
        out = np.kron(out, op)
    return out


def _dense_at_width(operator, num_qubits):
    matrix = np.zeros((1 << num_qubits, 1 << num_qubits), dtype=complex)
    for term in operator:
        word = term.get_pauli_word(num_qubits)
        coefficient = complex(term.evaluate_coefficient())
        factor = np.array([[1.0]], dtype=complex)
        for label in word:  # word[0] is qubit 0 (leftmost tensor factor)
            factor = np.kron(_PAULI[label], factor)
        matrix += coefficient * factor
    return matrix


def _fermionic_generators(num_qubits):
    """(single, double) anti-Hermitian generator builders over dense JW
    ladder operators — an independent construction of the fermionic
    excitations the generalized pools compile."""
    a = [_annihilator(j, num_qubits) for j in range(num_qubits)]
    ad = [op.conj().T for op in a]

    def single(p, q):
        return ad[p] @ a[q] - ad[q] @ a[p]

    def double(p, q, r, s):
        g = ad[p] @ ad[q] @ a[r] @ a[s]
        return g - g.conj().T

    return single, double


def _assert_bijection_up_to_global_phase(pool, references, num_qubits):
    """Each pool operator equals +/- i times exactly one independent
    generator, and vice versa (a content pin agnostic to the arbitrary
    global sign convention)."""
    pool_matrices = [_dense_at_width(op, num_qubits) for op in pool]
    assert len(pool_matrices) == len(references)
    unmatched = list(references)
    for matrix in pool_matrices:
        hit = next(
            (g for g in unmatched if np.allclose(matrix, 1.0j * g, atol=1e-10)
             or np.allclose(matrix, -1.0j * g, atol=1e-10)), None)
        assert hit is not None, "pool operator matches no fermionic generator"
        unmatched.remove(hit)
    assert not unmatched, "some generators are not produced by the pool"


def test_uccgsd_operator_pool_content_matches_fermionic_generators():
    num_qubits = 4
    single, double = _fermionic_generators(num_qubits)

    singles = [single(p, q) for p in range(1, num_qubits) for q in range(p)]
    doubles = [
        double(quad[i], quad[j], quad[k], quad[l])
        for quad in itertools.combinations(range(num_qubits), 4)
        for (i, j), (k, l) in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3),
                                                                    (1, 2)))
    ]
    _assert_bijection_up_to_global_phase(
        algorithms.stateprep.make_uccgsd_operator_pool(num_qubits),
        singles + doubles, num_qubits)
    # only_singles / only_doubles paths (count-checked elsewhere, never
    # content-checked): pin them independently too.
    _assert_bijection_up_to_global_phase(
        algorithms.stateprep.make_uccgsd_operator_pool(num_qubits,
                                                       only_singles=True),
        singles, num_qubits)
    _assert_bijection_up_to_global_phase(
        algorithms.stateprep.make_uccgsd_operator_pool(num_qubits,
                                                       only_doubles=True),
        doubles, num_qubits)


def test_upccgsd_operator_pool_content_matches_fermionic_generators():
    num_qubits = 4
    single, double = _fermionic_generators(num_qubits)

    # spin-preserving singles (both indices same parity) + paired doubles
    # (same spatial orbital for each pair): the k-UpCCGSD definition.
    singles = [
        single(p, q) for p in range(1, num_qubits) for q in range(p)
        if p % 2 == q % 2
    ]
    num_orbitals = num_qubits // 2
    doubles = [
        double(2 * q + 1, 2 * q, 2 * p + 1, 2 * p) for p in range(num_orbitals)
        for q in range(p + 1, num_orbitals)
    ]
    _assert_bijection_up_to_global_phase(
        algorithms.stateprep.make_upccgsd_operator_pool(num_qubits),
        singles + doubles, num_qubits)
    _assert_bijection_up_to_global_phase(
        algorithms.stateprep.make_upccgsd_operator_pool(num_qubits,
                                                        only_doubles=True),
        doubles, num_qubits)


def test_ceo_operator_pool_correctness():
    # CEO is a deliberately non-fermionic coupled-exchange construction
    # (single = 0.5 (Y_q X_p - X_q Y_p), no JW parity string), so it has no
    # simpler physical reference; pin it with an absolute known answer for
    # num_orbitals=2 (4 qubits), mirroring test_uccsd_operator_pool_correctness.
    pool = algorithms.stateprep.make_ceo_operator_pool(2)
    generated = [
        sorted((term.get_pauli_word(4), complex(term.evaluate_coefficient()))
               for term in op) for op in pool
    ]
    expected = [
        [("XIYI", -0.5 + 0j), ("YIXI", 0.5 + 0j)],
        [("IXIY", -0.5 + 0j), ("IYIX", 0.5 + 0j)],
        [("XXXY", 0.25 + 0j), ("XYXX", -0.25 + 0j), ("YXYY", 0.25 + 0j),
         ("YYYX", -0.25 + 0j)],
        [("XXYX", 0.25 + 0j), ("XYYY", 0.25 + 0j), ("YXXX", -0.25 + 0j),
         ("YYXY", -0.25 + 0j)],
    ]
    assert generated == [sorted(terms) for terms in expected]
