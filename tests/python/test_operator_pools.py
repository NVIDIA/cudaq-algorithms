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
