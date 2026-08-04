# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudaq
import cudaq_algorithms as algorithms


def test_uccsd_helper_counts():
    excitations = algorithms.stateprep.get_uccsd_excitations(4, 2, 0)

    assert [len(group) for group in excitations] == [1, 1, 1, 0, 0]
    assert algorithms.stateprep.get_num_uccsd_parameters(4, 2, 0) == 3
    assert len(algorithms.stateprep.make_uccsd_operator_pool(4, 2, 0)) == 3


def test_uccsd_active_space_excitations():
    singles_alpha, singles_beta, doubles_mixed, doubles_alpha, doubles_beta = \
        algorithms.stateprep.get_uccsd_excitations(8, 4, 0)

    assert singles_alpha == [[0, 4], [0, 6], [2, 4], [2, 6]]
    assert singles_beta == [[1, 5], [1, 7], [3, 5], [3, 7]]
    assert doubles_alpha == [[0, 2, 4, 6]]
    assert doubles_beta == [[1, 3, 5, 7]]
    assert len(doubles_mixed) == 16
    assert algorithms.stateprep.get_num_uccsd_parameters(8, 4, 0) == 26


def test_uccsd_open_shell_excitation_counts():
    # Open-shell system (3 electrons, doublet) exercises the spin>0 code path.
    singles_alpha, singles_beta, doubles_mixed, doubles_alpha, doubles_beta = \
        algorithms.stateprep.get_uccsd_excitations(6, 3, 1)

    assert len(singles_alpha) == 2
    assert len(singles_beta) == 2
    assert len(doubles_mixed) == 4
    assert doubles_alpha == []
    assert doubles_beta == []
    assert algorithms.stateprep.get_num_uccsd_parameters(6, 3, 1) == 8


def test_generalized_stateprep_helper_shapes():
    words, coeffs = algorithms.stateprep.get_uccgsd_pauli_lists(4)
    assert len(words) == len(coeffs) == 9
    assert [len(group) for group in words] == [2, 2, 2, 2, 2, 2, 8, 8, 8]
    assert all(
        len(word_group) == len(coeff_group)
        for word_group, coeff_group in zip(words, coeffs))
    assert len(algorithms.stateprep.make_uccgsd_operator_pool(4)) == 9
    assert len(algorithms.stateprep.make_uccgsd_operator_pool(4, True,
                                                              False)) == 6
    assert len(algorithms.stateprep.make_uccgsd_operator_pool(4, False,
                                                              True)) == 3

    words, coeffs = algorithms.stateprep.get_upccgsd_pauli_lists(4)
    assert len(words) == len(coeffs) == 3
    assert [len(group) for group in words] == [2, 2, 8]
    assert all(
        len(word_group) == len(coeff_group)
        for word_group, coeff_group in zip(words, coeffs))
    assert len(algorithms.stateprep.make_upccgsd_operator_pool(4)) == 3
    assert len(algorithms.stateprep.make_upccgsd_operator_pool(4, True)) == 1

    words, coeffs = algorithms.stateprep.get_ceo_pauli_lists(2)
    assert len(words) == len(coeffs) == 4
    assert [len(group) for group in words] == [2, 2, 4, 4]
    assert all(
        len(word_group) == len(coeff_group)
        for word_group, coeff_group in zip(words, coeffs))
    assert len(algorithms.stateprep.make_ceo_operator_pool(2)) == 4


def test_upccgsd_excitation_list_counts():
    num_qubits = 20
    num_orbitals = num_qubits // 2

    words, coeffs = algorithms.stateprep.get_upccgsd_pauli_lists(num_qubits)
    assert len(words) == len(coeffs) == 3 * num_orbitals * (num_orbitals -
                                                            1) // 2

    words, coeffs = algorithms.stateprep.get_upccgsd_pauli_lists(
        num_qubits, True)
    assert len(words) == len(coeffs) == num_orbitals * (num_orbitals - 1) // 2


def test_uccsd_kernel_smoke():
    num_qubits = 4
    num_electrons = 2
    spin = 0
    num_parameters = algorithms.stateprep.get_num_uccsd_parameters(
        num_qubits, num_electrons, spin)

    @cudaq.kernel
    def kernel(thetas: list[float]):
        q = cudaq.qvector(num_qubits)
        algorithms.stateprep.uccsd(q, thetas, num_electrons, spin)
        mz(q)

    counts = cudaq.sample(kernel, [0.0] * num_parameters, shots_count=10)
    assert counts is not None


def test_uccsd_fixed_parameter_regression():
    num_qubits = 8
    num_electrons = 2
    thetas = [
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558
    ]

    @cudaq.kernel
    def kernel(thetas: list[float]):
        q = cudaq.qvector(num_qubits)
        for i in range(num_electrons):
            x(q[i])
        algorithms.stateprep.uccsd(q, thetas, num_electrons, 0)
        mz(q)

    counts = cudaq.sample(kernel, thetas, shots_count=1000)

    assert "00000011" in counts
    assert "00000110" in counts
    assert "00010010" in counts
    assert "01000010" in counts
    assert "10000001" in counts
    assert "11000000" in counts


def test_uccgsd_kernel_smoke():
    words, coeffs = algorithms.stateprep.get_uccgsd_pauli_lists(4)

    @cudaq.kernel
    def kernel(thetas: list[float], pauli_words: list[list[cudaq.pauli_word]],
               coefficients: list[list[float]]):
        q = cudaq.qvector(4)
        algorithms.stateprep.uccgsd(q, thetas, pauli_words, coefficients)
        mz(q)

    counts = cudaq.sample(kernel, [0.0] * len(words),
                          words,
                          coeffs,
                          shots_count=10)
    assert counts is not None


def test_upccgsd_kernel_smoke():
    words, coeffs = algorithms.stateprep.get_upccgsd_pauli_lists(4)

    @cudaq.kernel
    def kernel(thetas: list[float], pauli_words: list[list[cudaq.pauli_word]],
               coefficients: list[list[float]]):
        q = cudaq.qvector(4)
        algorithms.stateprep.upccgsd(q, thetas, pauli_words, coefficients)
        mz(q)

    counts = cudaq.sample(kernel, [0.0] * len(words),
                          words,
                          coeffs,
                          shots_count=10)
    assert counts is not None


def test_ceo_kernel_smoke():
    words, coeffs = algorithms.stateprep.get_ceo_pauli_lists(2)

    @cudaq.kernel
    def kernel(thetas: list[float], pauli_words: list[list[cudaq.pauli_word]],
               coefficients: list[list[float]]):
        q = cudaq.qvector(4)
        algorithms.stateprep.ceo(q, thetas, pauli_words, coefficients)
        mz(q)

    counts = cudaq.sample(kernel, [0.0] * len(words),
                          words,
                          coeffs,
                          shots_count=10)
    assert counts is not None
