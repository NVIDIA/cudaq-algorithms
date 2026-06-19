#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Tests for PauliLCU, qubitization, and QSVT Python primitives.

The tests are organized by abstraction layer. The lower-level tests validate the
PauliLCU metadata, block encoding, and qubitization walk. The QSVT tests then
check that the public phase-sequence API composes those primitives correctly and
that QSPPACK-generated Hamiltonian-simulation phases flow through the same API.
"""

import pytest
import numpy as np
import cudaq
from cudaq import spin
import cudaq_algorithms as algorithms

_FOUR_QUBIT_TERMS = [
    (0.31, ((0, "X"), )),
    (-0.22, ((1, "Z"), )),
    (0.17, ((2, "Y"), )),
    (0.13, ((3, "X"), )),
    (0.11, ((0, "Z"), (2, "Z"))),
    (-0.19, ((1, "X"), (3, "Y"))),
    (0.07, ((0, "Y"), (2, "X"), (3, "Z"))),
    (0.05, ((0, "Z"), (1, "Z"), (2, "Z"), (3, "Z"))),
]

_QSVT_HAMILTONIAN_SIMULATION_TERMS = [
    (0.70, "ZIII"),
    (-0.43, "IZII"),
    (0.31, "IIZI"),
    (-0.22, "IIIZ"),
    (0.19, "XXII"),
    (-0.17, "IYYI"),
    (0.13, "IZZX"),
    (0.11, "XYYX"),
]

_QSVT_COS_PHASES = [0.15, -0.42, 0.88, -0.42, 0.15]
_QSVT_SIN_PHASES = [0.23, 0.54, -0.12, -0.54, -0.23]


def _pauli_string_terms_to_indexed_terms(terms):
    return [(coefficient,
             tuple((qubit, op) for qubit, op in enumerate(word) if op != "I"))
            for coefficient, word in terms]


def _spin_term(paulis):
    term = None
    for qubit, op_name in paulis:
        if op_name == "X":
            op = spin.x(qubit)
        elif op_name == "Y":
            op = spin.y(qubit)
        elif op_name == "Z":
            op = spin.z(qubit)
        else:
            raise ValueError(f"Unsupported Pauli operator: {op_name}")
        term = op if term is None else term * op
    return term


def _four_qubit_hamiltonian():
    hamiltonian = None
    for coefficient, paulis in _FOUR_QUBIT_TERMS:
        term = coefficient * _spin_term(paulis)
        hamiltonian = term if hamiltonian is None else hamiltonian + term
    return hamiltonian


def _hamiltonian_from_indexed_terms(terms):
    hamiltonian = None
    for coefficient, paulis in terms:
        term = coefficient * _spin_term(paulis)
        hamiltonian = term if hamiltonian is None else hamiltonian + term
    return hamiltonian


def _pauli_sum_matrix(terms, num_qubits):
    dimension = 1 << num_qubits
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)

    for coefficient, paulis in terms:
        for column in range(dimension):
            row = column
            phase = 1.0 + 0.0j
            for qubit, op_name in paulis:
                bit = (column >> qubit) & 1
                if op_name == "X":
                    row ^= (1 << qubit)
                elif op_name == "Y":
                    row ^= (1 << qubit)
                    phase *= 1.0j if bit == 0 else -1.0j
                elif op_name == "Z":
                    phase *= 1.0 if bit == 0 else -1.0
                else:
                    raise ValueError(f"Unsupported Pauli operator: {op_name}")
            matrix[row, column] += coefficient * phase

    return matrix


def _random_normalized_ket(num_qubits, seed):
    rng = np.random.default_rng(seed)
    dimension = 1 << num_qubits
    ket = rng.normal(size=dimension) + 1.0j * rng.normal(size=dimension)
    ket /= np.linalg.norm(ket)
    return ket.astype(np.complex128)


def _kernel_data(encoding):
    return encoding.kernel_data().unpack()


def _zero_ancilla_component(full_state, num_system, num_ancilla):
    state_vector = np.asarray(full_state, dtype=np.complex128)
    system_dimension = 1 << num_system
    expected_dimension = 1 << (num_system + num_ancilla)
    assert state_vector.shape == (expected_dimension, )
    # The tests allocate the system state first and the LCU ancillas second.
    # CUDA-Q stores q[0] as the least-significant statevector bit, so the
    # all-zero ancilla subspace is the first contiguous system block.
    return state_vector[:system_dimension].copy()


def _run_qsvt_good_component(initial_state, num_system, num_ancilla, phases,
                             walk_directions, kernel_data):
    phase_data, direction_data, angles, term_controls, term_ops, term_lengths, \
        term_signs = algorithms.qsvt.pauli_lcu_kernel_args(
            phases, kernel_data, walk_directions)

    @cudaq.kernel
    def qsvt_kernel(state: cudaq.State):
        system = cudaq.qvector(state)
        signal = cudaq.qvector(num_ancilla)
        algorithms.qsvt.apply_phase_sequence(signal, system, phase_data,
                                             direction_data, angles,
                                             term_controls, term_ops,
                                             term_lengths, term_signs)

    full_state = cudaq.get_state(qsvt_kernel, initial_state)
    return _zero_ancilla_component(full_state, num_system, num_ancilla)


def _run_explicit_qsvt_good_component(initial_state, num_system, num_ancilla,
                                      phases, walk_directions, kernel_data):
    assert all(direction == 0 for direction in walk_directions)
    angles, term_controls, term_ops, term_lengths, term_signs = kernel_data

    @cudaq.kernel
    def qsvt_kernel(state: cudaq.State):
        system = cudaq.qvector(state)
        signal = cudaq.qvector(num_ancilla)
        algorithms.qsvt.apply_signal_phase(signal, phases[0])
        for i in range(1, len(phases)):
            algorithms.block_encoding.apply(signal, system, angles,
                                            term_controls, term_ops,
                                            term_lengths, term_signs)
            algorithms.qubitization.reflect_about_zero(signal)
            algorithms.qsvt.apply_signal_phase(signal, phases[i])

    full_state = cudaq.get_state(qsvt_kernel, initial_state)
    return _zero_ancilla_component(full_state, num_system, num_ancilla)


def _qsp_to_projector_phases(phases):
    return algorithms.qsvt.projector_phases_from_qsp(phases)


def _qsppack_hamiltonian_simulation_phases(tau, degree=16):
    qsppack = pytest.importorskip("qsppack")
    scipy_special = pytest.importorskip("scipy.special")
    jv = scipy_special.jv

    cos_coefficients = np.array([0.5 * jv(0, tau)] +
                                [((-1)**k) * jv(2 * k, tau)
                                 for k in range(1, degree // 2 + 1)],
                                dtype=np.float64)
    sin_coefficients = np.array([((-1)**k) * jv(2 * k + 1, tau)
                                 for k in range(degree // 2)],
                                dtype=np.float64)
    common_options = {
        "criteria": 1e-12,
        "method": "Newton",
        "typePhi": "full",
        "useReal": True,
    }

    cos_phases, cos_info = qsppack.solve(cos_coefficients, 0, {
        **common_options, "targetPre": True
    })
    sin_phases, sin_info = qsppack.solve(sin_coefficients, 1, {
        **common_options, "targetPre": False
    })
    return ([float(phase) for phase in cos_phases],
            [float(phase) for phase in sin_phases], cos_info, sin_info)


def _exact_time_evolved_state(hamiltonian_matrix, initial_ket, evolution_time):
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix)
    amplitudes = eigenvectors.conj().T @ initial_ket
    phases = np.exp(-1.0j * evolution_time * eigenvalues)
    return eigenvectors @ (phases * amplitudes), eigenvalues, eigenvectors


@pytest.fixture
def qpp_cpu_target():
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


def _assert_good_component_matches(good_component, expected_component):
    observed_probability = np.vdot(good_component, good_component).real
    expected_probability = np.vdot(expected_component, expected_component).real

    assert observed_probability > 1e-14
    assert observed_probability == pytest.approx(expected_probability,
                                                 abs=1e-10)
    assert np.allclose(good_component, expected_component, atol=1e-10)


def test_pauli_lcu_metadata_binding():
    """Validate host-side PauliLCU metadata and flattened kernel data access."""

    h = 2.0 + 0.5 * spin.z(0) - 0.25 * spin.x(0)
    encoding = algorithms.PauliLCU(h, num_qubits=1)

    assert encoding.constant_term == pytest.approx(2.0)
    assert encoding.term_count == 3
    assert encoding.padded_term_count == 4
    assert hasattr(encoding, 'controlled_select')

    metadata = encoding.metadata()
    assert isinstance(metadata, algorithms.PauliLCUMetadata)
    assert metadata.num_system_qubits == 1
    assert metadata.num_ancilla_qubits == 2
    assert metadata.num_terms == 3
    assert metadata.padded_num_terms == 4
    assert metadata.normalization == pytest.approx(2.75)
    assert metadata.constant_term == pytest.approx(2.0)
    assert metadata.coefficient_threshold == pytest.approx(1e-12)
    assert metadata.include_identity is True
    assert encoding.include_identity is True

    kernel_data = encoding.kernel_data()
    assert isinstance(kernel_data, algorithms.PauliLCUKernelData)
    assert kernel_data.num_system_qubits == metadata.num_system_qubits
    assert kernel_data.num_ancilla_qubits == metadata.num_ancilla_qubits
    assert kernel_data.num_terms == metadata.num_terms
    assert kernel_data.padded_num_terms == metadata.padded_num_terms
    assert kernel_data.angles == pytest.approx(list(encoding.get_angles()))
    assert kernel_data.state_prep_angles == pytest.approx(
        list(encoding.get_angles()))
    assert kernel_data.term_controls == list(encoding.get_term_controls())
    assert kernel_data.term_ops == list(encoding.get_term_ops())
    assert kernel_data.term_lengths == list(encoding.get_term_lengths())
    assert kernel_data.term_signs == list(encoding.get_term_signs())
    assert kernel_data.as_tuple() == kernel_data.unpack()
    assert len(kernel_data.unpack()) == 5

    # Identity terms are retained in H / alpha and also reported separately so
    # future algorithms may choose to account for scalar shifts outside LCU.
    assert encoding.constant_term == pytest.approx(metadata.constant_term)


def test_pauli_lcu_constant_term_policy_matches_numpy(qpp_cpu_target):
    """Validate include_identity=True/False against dense block action."""

    h = 1.25 + 0.5 * spin.z(0) - 0.25 * spin.x(0)
    initial_ket = _random_normalized_ket(1, seed=111)
    initial_state = cudaq.State.from_data(initial_ket)
    matrix_full = np.array([[1.75, -0.25], [-0.25, 0.75]], dtype=np.complex128)
    matrix_shifted = matrix_full - 1.25 * np.eye(2, dtype=np.complex128)

    def run_good_component(encoding):
        angles, term_controls, term_ops, term_lengths, term_signs = _kernel_data(
            encoding)
        num_ancilla = encoding.num_ancilla

        @cudaq.kernel
        def block_encode(state: cudaq.State):
            system = cudaq.qvector(state)
            ancilla = cudaq.qvector(num_ancilla)
            algorithms.block_encoding.apply(ancilla, system, angles,
                                            term_controls, term_ops,
                                            term_lengths, term_signs)

        full_state = cudaq.get_state(block_encode, initial_state)
        return _zero_ancilla_component(full_state, 1, num_ancilla)

    include_encoding = algorithms.PauliLCU(h,
                                           num_qubits=1,
                                           include_identity=True)
    assert include_encoding.include_identity is True
    assert include_encoding.term_count == 3
    assert include_encoding.constant_term == pytest.approx(1.25)
    assert include_encoding.normalization == pytest.approx(2.0)
    _assert_good_component_matches(run_good_component(include_encoding),
                                   (matrix_full @ initial_ket) /
                                   include_encoding.normalization)

    shifted_encoding = algorithms.PauliLCU(h,
                                           num_qubits=1,
                                           include_identity=False)
    assert shifted_encoding.include_identity is False
    assert shifted_encoding.term_count == 2
    assert shifted_encoding.constant_term == pytest.approx(1.25)
    assert shifted_encoding.normalization == pytest.approx(0.75)
    assert shifted_encoding.metadata().include_identity is False
    _assert_good_component_matches(run_good_component(shifted_encoding),
                                   (matrix_shifted @ initial_ket) /
                                   shifted_encoding.normalization)

    with pytest.raises(RuntimeError, match="non-identity"):
        algorithms.PauliLCU(1.25 + 0.0 * spin.z(0),
                            num_qubits=1,
                            include_identity=False)


def test_pauli_lcu_block_encoding_device_interop():
    """Smoke-test that public LCU and qubitization helpers compile in kernels."""

    h = 0.6 * spin.x(0) + 0.8 * spin.z(0)
    encoding = algorithms.PauliLCU(h, num_qubits=1)

    angles = list(encoding.get_angles())
    term_controls = list(encoding.get_term_controls())
    term_ops = list(encoding.get_term_ops())
    term_lengths = list(encoding.get_term_lengths())
    term_signs = list(encoding.get_term_signs())
    num_ancilla = encoding.num_ancilla
    num_system = encoding.num_system

    @cudaq.kernel
    def kernel():
        ancilla = cudaq.qvector(num_ancilla)
        system = cudaq.qvector(num_system)
        algorithms.block_encoding.prepare(ancilla, angles)
        algorithms.block_encoding.select(ancilla, system, term_controls,
                                         term_ops, term_lengths, term_signs)
        algorithms.block_encoding.unprepare(ancilla, angles)
        algorithms.block_encoding.apply(ancilla, system, angles, term_controls,
                                        term_ops, term_lengths, term_signs)
        algorithms.qubitization.reflect_about_zero(ancilla)
        algorithms.qubitization.reflect_about_prepare(ancilla, angles)
        algorithms.qubitization.apply_walk(ancilla, system, angles,
                                           term_controls, term_ops,
                                           term_lengths, term_signs)

    counts = cudaq.sample(kernel, shots_count=16)
    assert len(counts) > 0


def test_pauli_lcu_block_encoding_matches_numpy_good_subspace(qpp_cpu_target):
    """Check that PauliLCU block encoding produces H|psi>/alpha.

    The CUDA-Q statevector is only used in the test: we postselect the all-zero
    ancilla component and compare it with dense NumPy multiplication.
    """

    num_system = 4
    hamiltonian = _four_qubit_hamiltonian()
    hamiltonian_matrix = _pauli_sum_matrix(_FOUR_QUBIT_TERMS, num_system)
    initial_ket = _random_normalized_ket(num_system, seed=2026)
    initial_state = cudaq.State.from_data(initial_ket)

    encoding = algorithms.PauliLCU(hamiltonian, num_qubits=num_system)
    assert encoding.num_ancilla == 3
    assert encoding.term_count == len(_FOUR_QUBIT_TERMS)

    angles, term_controls, term_ops, term_lengths, term_signs = _kernel_data(
        encoding)
    num_ancilla = encoding.num_ancilla

    @cudaq.kernel
    def block_encode(state: cudaq.State):
        system = cudaq.qvector(state)
        ancilla = cudaq.qvector(num_ancilla)
        algorithms.block_encoding.apply(ancilla, system, angles, term_controls,
                                        term_ops, term_lengths, term_signs)

    full_state = cudaq.get_state(block_encode, initial_state)
    good_component = _zero_ancilla_component(full_state, num_system,
                                             num_ancilla)
    expected_component = (
        hamiltonian_matrix @ initial_ket) / encoding.normalization

    _assert_good_component_matches(good_component, expected_component)


def test_pauli_lcu_qubitization_walk_matches_numpy_good_subspace(
        qpp_cpu_target):
    """Check one qubitization walk against dense Hamiltonian action."""

    num_system = 4
    hamiltonian = _four_qubit_hamiltonian()
    hamiltonian_matrix = _pauli_sum_matrix(_FOUR_QUBIT_TERMS, num_system)
    initial_ket = _random_normalized_ket(num_system, seed=2027)
    initial_state = cudaq.State.from_data(initial_ket)

    encoding = algorithms.PauliLCU(hamiltonian, num_qubits=num_system)
    assert encoding.num_ancilla == 3

    angles, term_controls, term_ops, term_lengths, term_signs = _kernel_data(
        encoding)
    num_ancilla = encoding.num_ancilla

    @cudaq.kernel
    def walk_once(state: cudaq.State):
        system = cudaq.qvector(state)
        ancilla = cudaq.qvector(num_ancilla)
        algorithms.block_encoding.prepare(ancilla, angles)
        algorithms.qubitization.apply_walk(ancilla, system, angles,
                                           term_controls, term_ops,
                                           term_lengths, term_signs)

        # Move the PREPARE signal state back to |0...0> so the statevector
        # test can postselect the good subspace and ignore the orthogonal junk
        # space.
        algorithms.block_encoding.unprepare(ancilla, angles)

    full_state = cudaq.get_state(walk_once, initial_state)
    good_component = _zero_ancilla_component(full_state, num_system,
                                             num_ancilla)

    # The current zero-state reflection applies a -1 phase to |0...0>, so the
    # prepared-signal block of one walk is -H / alpha for this convention.
    expected_component = -(
        hamiltonian_matrix @ initial_ket) / encoding.normalization

    _assert_good_component_matches(good_component, expected_component)


def test_qsvt_hamiltonian_simulation_sequence_matches_explicit_loop(
        qpp_cpu_target):
    """Validate the public QSVT sequence helper against an explicit circuit.

    This does not test phase-generation accuracy. The fixed phases are a stable
    fixture used to ensure apply_phase_sequence() is equivalent to manually
    composing signal phases, PauliLCU block encodings, and zero-signal
    reflections.
    """

    num_system = 4
    terms = _pauli_string_terms_to_indexed_terms(
        _QSVT_HAMILTONIAN_SIMULATION_TERMS)
    hamiltonian = _hamiltonian_from_indexed_terms(terms)
    initial_ket = np.zeros(1 << num_system, dtype=np.complex128)
    initial_ket[0] = 1.0
    initial_state = cudaq.State.from_data(initial_ket)

    encoding = algorithms.PauliLCU(hamiltonian, num_qubits=num_system)
    assert encoding.num_ancilla == 3
    assert encoding.term_count == len(terms)

    kernel_data = _kernel_data(encoding)
    cos_sequence = algorithms.qsvt.phase_sequence(_QSVT_COS_PHASES)
    sin_sequence = algorithms.qsvt.phase_sequence(_QSVT_SIN_PHASES)
    cos_phases = cos_sequence.phase_data
    sin_phases = sin_sequence.phase_data
    cos_walk_directions = cos_sequence.walk_direction_data
    sin_walk_directions = sin_sequence.walk_direction_data

    quantum_cos_state = _run_qsvt_good_component(initial_state, num_system,
                                                 encoding.num_ancilla,
                                                 cos_phases,
                                                 cos_walk_directions,
                                                 kernel_data)
    quantum_sin_state = _run_qsvt_good_component(initial_state, num_system,
                                                 encoding.num_ancilla,
                                                 sin_phases,
                                                 sin_walk_directions,
                                                 kernel_data)
    quantum_time_evolved_state = quantum_cos_state - 1.0j * quantum_sin_state

    explicit_cos_state = _run_explicit_qsvt_good_component(
        initial_state, num_system, encoding.num_ancilla, cos_phases,
        cos_walk_directions, kernel_data)
    explicit_sin_state = _run_explicit_qsvt_good_component(
        initial_state, num_system, encoding.num_ancilla, sin_phases,
        sin_walk_directions, kernel_data)
    explicit_time_evolved_state = explicit_cos_state - 1.0j * explicit_sin_state

    # These fixed phase sequences are deterministic stand-ins for externally
    # generated Hamiltonian-simulation phases. This test validates that the
    # public QSVT sequence helper matches the equivalent explicit composition of
    # signal phases, PauliLCU block encodings, and zero-signal reflections.
    assert np.linalg.norm(quantum_cos_state - explicit_cos_state) < 1e-10
    assert np.linalg.norm(quantum_sin_state - explicit_sin_state) < 1e-10
    assert (np.linalg.norm(quantum_time_evolved_state -
                           explicit_time_evolved_state) < 1e-10)
    assert np.linalg.norm(quantum_time_evolved_state) > 1e-8


def test_qsppack_generated_phases_validate_device_sequence_and_exact_response(
        qpp_cpu_target):
    """Run a small QSPPACK Hamiltonian-simulation flow end to end.

    QSPPACK provides cos/sin phase sequences. The test converts those phases to
    the projector-phase convention used by apply_phase_sequence(), executes the
    quantum circuit, and compares the resulting evolved state with exact dense
    diagonalization. The scalar phases_to_poly() checks diagnose phase quality
    separately from device execution.
    """

    num_system = 4
    evolution_time = 0.8
    phase_generation_degree = 16
    terms = _pauli_string_terms_to_indexed_terms(
        _QSVT_HAMILTONIAN_SIMULATION_TERMS)
    hamiltonian = _hamiltonian_from_indexed_terms(terms)
    hamiltonian_matrix = _pauli_sum_matrix(terms, num_system)
    rng = np.random.default_rng(13)
    initial_ket = rng.normal(size=1 << num_system)
    initial_ket = (initial_ket / np.linalg.norm(initial_ket)).astype(
        np.complex128)

    encoding = algorithms.PauliLCU(hamiltonian, num_qubits=num_system)
    alpha = float(encoding.normalization)
    tau = alpha * evolution_time
    cos_phases, sin_phases, cos_info, sin_info = (
        _qsppack_hamiltonian_simulation_phases(tau, phase_generation_degree))

    assert len(cos_phases) == phase_generation_degree + 1
    assert len(sin_phases) == phase_generation_degree
    assert cos_info["value"] < 1e-12
    assert sin_info["value"] < 1e-12

    kernel_data = _kernel_data(encoding)
    initial_state = cudaq.State.from_data(initial_ket)
    cos_sequence = algorithms.qsvt.phase_sequence(
        _qsp_to_projector_phases(cos_phases))
    sin_sequence = algorithms.qsvt.phase_sequence(
        _qsp_to_projector_phases(sin_phases))
    quantum_cos_state = _run_qsvt_good_component(
        initial_state, num_system, encoding.num_ancilla,
        cos_sequence.phase_data, cos_sequence.walk_direction_data, kernel_data)
    quantum_sin_state = _run_qsvt_good_component(
        initial_state, num_system, encoding.num_ancilla,
        sin_sequence.phase_data, sin_sequence.walk_direction_data, kernel_data)

    exact_state, eigenvalues, eigenvectors = _exact_time_evolved_state(
        hamiltonian_matrix, initial_ket, evolution_time)
    cos_poly = algorithms.qsvt.phases_to_poly(
        cos_phases, algorithms.QSVTPhaseConvention.qsp)
    sin_poly = algorithms.qsvt.phases_to_poly(
        sin_phases, algorithms.QSVTPhaseConvention.qsp)
    sample_errors = []
    for scaled_eigenvalue in eigenvalues / alpha:
        walk_eigenvalue = -scaled_eigenvalue
        cos_response = cos_poly(float(walk_eigenvalue))
        sin_response = sin_poly(float(walk_eigenvalue))
        target = np.exp(-1.0j * tau * scaled_eigenvalue)
        response = 2.0 * (cos_response.real + 1.0j * sin_response.imag)
        sample_errors.append(abs(response - target))

    # With W = R_zero U, the QSP response is evaluated at -H / alpha. QSPPACK's
    # cosine target is in the real part and its sine target is in the imaginary
    # part. For this real Hamiltonian and real input state, those components can
    # be recovered from statevector simulation and combined into exp(-i H t)|psi>.
    quantum_time_evolved_state = algorithms.qsvt.recover_real_time_evolution(
        quantum_cos_state, quantum_sin_state, cos_phases, sin_phases)
    qsp_l2_error = np.linalg.norm(quantum_time_evolved_state - exact_state)
    qsp_max_error = np.max(np.abs(quantum_time_evolved_state - exact_state))
    qsp_fidelity = abs(np.vdot(exact_state, quantum_time_evolved_state))**2

    assert max(sample_errors) < 1e-10
    assert qsp_l2_error < 1e-10
    assert qsp_max_error < 1e-10
    assert qsp_fidelity == pytest.approx(1.0, abs=1e-10)


def test_qsvt_phase_sequence_helper():
    """Validate the Python-facing QSVT sequence helper.

    The public Python API should hand kernels plain phase and walk-direction
    arrays without exposing the lower-level C++ planning structs. This test
    checks default, custom, and alternating walk-direction layouts.
    """

    sequence = algorithms.qsvt.phase_sequence([0.1, -0.2, 0.3])
    assert isinstance(sequence, algorithms.qsvt.PhaseSequence)
    assert sequence.degree == 2
    assert sequence.phase_data == pytest.approx([0.1, -0.2, 0.3])
    assert sequence.walk_direction_data == [algorithms.qsvt.FORWARD] * 2
    assert sequence.kernel_data()["phases"] == pytest.approx([0.1, -0.2, 0.3])
    assert sequence.kernel_data(
    )["walk_directions"] == [algorithms.qsvt.FORWARD] * 2

    custom = algorithms.qsvt.phase_sequence(
        [0.1, -0.2, 0.3], walk_directions=["forward", "adjoint"])
    assert custom.walk_direction_data == [
        algorithms.qsvt.FORWARD, algorithms.qsvt.ADJOINT
    ]

    alternating = algorithms.qsvt.phase_sequence(
        [0.1, -0.2, 0.3, -0.4],
        walk_directions=algorithms.qsvt.alternating_walk_directions(
            3, first="adjoint"))
    assert alternating.walk_direction_data == [
        algorithms.qsvt.ADJOINT, algorithms.qsvt.FORWARD,
        algorithms.qsvt.ADJOINT
    ]

    qsp_sequence = algorithms.qsvt.phase_sequence([0.1, -0.2],
                                                  convention="qsp")
    assert qsp_sequence.convention == algorithms.qsvt.PhaseConvention.qsp
    assert algorithms.qsvt.projector_phases_from_qsp(
        [0.1, -0.2]) == pytest.approx([0.2, -0.4])

    cos_state = np.asarray([1.0 + 0.5j, -0.25j], dtype=np.complex128)
    sin_state = np.asarray([0.25 - 0.5j, 0.75j], dtype=np.complex128)
    cos_qsp = [0.1, -0.2]
    sin_qsp = [0.3]
    expected = 2.0 * (
        (cos_state * np.exp(-1.0j * np.sum(cos_qsp))).real + 1.0j *
        (sin_state * np.exp(-1.0j * np.sum(sin_qsp))).imag)
    recovered = algorithms.qsvt.recover_real_time_evolution(
        cos_state, sin_state, cos_qsp, sin_qsp)
    assert np.allclose(recovered, expected)

    hamiltonian = 0.6 * spin.x(0) + 0.8 * spin.z(0)
    encoding = algorithms.PauliLCU(hamiltonian, num_qubits=1)
    args = algorithms.qsvt.pauli_lcu_kernel_args(sequence,
                                                 encoding.kernel_data())
    assert len(args) == 7
    assert args[0] == pytest.approx(sequence.phase_data)
    assert args[1] == sequence.walk_direction_data
    assert args[2] == pytest.approx(list(encoding.get_angles()))
    assert args[3] == list(encoding.get_term_controls())
    assert args[4] == list(encoding.get_term_ops())
    assert args[5] == list(encoding.get_term_lengths())
    assert args[6] == list(encoding.get_term_signs())

    legacy_args = algorithms.qsvt.pauli_lcu_kernel_args(
        sequence, _kernel_data(encoding))
    assert legacy_args[0] == pytest.approx(args[0])
    assert legacy_args[1] == args[1]
    assert legacy_args[2] == pytest.approx(args[2])
    assert legacy_args[3] == args[3]
    assert legacy_args[4] == args[4]


def test_qsvt_python_layer_hides_cpp_planning_objects():
    """Keep C++ QSVT plan/descriptor types out of the public Python surface."""

    hidden_names = [
        "QSVTWalkDirection",
        "QSVTTransformKind",
        "QSVTResponse",
        "QSVTResponseError",
        "QSVTPhaseSequence",
        "QSVTSequencePolicy",
        "QSVTPlan",
        "QSVTTransformDescriptor",
        "QSVTTransformPlan",
        "make_qsvt_phase_sequence",
        "make_qsvt_sequence_policy",
        "make_custom_qsvt_sequence_policy",
        "make_alternating_qsvt_sequence_policy",
        "make_qsvt_plan",
        "make_qsvt_transform_plan",
        "make_linear_solve_qsvt_transform",
        "make_real_time_hamiltonian_simulation_qsvt_transform",
        "make_imaginary_time_hamiltonian_simulation_qsvt_transform",
    ]
    for name in hidden_names:
        assert not hasattr(algorithms, name)


def test_qsvt_response_evaluation_and_error_estimation():
    """Check classical phase-to-polynomial diagnostics used by QSVT tests."""

    poly = algorithms.qsvt.phases_to_poly([0.0, 0.0])
    response = poly(0.5)
    assert isinstance(response, complex)
    assert response.real == pytest.approx(0.5)
    assert response.imag == pytest.approx(0.0)

    qsp_sequence = algorithms.qsvt.phase_sequence([0.1, -0.2],
                                                  convention="qsp")
    qsp_poly = algorithms.qsvt.phases_to_poly(qsp_sequence)
    qsp_response = qsp_poly(0.5)
    assert isinstance(qsp_response, complex)

    sample_points = algorithms.make_uniform_qsvt_sample_points(-1.0, 1.0, 5)
    assert sample_points == pytest.approx([-1.0, -0.5, 0.0, 0.5, 1.0])
    chebyshev_points = algorithms.make_chebyshev_qsvt_sample_points(
        -1.0, 1.0, 3)
    assert chebyshev_points == pytest.approx([-1.0, 0.0, 1.0])

    error = algorithms.qsvt.estimate_poly_error(poly,
                                                lambda x: complex(x, 0.0),
                                                domain=(-1.0, 1.0),
                                                num_points=5)
    assert isinstance(error, algorithms.qsvt.PolyError)
    assert error.max_abs_error == pytest.approx(0.0, abs=1e-12)
    assert error.rms_error == pytest.approx(0.0, abs=1e-12)
    assert error.num_samples == 5


def test_qsvt_validation_errors():
    """Confirm invalid Python-facing QSVT inputs are rejected."""

    with pytest.raises(ValueError):
        algorithms.qsvt.phase_sequence([])
    with pytest.raises(ValueError):
        algorithms.qsvt.phase_sequence([0.1, 0.2], walk_directions=[0, 1])
    with pytest.raises(ValueError):
        algorithms.qsvt.phase_sequence([0.1, 0.2],
                                       walk_directions=["sideways"])
    with pytest.raises(ValueError):
        algorithms.qsvt.phase_sequence([0.1, 0.2], convention="phaseish")
    with pytest.raises(ValueError):
        algorithms.qsvt.forward_walk_directions(-1)
    with pytest.raises(ValueError):
        algorithms.qsvt.alternating_walk_directions(1, first="sideways")
