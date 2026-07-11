# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Tests for the Suzuki-Trotter module.

Product-formula correctness is pinned against independent dense references
(matrix exponentials and an explicit Pauli-rotation simulator), and the
host-side extraction, planning, and resource machinery is covered for
every accepted input form.
"""

import numpy as np
import pytest

import cudaq
from cudaq import spin

from cudaq_algorithms import sim_utils, trotter

from dense_references import dense_matrix, random_ket

FOREST_RUTH_W1 = 1.3512071919596578
FOREST_RUTH_W0 = -1.7024143839193153

# ----------------------------------------------------------------------------
# Dense references (ported verbatim from the upstream test file)
# ----------------------------------------------------------------------------


def _apply_pauli_to_vector(word, vector):
    word = str(word)
    result = np.zeros_like(vector, dtype=np.complex128)
    for basis, amplitude in enumerate(vector):
        target = basis
        phase = 1.0 + 0.0j
        for qubit, op in enumerate(word):
            bit = (basis >> qubit) & 1
            if op == "I":
                continue
            if op == "X":
                target ^= 1 << qubit
            elif op == "Y":
                target ^= 1 << qubit
                phase *= -1.0j if bit else 1.0j
            elif op == "Z":
                phase *= -1.0 if bit else 1.0
            else:
                raise ValueError(op)
        result[target] += phase * amplitude
    return result


def _apply_pauli_rotation(vector, word, angle):
    return (np.cos(angle) * vector -
            1.0j * np.sin(angle) * _apply_pauli_to_vector(word, vector))


def _second_order_step(vector, coefficients, words, tau):
    state = vector
    for coefficient, word in zip(coefficients, words):
        state = _apply_pauli_rotation(state, word, 0.5 * tau * coefficient)
    for coefficient, word in reversed(list(zip(coefficients, words))):
        state = _apply_pauli_rotation(state, word, 0.5 * tau * coefficient)
    return state


def _simulate_trotter(coefficients,
                      words,
                      identity,
                      num_qubits,
                      time,
                      steps,
                      order,
                      ket,
                      include_identity=True):
    state = np.array(ket, dtype=np.complex128, copy=True)
    dt = time / steps
    for _ in range(steps):
        if order == 1:
            for coefficient, word in zip(coefficients, words):
                state = _apply_pauli_rotation(state, word, dt * coefficient)
        elif order == 2:
            state = _second_order_step(state, coefficients, words, dt)
        else:
            state = _second_order_step(state, coefficients, words,
                                       FOREST_RUTH_W1 * dt)
            state = _second_order_step(state, coefficients, words,
                                       FOREST_RUTH_W0 * dt)
            state = _second_order_step(state, coefficients, words,
                                       FOREST_RUTH_W1 * dt)
    if include_identity and identity != 0.0:
        state *= np.exp(-1.0j * identity * time)
    return state


def _exact_evolve(coefficients, words, identity, time, ket):
    # Dense Pauli-sum reference shared with the other suites.
    matrix = identity * np.eye(ket.size, dtype=np.complex128)
    if words:
        matrix = matrix + dense_matrix(list(zip(coefficients, words)),
                                       len(str(words[0])))
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return eigenvectors @ (np.exp(-1.0j * time * eigenvalues) *
                           (eigenvectors.conj().T @ ket))


def _two_qubit_product_state(rx_angle, ry_angle):
    q0 = np.array([np.cos(0.5 * rx_angle), -1.0j * np.sin(0.5 * rx_angle)],
                  dtype=np.complex128)
    q1 = np.array([np.cos(0.5 * ry_angle),
                   np.sin(0.5 * ry_angle)],
                  dtype=np.complex128)
    return np.array(
        [q0[basis & 1] * q1[(basis >> 1) & 1] for basis in range(4)],
        dtype=np.complex128)


@cudaq.kernel
def _evolve_prepared(coeffs: list[float], paulis: list[cudaq.pauli_word],
                     t: float, n_steps: int, order: int, theta0: float,
                     theta1: float):
    q = cudaq.qvector(2)
    rx(theta0, q[0])
    ry(theta1, q[1])
    trotter.apply_trotter(coeffs, paulis, t, n_steps, order, q)


def _phase_align_error(actual, expected):
    overlap = np.vdot(expected, actual)
    if abs(overlap) > 0.0:
        actual = actual * np.exp(-1.0j * np.angle(overlap))
    return np.linalg.norm(actual - expected)


# ----------------------------------------------------------------------------
# Term extraction
# ----------------------------------------------------------------------------


def test_make_trotter_terms_extracts_spin_operator():
    hamiltonian = (0.7 * spin.x(0) + 0.4 * spin.z(1) -
                   0.2 * cudaq.SpinOperator.from_word("II"))
    coefficients, words, identity, num_qubits = trotter.make_trotter_terms(
        hamiltonian)

    by_word = {
        str(word): coefficient
        for coefficient, word in zip(coefficients, words)
    }
    assert num_qubits == 2
    assert identity == pytest.approx(-0.2)
    assert by_word["XI"] == pytest.approx(0.7)
    assert by_word["IZ"] == pytest.approx(0.4)


def test_make_trotter_terms_accepts_single_spin_term():
    coefficients, words, identity, num_qubits = trotter.make_trotter_terms(
        0.5 * spin.y(2))
    assert coefficients == pytest.approx([0.5])
    assert [str(word) for word in words] == ["IIY"]
    assert identity == pytest.approx(0.0)
    assert num_qubits == 3


def test_make_trotter_terms_accepts_dict_and_pairs():
    from_dict = trotter.make_trotter_terms({"XI": 0.7, "IZ": 0.4, "II": -0.2})
    from_pairs = trotter.make_trotter_terms([(0.7, "XI"), (0.4, "IZ"),
                                             (-0.2, "II")])
    for coefficients, words, identity, num_qubits in (from_dict, from_pairs):
        assert num_qubits == 2
        assert identity == pytest.approx(-0.2)
        assert coefficients == pytest.approx([0.7, 0.4])
        assert [str(w) for w in words] == ["XI", "IZ"]


def test_make_trotter_terms_validation():
    with pytest.raises(ValueError, match="non-negative"):
        trotter.make_trotter_terms(spin.x(0), coefficient_tolerance=-1.0)
    with pytest.raises(ValueError, match="complex"):
        trotter.make_trotter_terms({"X": 0.5 + 0.3j})
    with pytest.raises(ValueError, match="complex"):
        trotter.make_trotter_terms([(0.5j, "X")])
    with pytest.raises(ValueError, match="complex"):
        trotter.make_trotter_terms(0.5j * spin.x(0))
    with pytest.raises(ValueError, match="same length"):
        trotter.make_trotter_terms({"XI": 0.5, "X": 0.3})
    with pytest.raises(ValueError, match="unsupported Pauli"):
        trotter.make_trotter_terms({"XQ": 0.5})
    with pytest.raises(TypeError):
        trotter.make_trotter_terms(42)


# ----------------------------------------------------------------------------
# Product-formula reference sanity
# ----------------------------------------------------------------------------


def test_product_formula_reference_improves_with_order():
    hamiltonian = (0.7 * spin.x(0) + 0.4 * spin.z(1) +
                   0.31 * spin.x(0) * spin.z(1) + 0.23 * spin.y(0) * spin.y(1))
    coefficients, words, identity, num_qubits = trotter.make_trotter_terms(
        hamiltonian)

    rng = np.random.default_rng(7)
    ket = rng.normal(size=4) + 1.0j * rng.normal(size=4)
    ket = ket / np.linalg.norm(ket)

    time, steps = 0.8, 2
    exact = _exact_evolve(coefficients, words, identity, time, ket)
    errors = {
        order:
        _phase_align_error(
            _simulate_trotter(coefficients, words, identity, num_qubits, time,
                              steps, order, ket), exact)
        for order in (1, 2, 4)
    }
    assert errors[2] < errors[1]
    assert errors[4] < errors[2]


# ----------------------------------------------------------------------------
# Term ordering, validation, resources
# ----------------------------------------------------------------------------


def test_trotter_orders_terms_and_estimates_resources():
    hamiltonian = (0.1 * spin.x(0) + 0.7 * spin.z(1) +
                   0.4 * spin.x(0) * spin.z(1) -
                   0.2 * cudaq.SpinOperator.from_word("II"))
    evolution = trotter.Trotter(
        hamiltonian,
        ordering=trotter.TrotterOrdering.COEFFICIENT_MAGNITUDE_DESCENDING)

    assert evolution.identity_coefficient == pytest.approx(-0.2)
    assert evolution.coefficients == pytest.approx([0.7, 0.4, 0.1])
    assert [str(word) for word in evolution.words] == ["IZ", "XZ", "XI"]

    resources = evolution.resources(steps=3, order=4)
    assert resources.num_terms == 3
    assert resources.steps == 3
    assert resources.order == 4
    assert resources.pauli_rotations == 3 * 3 * 6
    assert resources.estimated_cx_count == 2 * 3 * 6


def test_trotter_rejects_invalid_options():
    evolution = trotter.Trotter(spin.x(0))
    with pytest.raises(ValueError, match="steps"):
        evolution.kernel(time=0.1, steps=0)
    with pytest.raises(ValueError, match="order"):
        evolution.kernel(time=0.1, order=3)
    with pytest.raises(ValueError, match="unsupported"):
        trotter.Trotter(spin.x(0), ordering="bogus")
    with pytest.raises(ValueError, match="finite"):
        evolution.kernel(time=float("nan"))


def test_estimate_trotter_resources_accepts_flattened_terms():
    coefficients, words, identity, _ = trotter.make_trotter_terms(
        0.5 * spin.x(0) * spin.y(1) + 0.25 * spin.z(0))
    resources = trotter.estimate_trotter_resources(
        coefficients, words, steps=2, order=2, identity_coefficient=identity)
    assert resources.num_terms == 2
    assert resources.pauli_rotations == 8
    assert resources.estimated_cx_count == 8


# ----------------------------------------------------------------------------
# Kernel behavior (escape hatch: apply_trotter inside a user kernel)
# ----------------------------------------------------------------------------


def test_apply_trotter_kernel_interop_with_flattened_terms():
    coefficients, words, identity, num_qubits = trotter.make_trotter_terms(
        spin.x(0))
    assert identity == 0.0
    assert num_qubits == 1

    @cudaq.kernel
    def evolve(coeffs: list[float], paulis: list[cudaq.pauli_word], t: float):
        q = cudaq.qvector(1)
        trotter.apply_trotter(coeffs, paulis, t, 1, 2, q)

    state = np.asarray(cudaq.get_state(evolve, coefficients, words, 0.25),
                       dtype=np.complex128)
    expected = _simulate_trotter(coefficients, words, identity, num_qubits,
                                 0.25, 1, 2,
                                 np.array([1.0, 0.0], dtype=np.complex128))
    np.testing.assert_allclose(state, expected, atol=1e-6)


def test_apply_trotter_kernel_invalid_inputs_are_noops():
    coefficients, words, _, _ = trotter.make_trotter_terms(spin.x(0))

    @cudaq.kernel
    def evolve(coeffs: list[float], paulis: list[cudaq.pauli_word], steps: int,
               order: int):
        q = cudaq.qvector(1)
        trotter.apply_trotter(coeffs, paulis, 0.25, steps, order, q)

    expected = np.array([1.0, 0.0], dtype=np.complex128)
    for bad_coefficients, bad_words, bad_steps, bad_order in (
        (coefficients, words, 0, 2),
        (coefficients, words, 1, 3),
        (coefficients, [], 1, 2),
    ):
        state = np.asarray(cudaq.get_state(evolve, bad_coefficients, bad_words,
                                           bad_steps, bad_order),
                           dtype=np.complex128)
        np.testing.assert_allclose(state, expected, atol=1e-12)


def test_apply_trotter_kernel_matches_reference_for_orders():
    hamiltonian = (0.7 * spin.x(0) + 0.4 * spin.z(1) +
                   0.31 * spin.x(0) * spin.z(1) + 0.23 * spin.y(0) * spin.y(1))
    coefficients, words, identity, num_qubits = trotter.make_trotter_terms(
        hamiltonian)
    ket = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)

    @cudaq.kernel
    def evolve(coeffs: list[float], paulis: list[cudaq.pauli_word], t: float,
               steps: int, order: int):
        q = cudaq.qvector(2)
        trotter.apply_trotter(coeffs, paulis, t, steps, order, q)

    for order in (1, 2, 4):
        state = np.asarray(cudaq.get_state(evolve, coefficients, words, 0.8, 3,
                                           order),
                           dtype=np.complex128)
        expected = _simulate_trotter(coefficients, words, identity, num_qubits,
                                     0.8, 3, order, ket)
        np.testing.assert_allclose(state, expected, atol=1e-6)


def test_apply_trotter_kernel_error_scaling_tracks_order():
    """Order-p Trotter error must scale ~ dt**p (fitted slope ~ -p)."""
    hamiltonian = (0.37 * spin.x(0) - 0.22 * spin.z(1) +
                   0.19 * spin.x(0) * spin.x(1) +
                   0.41 * spin.y(0) * spin.y(1) + 0.13 * spin.z(0) * spin.x(1))
    coefficients, words, identity, num_qubits = trotter.make_trotter_terms(
        hamiltonian)

    time = 0.7
    rx_angle, ry_angle = 0.37, -0.52
    ket = _two_qubit_product_state(rx_angle, ry_angle)
    exact = _exact_evolve(coefficients, words, identity, time, ket)

    step_counts = [1, 2, 4]
    log_steps = np.log(np.array(step_counts, dtype=float))
    error_at_four_steps = {}
    for order in (1, 2, 4):
        errors = []
        for steps in step_counts:
            state = np.asarray(cudaq.get_state(_evolve_prepared, coefficients,
                                               words, time, steps, order,
                                               rx_angle, ry_angle),
                               dtype=np.complex128)
            errors.append(_phase_align_error(state, exact))
        error_at_four_steps[order] = errors[-1]
        slope = np.polyfit(log_steps, np.log(np.array(errors)), 1)[0]
        assert slope == pytest.approx(
            -order,
            abs=0.4), (f"order {order}: fitted scaling slope {slope:.3f} "
                       f"is not close to -{order}")

    # Fixed-tolerance checks at steps=4 (folded from a separate test that
    # re-ran these exact launches): higher order strictly wins.
    assert error_at_four_steps[2] < error_at_four_steps[1]
    assert error_at_four_steps[4] < error_at_four_steps[2]
    assert error_at_four_steps[1] < 2.0e-2
    assert error_at_four_steps[2] < 6.0e-4
    assert error_at_four_steps[4] < 5.0e-6


def test_apply_trotter_kernel_exact_for_commuting_hamiltonian():
    """All-Z terms commute: every order/step count must be exact."""
    hamiltonian = (0.5 * spin.z(0) + 0.4 * spin.z(1) +
                   0.3 * spin.z(0) * spin.z(1))
    coefficients, words, identity, num_qubits = trotter.make_trotter_terms(
        hamiltonian)
    assert identity == pytest.approx(0.0)

    time = 0.7
    rx_angle, ry_angle = 0.41, -0.33
    ket = _two_qubit_product_state(rx_angle, ry_angle)
    exact = _exact_evolve(coefficients, words, identity, time, ket)

    for order in (1, 2, 4):
        for steps in (1, 3):
            state = np.asarray(cudaq.get_state(_evolve_prepared, coefficients,
                                               words, time, steps, order,
                                               rx_angle, ry_angle),
                               dtype=np.complex128)
            assert _phase_align_error(state, exact) < 1.0e-9


def test_apply_trotter_kernel_handles_four_qubit_hamiltonian_with_many_terms():
    hamiltonian = (0.11 * spin.x(0) - 0.17 * spin.y(1) + 0.23 * spin.z(2) -
                   0.29 * spin.x(3) + 0.31 * spin.x(0) * spin.x(1) +
                   0.37 * spin.y(1) * spin.z(2) -
                   0.41 * spin.z(0) * spin.x(3) +
                   0.43 * spin.x(0) * spin.y(2) * spin.z(3) -
                   0.47 * spin.y(0) * spin.y(1) * spin.x(2) +
                   0.53 * spin.z(0) * spin.x(1) * spin.y(2) * spin.z(3))
    coefficients, words, identity, num_qubits = trotter.make_trotter_terms(
        hamiltonian)
    assert num_qubits == 4
    assert len(coefficients) > 8
    assert len(coefficients) == len(words)

    ket = np.zeros(16, dtype=np.complex128)
    ket[0] = 1.0

    @cudaq.kernel
    def evolve(coeffs: list[float], paulis: list[cudaq.pauli_word], t: float,
               steps: int, order: int):
        q = cudaq.qvector(4)
        trotter.apply_trotter(coeffs, paulis, t, steps, order, q)

    state = np.asarray(cudaq.get_state(evolve, coefficients, words, 0.37, 2,
                                       2),
                       dtype=np.complex128)
    expected = _simulate_trotter(coefficients, words, identity, num_qubits,
                                 0.37, 2, 2, ket)
    np.testing.assert_allclose(state, expected, atol=1e-6)


# ----------------------------------------------------------------------------
# Kernel factory and simulation-helper evolution
# ----------------------------------------------------------------------------


@cudaq.kernel
def _product_prep(qubits: cudaq.qview):
    rx(0.37, qubits[0])
    ry(-0.52, qubits[1])


def test_kernel_factory_accepts_state_prep_injection():
    # With state_prep the factory returns a zero-argument, fully
    # hardware-shaped circuit; pin it against state_kernel fed the same
    # prepared state as data.
    evolution = trotter.Trotter({"XI": 0.7, "IZ": 0.4, "XZ": 0.31})
    ket = _two_qubit_product_state(0.37, -0.52)

    via_prep = np.asarray(cudaq.get_state(
        evolution.kernel(time=0.8, steps=3, order=2,
                         state_prep=_product_prep)),
                          dtype=np.complex128)
    via_state = np.asarray(cudaq.get_state(
        evolution.state_kernel(time=0.8, steps=3, order=2),
        sim_utils.state_from(ket)),
                           dtype=np.complex128)
    np.testing.assert_allclose(via_prep, via_state, atol=1e-12)

    counts = cudaq.sample(evolution.kernel(time=0.8,
                                           steps=1,
                                           state_prep=_product_prep),
                          shots_count=100)
    assert sum(counts.values()) == 100


def test_kernel_factory_evolves_the_zero_state():
    hamiltonian = {"XI": 0.7, "IZ": 0.4, "XZ": 0.31, "YY": 0.23}
    evolution = trotter.Trotter(hamiltonian)

    ket0 = np.zeros(4, dtype=np.complex128)
    ket0[0] = 1.0
    factory_state = np.asarray(cudaq.get_state(
        evolution.kernel(time=0.8, steps=3, order=2)),
                               dtype=np.complex128)
    expected = _simulate_trotter(evolution.coefficients, evolution.words, 0.0,
                                 evolution.num_qubits, 0.8, 3, 2, ket0)
    np.testing.assert_allclose(factory_state, expected, atol=1e-6)


def test_sim_utils_evolve_includes_identity_phase():
    hamiltonian = {"XI": 0.7, "IZ": 0.4, "II": -0.2}
    evolution = trotter.Trotter(hamiltonian)

    ket = random_ket(2, seed=3)

    exact = _exact_evolve(evolution.coefficients, evolution.words,
                          evolution.identity_coefficient, 0.8, ket)
    evolved = sim_utils.evolve(evolution, ket, time=0.8, steps=64, order=2)
    # Direct comparison, NOT phase-aligned: evolve() reintroduces the
    # identity phase the circuit primitive omits.
    assert np.linalg.norm(evolved - exact) < 1e-3

    without_phase = sim_utils.evolve(evolution,
                                     ket,
                                     time=0.8,
                                     steps=64,
                                     order=2,
                                     include_identity_phase=False)
    assert np.linalg.norm(without_phase - exact) > 0.1


def test_identity_only_hamiltonian_is_a_global_phase():
    evolution = trotter.Trotter({"II": -0.2})
    assert evolution.num_terms == 0

    ket = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.complex128)
    evolved = sim_utils.evolve(evolution, ket, time=0.5, steps=2)
    np.testing.assert_allclose(evolved, np.exp(0.1j) * ket, atol=1e-12)
    np.testing.assert_allclose(sim_utils.evolve(evolution,
                                                ket,
                                                time=0.5,
                                                steps=2,
                                                include_identity_phase=False),
                               ket,
                               atol=1e-12)


# ----------------------------------------------------------------------------
# Input forms, degenerate inputs, and validation (public API)
# ----------------------------------------------------------------------------


def test_generator_input_normalizes_like_pairs():
    via_list = trotter.make_trotter_terms([(0.7, "XI"), (0.4, "IZ")])
    via_generator = trotter.make_trotter_terms(
        (c, w) for c, w in [(0.7, "XI"), (0.4, "IZ")])
    assert via_generator == via_list


def test_spin_operator_padding_uses_widest_term():
    coefficients, words, _, num_qubits = trotter.make_trotter_terms(
        0.3 * spin.x(0) + 0.2 * spin.z(3))
    assert num_qubits == 4
    assert sorted(words) == ["IIIZ", "XIII"]


def test_complex_noise_tolerance_matches_pauli_lcu():
    # One validation across the package: the same imaginary-noise
    # tolerance accepts this input in both primitives.
    from cudaq_algorithms import PauliLCU

    coefficients, words, _, _ = trotter.make_trotter_terms({"X": 0.5 + 1e-11j})
    assert coefficients == [pytest.approx(0.5)] and words == ["X"]
    assert PauliLCU({"X": 0.5 + 1e-11j}).alpha == pytest.approx(0.5)


def test_empty_hamiltonian_rejected():
    # Regression: an accepted empty Hamiltonian produced a 0-qubit kernel
    # whose launch crashed the interpreter.
    with pytest.raises(ValueError, match="no terms"):
        trotter.Trotter({})
    with pytest.raises(ValueError, match="no terms"):
        trotter.Trotter([])
    with pytest.raises(ValueError, match="at least one qubit"):
        trotter.Trotter({"": 0.5})


def test_string_hamiltonian_rejected_with_type_error():
    with pytest.raises(TypeError, match="spin operator"):
        trotter.Trotter("XZ")
    with pytest.raises(TypeError, match="spin operator"):
        trotter.Trotter(bytearray(b"XZ"))


def test_zero_coefficient_terms_dropped():
    coefficients, words, _, _ = trotter.make_trotter_terms({
        "XI": 0.0,
        "IZ": 0.4
    })
    assert words == ["IZ"]
    # Exactly-zero terms are dropped even when the tolerance filter is
    # disabled entirely.
    _, words_no_tolerance, _, _ = trotter.make_trotter_terms(
        {
            "XI": 0.0,
            "IZ": 0.4
        }, coefficient_tolerance=0.0)
    assert words_no_tolerance == ["IZ"]
    resources = trotter.Trotter({
        "XI": 0.0,
        "IZ": 0.4
    }).resources(steps=2, order=2)
    assert resources.num_terms == 1
    assert resources.pauli_rotations == 4


def test_kernel_rejects_fractional_steps():
    # Parity with Walk.kernel(power=...): no silent truncation.
    evolution = trotter.Trotter({"XI": 0.7, "IZ": 0.4})
    with pytest.raises(ValueError, match="steps"):
        evolution.kernel(time=0.5, steps=2.9)
    with pytest.raises(ValueError, match="steps"):
        evolution.state_kernel(time=0.5, steps=2.9)


def test_resources_requires_explicit_parameters():
    # No defaults: the estimate can never silently describe a different
    # circuit than the kernel that was built.
    evolution = trotter.Trotter({"XI": 0.7, "IZ": 0.4})
    with pytest.raises(TypeError):
        evolution.resources()


def test_sim_utils_evolve_validates_parameters():
    # Regression: evolve() previously coerced silently and the kernel
    # guard turned invalid parameters into a no-op circuit, returning the
    # unevolved state as if it were the evolution result.
    evolution = trotter.Trotter({"XI": 0.7, "IZ": 0.4})
    ket = random_ket(2, seed=5)
    with pytest.raises(ValueError, match="steps"):
        sim_utils.evolve(evolution, ket, time=0.8, steps=0)
    with pytest.raises(ValueError, match="order"):
        sim_utils.evolve(evolution, ket, time=0.8, order=3)
    with pytest.raises(ValueError, match="finite"):
        sim_utils.evolve(evolution, ket, time=float("nan"))
    with pytest.raises(ValueError, match="dimension"):
        sim_utils.evolve(evolution, np.ones(8) / np.sqrt(8), time=0.8)
    with pytest.raises(ValueError, match="1-D"):
        sim_utils.evolve(evolution,
                         np.eye(2, dtype=np.complex128) / np.sqrt(2),
                         time=0.8)
