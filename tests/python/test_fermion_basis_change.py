# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public planning and encoding contracts, independent of prototype core gates."""

import copy
import decimal
import math
import pickle
from dataclasses import FrozenInstanceError

import cudaq
import numpy as np
import pytest

import cudaq_algorithms as algorithms
from cudaq_algorithms import state_from


def _api():
    assert hasattr(algorithms, "fermion_basis_change"), (
        "fermion_basis_change must be exported outside stateprep")
    return algorithms.fermion_basis_change


def _encoding(width):
    _api()
    from cudaq_algorithms.basis_change import AngleEncoding
    return AngleEncoding(width)


def _structure(kind, num_modes, **kwargs):
    _api()
    from cudaq_algorithms.basis_change import GivensStructure
    return getattr(GivensStructure, kind)(num_modes, **kwargs)


def _fock_matrix(physical):
    """Independent determinant minors, with qubit zero least significant."""
    physical = np.asarray(physical, dtype=complex)
    occupied = [[p for p in range(len(physical)) if bits >> p & 1]
                for bits in range(1 << len(physical))]
    result = np.zeros((len(occupied), len(occupied)), dtype=complex)
    for row, rows in enumerate(occupied):
        for col, cols in enumerate(occupied):
            if len(rows) == len(cols):
                result[row, col] = np.linalg.det(physical[np.ix_(rows, cols)])
    return result


def _embedded(orbital, mode_map):
    """Place logical modes on their physical qubits; spectators stay fixed."""
    physical = np.eye(max(mode_map) + 1, dtype=complex)
    physical[np.ix_(mode_map, mode_map)] = orbital
    return physical


def _creation(qubit, size):
    """Jordan-Wigner creation operator with parity on every lower qubit."""
    result = np.zeros((1 << size, 1 << size), dtype=complex)
    for bits in range(1 << size):
        if not bits >> qubit & 1:
            parity = (bits & ((1 << qubit) - 1)).bit_count()
            result[bits | 1 << qubit, bits] = (-1)**parity
    return result


def _random_unitary(size, seed):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    unitary, upper = np.linalg.qr(data)
    return unitary * (np.diag(upper) / abs(np.diag(upper)))


def _random_state(size, seed):
    rng = np.random.default_rng(seed)
    state = rng.normal(size=1 << size) + 1j * rng.normal(size=1 << size)
    return state / np.linalg.norm(state)


def _evolve(operation, *initial_states):
    """Apply a register kernel to complete initial states on the target."""

    @cudaq.kernel
    def run(state: cudaq.State):
        register = cudaq.qvector(state)
        operation(register)

    return [
        np.asarray(
            cudaq.get_state(run, state_from(np.ascontiguousarray(initial))))
        for initial in initial_states
    ]


def _matrix(operation, size):
    return np.column_stack(
        _evolve(operation, *np.eye(1 << size, dtype=complex)))


def _two_mode_full_orbital(values):
    """Closed form P(phi) G(theta) P(-phi) D(a, b) in full slot order."""
    a, b, theta, phi = values
    c, s, e = np.cos(theta), np.sin(theta), np.exp(1j * phi)
    mixing = np.array([[c, -s * np.conj(e)], [s * e, c]])
    return mixing @ np.diag(np.exp(1j * np.array([a, b])))


def test_root_api_full_and_partial_preserve_phases():
    factory = _api()
    matrix = np.array([[1., -1.], [1.j, 1.j]]) / np.sqrt(2)
    plan = factory(matrix)
    np.testing.assert_allclose(plan.schedule.orbital_matrix(),
                               matrix,
                               atol=1e-14)
    vector = np.array([1.j, -1.]) / np.sqrt(2)
    partial = factory(vector)
    column = factory(vector[:, None])
    np.testing.assert_allclose(partial.schedule.orbital_matrix()[:, 0],
                               vector,
                               atol=1e-14)
    assert partial.schedule == column.schedule
    assert plan.num_qubits == 2
    assert not plan.is_programmable
    assert plan.angle_bits is None
    assert plan.encoded_words is None
    assert plan.represented_parameters is None
    assert plan.quantization_error is None
    assert plan.quantization_error_bound is None


def test_public_classical_kernels_match_fock_operator_and_control():
    unitary = _random_unitary(3, seed=31)
    plan = _api()(unitary)
    forward = _fock_matrix(unitary)
    np.testing.assert_allclose(_matrix(plan.kernel(), 3),
                               forward,
                               atol=1e-12,
                               rtol=0)
    np.testing.assert_allclose(_matrix(plan.adjoint_kernel(), 3),
                               forward.conj().T,
                               atol=1e-12,
                               rtol=0)
    for adjoint, block in ((False, forward), (True, forward.conj().T)):
        # Enable on qubit zero: identity when off, the base map when on.
        expected = np.eye(16, dtype=complex)
        expected[1::2, 1::2] = block
        actual = _matrix(plan.controlled_kernel(adjoint=adjoint), 4)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("reference_mode", [0, 1, 2])
def test_numeric_basis_keeps_explicit_partial_structure(reference_mode):
    rng = np.random.default_rng(99)
    vector = rng.normal(size=3) + 1j * rng.normal(size=3)
    vector /= np.linalg.norm(vector)
    mode_map = (3, 0, 1)
    structure = _structure("partial",
                           3,
                           reference_mode=reference_mode,
                           mode_map=mode_map)
    plan = _api()(vector, structure=structure)
    assert plan.structure == structure
    actual = _matrix(plan.kernel(), 4)
    # W a_r^dagger W^dagger = sum_p v_p a_p^dagger on physical qubits.
    expected = sum(value * _creation(qubit, 4)
                   for qubit, value in zip(mode_map, vector))
    np.testing.assert_allclose(
        actual @ _creation(mode_map[reference_mode], 4) @ actual.conj().T,
        expected,
        atol=1e-12,
        rtol=0)
    np.testing.assert_allclose(actual[:, 0],
                               np.eye(16)[:, 0],
                               atol=1e-12,
                               rtol=0)


def test_numeric_basis_keeps_explicit_full_mode_map():
    unitary = _random_unitary(3, seed=21)
    mode_map = (4, 0, 2)
    structure = _structure("full", 3, mode_map=mode_map)
    plan = _api()(unitary, structure=structure)
    assert plan.structure == structure
    assert plan.num_qubits == 5
    np.testing.assert_allclose(_matrix(plan.kernel(), 5),
                               _fock_matrix(_embedded(unitary, mode_map)),
                               atol=1e-12,
                               rtol=0)


def test_classical_span_includes_spectator_for_controlled_kernel():
    unitary = _random_unitary(2, seed=55)
    plan = _api()(unitary, structure=_structure("full", 2, mode_map=(0, 2)))
    assert plan.num_qubits == 3
    # Exactly num_qubits + 1 qubits: enable first, then the spectator span.
    expected = np.eye(16, dtype=complex)
    expected[1::2, 1::2] = _fock_matrix(_embedded(unitary, (0, 2)))
    actual = _matrix(plan.controlled_kernel(), plan.num_qubits + 1)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def test_program_plan_reports_slot_order_codes_and_registers():
    matrix = np.array([[1., -1.], [1.j, 1.j]]) / np.sqrt(2)
    plan = _api()(matrix, programmable=True, angle_bits=3)
    assert plan.is_programmable
    assert plan.angle_bits == 3
    assert plan.angle_encoding.width == 3
    assert plan.angle_encoding.bit_order == "little"
    assert plan.angle_encoding.step == math.pi / 4
    assert plan.angle_encoding.weights == (math.pi / 4, math.pi / 2, math.pi)
    assert plan.schedule.parameter_names == ("diagonal_phase[0]",
                                             "diagonal_phase[1]", "theta[0]",
                                             "phi[0]")
    assert plan.encoded_words == (0, 2, 1, 2)
    np.testing.assert_allclose(plan.represented_parameters,
                               (0, math.pi / 2, math.pi / 4, math.pi / 2))
    assert plan.parameter_registers == ((2, 3, 4), (5, 6, 7), (8, 9, 10),
                                        (11, 12, 13))
    assert plan.num_qubits == 14
    assert plan.quantization_error_bound < 1e-14


def test_program_plan_does_not_need_companion_core(monkeypatch):
    factory = _api()
    for name in ("AngleFormat", "programmable_rx", "programmable_ry",
                 "programmable_rz", "programmable_phase"):
        monkeypatch.delattr(cudaq, name, raising=False)
    plan = factory(np.eye(2), programmable=True, angle_bits=1)
    kernel = plan.kernel()
    assert "programmable_r" not in str(kernel)
    assert plan.encoded_words == (0, 0, 0, 0)


def test_structure_only_plan_has_layout_without_numerical_reference():
    factory = _api()
    from cudaq_algorithms.basis_change import GivensStructure
    structure = GivensStructure.partial(3, reference_mode=1)
    plan = factory(structure=structure, programmable=True, angle_bits=2)
    assert plan.num_qubits == 13
    # Slots keep schedule order: two mixing angles, then three phases.
    words = ((3, 4), (5, 6), (7, 8), (9, 10), (11, 12))
    assert plan.parameter_registers == words
    assert plan.schedule.parameters is None
    assert plan.encoded_words is None
    assert plan.represented_parameters is None
    assert plan.quantization_error is None
    assert plan.quantization_error_bound is None
    with pytest.raises(ValueError, match="numerical|classical|basis"):
        factory(structure=structure)


def test_explicit_parameter_registers_and_direct_plan_default():
    factory = _api()
    from cudaq_algorithms.basis_change import FermionBasisChange
    layout = ((4, 5), (7, 8), (10, 11), (13, 14))
    plan = factory(np.eye(2),
                   programmable=True,
                   angle_bits=2,
                   parameter_registers=layout)
    assert plan.parameter_registers == layout
    assert plan.num_qubits == 15
    direct = FermionBasisChange(plan.schedule, plan.angle_encoding)
    assert direct.parameter_registers == ((2, 3), (4, 5), (6, 7), (8, 9))
    assert direct.num_qubits == 10


def _load_words(operation, qubits):
    """Set the listed word qubits with X gates, then apply operation."""

    @cudaq.kernel
    def load_and_apply(register: cudaq.qview):
        for k in range(len(qubits)):
            x(register[qubits[k]])
        operation(register)

    return load_and_apply


def test_explicit_words_drive_plan_kernels_after_round_trips():
    # Exact on the three-bit grid, so the loaded words represent U itself.
    unitary = np.array([[1., -1.], [1j, 1j]]) / np.sqrt(2)
    layout = ((8, 9, 10), (2, 3, 4), (11, 12, 13), (5, 6, 7))
    plan = _api()(unitary,
                  programmable=True,
                  angle_bits=3,
                  parameter_registers=layout)
    assert plan.encoded_words == (0, 2, 1, 2)
    ones = [
        qubit for word, code in zip(layout, plan.encoded_words)
        for bit, qubit in enumerate(word) if code >> bit & 1
    ]
    loaded = sum(1 << qubit for qubit in ones)
    system = _random_state(2, seed=17)
    initial = np.zeros(1 << 14, dtype=complex)
    initial[:4] = system
    forward = _fock_matrix(unitary)
    for candidate in (plan, pickle.loads(pickle.dumps(plan)),
                      copy.deepcopy(plan)):
        assert candidate == plan
        assert candidate.parameter_registers == layout
        kernels = (candidate.kernel(), candidate.adjoint_kernel())
        for operation, block in zip(kernels, (forward, forward.conj().T)):
            # The system transforms and every word keeps its loaded value.
            expected = np.zeros_like(initial)
            expected[loaded:loaded + 4] = block @ system
            actual = _evolve(_load_words(operation, ones), initial)[0]
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("layout", [
    ((2, ), ),
    ((2, 3), (3, 4), (5, 6), (7, 8)),
    ((1, 2), (3, 4), (5, 6), (7, 8)),
    ((2, 4), (5, 6), (7, 8), (9, 10)),
    ((True, 2), (3, 4), (5, 6), (7, 8)),
])
def test_invalid_program_layout_is_rejected(layout):
    with pytest.raises(ValueError):
        _api()(np.eye(2),
               programmable=True,
               angle_bits=2,
               parameter_registers=layout)


@pytest.mark.parametrize("count", [3, 5])
def test_program_layout_must_match_slot_count(count):
    layout = tuple((2 + slot, ) for slot in range(count))
    with pytest.raises(ValueError, match="slot count"):
        _api()(np.eye(2),
               programmable=True,
               angle_bits=1,
               parameter_registers=layout)


@pytest.mark.parametrize("layout", [
    {(2, 3), (4, 5), (6, 7), (8, 9)},
    frozenset({(2, 3), (4, 5), (6, 7), (8, 9)}),
    ({2, 3}, (4, 5), (6, 7), (8, 9)),
    iter(((2, 3), (4, 5), (6, 7), (8, 9))),
    dict.fromkeys(((2, 3), (4, 5), (6, 7), (8, 9))).keys(),
    ((2, 3), (4, 5), (6, 7), iter({8, 9})),
    (b"\x02\x03", (4, 5), (6, 7), (8, 9)),
    (bytearray([2, 3]), (4, 5), (6, 7), (8, 9)),
])
def test_unordered_program_layout_is_rejected(layout):
    # Hash order must never decide slot or bit placement, so iterators and
    # views are rejected too; bytes are not index sequences.
    with pytest.raises(ValueError, match="ordered sequence"):
        _api()(np.eye(2),
               programmable=True,
               angle_bits=2,
               parameter_registers=layout)


def test_program_layout_indices_are_bounded():

    def plan(index):
        return _api()(np.eye(2),
                      programmable=True,
                      angle_bits=1,
                      parameter_registers=((2, ), (3, ), (4, ), (index, )))

    assert plan(2**31 - 1).num_qubits == 2**31
    for index in (-1, 2**31):
        with pytest.raises(ValueError, match="qubit index"):
            plan(index)


def test_default_words_obey_the_qubit_index_bound():

    def plan(top):
        structure = _structure("full", 2, mode_map=(0, top))
        return _api()(structure=structure, programmable=True, angle_bits=1)

    # Four one-bit default words follow the system span, ending at top + 4.
    widest = plan(2**31 - 5)
    assert widest.parameter_registers[-1] == (2**31 - 1, )
    # Loading validates the words again as explicit ones, without expanding
    # the system span into a set of 2**31 indices.
    assert pickle.loads(pickle.dumps(widest)) == widest
    for top in (2**31 - 4, 2**31 - 1):
        with pytest.raises(ValueError, match="qubit index"):
            plan(top)
    # The bound concerns words; a classical plan may use the top index.
    top_mode = _structure("full", 2, mode_map=(0, 2**31 - 1))
    assert _api()(np.eye(2), structure=top_mode).num_qubits == 2**31


@pytest.mark.parametrize("layout", [
    ((2, 3), (4, 5), (6, 7), (8, 9)),
    [[2, 3], [4, 5], [6, 7], [8, 9]],
    np.arange(2, 10).reshape(4, 2),
    (np.array([2, 3], dtype=np.uint8), np.arange(4, 6), range(6, 8), [8, 9]),
])
def test_ordered_program_layout_is_normalized(layout):
    plan = _api()(np.eye(2),
                  programmable=True,
                  angle_bits=2,
                  parameter_registers=layout)
    assert plan.parameter_registers == ((2, 3), (4, 5), (6, 7), (8, 9))
    assert all(
        type(index) is int for word in plan.parameter_registers
        for index in word)


def test_program_words_cannot_overlap_spectator_span():
    factory = _api()
    from cudaq_algorithms.basis_change import GivensStructure
    structure = GivensStructure.full(2, mode_map=(0, 2))
    with pytest.raises(ValueError, match="overlap|system"):
        factory(np.eye(2),
                structure=structure,
                programmable=True,
                angle_bits=1,
                parameter_registers=((1, ), (3, ), (4, ), (5, )))


def test_default_words_follow_physical_span_with_gap():
    structure = _structure("full", 2, mode_map=(0, 2))
    plan = _api()(structure=structure, programmable=True, angle_bits=1)
    assert plan.parameter_registers == ((3, ), (4, ), (5, ), (6, ))
    assert plan.num_qubits == 7
    # Qubit 2 holds mapped mode 1, beyond num_modes but inside the span.
    with pytest.raises(ValueError, match="overlap"):
        _api()(np.eye(2),
               structure=structure,
               programmable=True,
               angle_bits=1,
               parameter_registers=((2, ), (3, ), (4, ), (5, )))


def test_default_word_layout_executes_with_gapped_mode_map():
    plan = _api()(structure=_structure("full", 2, mode_map=(0, 2)),
                  programmable=True,
                  angle_bits=2)
    layout = ((3, 4), (5, 6), (7, 8), (9, 10))
    assert plan.parameter_registers == layout
    weights = (np.pi / 2, np.pi)
    initial = _random_state(11, seed=5)
    expected = initial.copy()
    # Each computational word value selects its own exact transformation.
    for words in range(1 << 8):
        index = words << 3
        values = [
            sum(weight * (index >> qubit & 1)
                for qubit, weight in zip(word, weights)) for word in layout
        ]
        operator = _fock_matrix(
            _embedded(_two_mode_full_orbital(values), (0, 2)))
        positions = index + np.arange(8)
        expected[positions] = operator @ initial[positions]
    actual = _evolve(plan.kernel(), initial)[0]
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("width", [None, 0, -1, 53, True, np.bool_(True), 1.5])
def test_invalid_program_width_is_rejected(width):
    with pytest.raises(ValueError, match="angle_bits|width"):
        _api()(np.eye(2), programmable=True, angle_bits=width)


@pytest.mark.parametrize("kwargs", [
    {
        "angle_bits": 3
    },
    {
        "parameter_registers": ()
    },
    {
        "programmable": 1
    },
])
def test_inconsistent_factory_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _api()(np.eye(2), **kwargs)


@pytest.mark.parametrize(
    "tolerance",
    [0, -1, float("nan"), float("inf"), "bad"])
def test_compiled_schedule_still_validates_tolerance(tolerance):
    factory = _api()
    schedule = factory(np.eye(2)).schedule
    with pytest.raises(ValueError, match="tolerance"):
        factory(schedule, tolerance=tolerance)


def test_compiled_schedule_rejects_a_different_structure():
    factory = _api()
    schedule = factory(np.eye(2)).schedule
    assert factory(schedule, structure=schedule.structure).schedule is schedule
    gapped = _structure("full", 2, mode_map=(0, 2))
    for other in (gapped, _structure("partial", 2)):
        with pytest.raises(ValueError, match="structure"):
            factory(schedule, structure=other)


def test_factory_forwards_tolerance_with_strict_default():
    factory = _api()
    unitary = _random_unitary(3, seed=10)
    # Deviations near 1e-10 exceed the documented default of 1e-12.
    for basis in (unitary * (1 + 1e-10), [1 + 1e-10, 0.]):
        with pytest.raises(ValueError, match="within tolerance"):
            factory(basis)
    factory(unitary * (1 + 1e-9), tolerance=1e-8)
    with pytest.raises(ValueError, match="within tolerance"):
        factory(unitary * (1 + 1e-13), tolerance=1e-14)


def test_binary_encoding_rounding_wrapping_and_python_integer_codes():
    encoding = _encoding(np.int64(3))
    step = math.pi / 4
    assert encoding.encode(step / 2) == 0
    assert encoding.encode(3 * step / 2) == 2
    assert encoding.encode(-step / 2) == 0
    assert encoding.encode(-step) == 7
    assert encoding.encode(2 * math.pi) == 0
    assert encoding.encode(-0.) == 0
    assert encoding.decode(7) == 7 * step
    assert type(encoding.encode(step)) is int
    assert _encoding(52).encode(math.pi) == 1 << 51
    with pytest.raises(FrozenInstanceError):
        encoding.width = 5


@pytest.mark.parametrize("angle", [
    float("nan"),
    float("inf"),
    True,
    "1",
    pytest.param(10**1000, id="unrepresentable_integer"),
])
def test_encoding_rejects_invalid_angles(angle):
    with pytest.raises(ValueError):
        _encoding(3).encode(angle)


@pytest.mark.parametrize("code", [-1, 8, True, 1.5])
def test_encoding_rejects_invalid_codes(code):
    with pytest.raises(ValueError):
        _encoding(3).decode(code)


def test_negative_phase_quantization_uses_circular_error():
    original = np.diag([np.exp(-0.01j), 1.])
    plan = _api()(original, programmable=True, angle_bits=4)
    assert plan.encoded_words == (0, 0, 0, 0)
    assert plan.represented_parameters == (0., 0., 0., 0.)
    assert plan.quantization_error_bound == pytest.approx(0.01)
    represented = plan.schedule.orbital_matrix(plan.represented_parameters)
    measured = np.linalg.norm(represented - original, ord=2)
    assert measured <= plan.quantization_error_bound + 1e-14


def test_quantization_report_excludes_reconstruction_residual():
    # Every angle is exactly zero; only the accepted input is inexact.
    plan = _api()(np.diag([1., 1. + 1e-6]),
                  programmable=True,
                  angle_bits=3,
                  tolerance=1e-5)
    assert plan.schedule.reconstruction_error == pytest.approx(1e-6, rel=1e-6)
    assert plan.encoded_words == (0, 0, 0, 0)
    assert plan.quantization_error.synthesis_error_bound == 0
    assert plan.quantization_error_bound == 0


def test_cached_kernels_do_not_prevent_plan_serialization():
    plan = _api()(np.eye(2), programmable=True, angle_bits=2)
    plan.kernel()
    plan.adjoint_kernel()
    restored = pickle.loads(pickle.dumps(plan))
    assert restored == plan
    assert "_forward" not in restored.__dict__
    assert "_inverse" not in restored.__dict__
    assert copy.deepcopy(plan) == plan


@pytest.mark.parametrize("magnitude", [1e16, 1e20, 1e308])
@pytest.mark.parametrize("sign", [-1, 1])
def test_large_finite_schedule_angles_preserve_phase_when_quantized(
        magnitude, sign):
    from cudaq_algorithms.basis_change import (GivensBasisChangeSchedule,
                                               GivensStructure)
    angle = sign * magnitude
    schedule = GivensBasisChangeSchedule(GivensStructure.full(2),
                                         (angle, 0., 0., 0.))
    plan = _api()(schedule, programmable=True, angle_bits=20)
    assert plan.encoded_words[1:] == (0, 0, 0)
    # Only mode 0 carries a phase, so the Fock operator is diagonal with
    # exp(i * angle) on every state in which mode 0 is occupied.
    measured = abs(
        np.exp(1j * angle) - np.exp(1j * plan.represented_parameters[0]))
    assert measured <= plan.angle_encoding.step / 2 + 1e-14
    assert measured <= plan.quantization_error_bound + 1e-14


def _machin_pi(digits):
    """Return pi = 16 atan(1/5) - 4 atan(1/239) in Decimal arithmetic."""
    with decimal.localcontext() as context:
        context.prec = digits + 10
        threshold = decimal.Decimal(10)**-(digits + 5)

        def arctan_inverse(n):
            total, power, k = decimal.Decimal(0), decimal.Decimal(1) / n, 0
            while power > threshold:
                total += (-1)**k * power / (2 * k + 1)
                power /= n * n
                k += 1
            return total

        return 16 * arctan_inverse(5) - 4 * arctan_inverse(239)


@pytest.mark.parametrize("width", [40, 52])
@pytest.mark.parametrize("angle", [100., 12345.678, 1e6, -1e6])
def test_moderately_large_angles_keep_physical_phase(angle, width):
    # A remainder by floating-point 2*pi drifts by many steps at these sizes.
    size = 1 << width
    with decimal.localcontext() as context:
        context.prec = 90
        turns = decimal.Decimal(angle) / (2 * _machin_pi(80))
        turns -= turns.to_integral_value(rounding=decimal.ROUND_FLOOR)
        nearest = int((turns * size).to_integral_value()) % size
    distance = (_encoding(width).encode(angle) - nearest) % size
    assert min(distance, size - distance) <= 1


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("bits", [1, 3, 20, 52])
def test_canonical_half_step_ties_remain_nearest_even(bits, sign):
    encoding = _encoding(bits)
    assert encoding.encode(sign * encoding.step / 2) == 0
    assert encoding.encode(sign * 3 * encoding.step /
                           2) == (sign * 2) % (1 << bits)


@pytest.mark.parametrize("basis", [np.eye(1), [1j], [[np.exp(.4j)]]])
def test_single_mode_input_is_rejected(basis):
    # One mode admits only a number phase; callers apply r1 directly.
    for kwargs in ({}, {"programmable": True, "angle_bits": 2}):
        with pytest.raises(ValueError, match="at least 2"):
            _api()(basis, **kwargs)
