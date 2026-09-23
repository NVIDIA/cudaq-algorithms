# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small-register E02 assertions. References do not use artifact dense helpers."""
import numpy as np
import operator

ATOL = RTOL = 1e-11
FACTORS = (2., .25, -1., -.7)
CASES = (
    ({
        "I": .23,
        "X": -.51,
        "Y": .37,
        "Z": .69
    }, np.array([[.92, -.51 - .37j], [-.51 + .37j, -.46]], complex)),
    ({
        "II": .17,
        "XI": -.31,
        "IY": .43,
        "ZZ": -.59,
        "ZX": .23
    },
     np.array([[-.42, -.31, .23 - .43j, 0], [-.31, .76, 0, -.23 - .43j],
               [.23 + .43j, 0, .76, -.31], [0, -.23 + .43j, -.31, -.42]],
              complex)),
)


class ProbeLimit(RuntimeError):
    """Valid-looking artifact exceeds this bounded oracle: acceptance unknown."""


class ZeroUnsupported(RuntimeError):
    """Documented zero rejection: incomplete acceptance, not a numerical fail."""


def zero_attempt(invoke, encoding, zero_policy):
    try:
        return invoke(encoding, 0.)
    except (ValueError, TypeError, NotImplementedError) as exc:
        if zero_policy == "reject":
            raise ZeroUnsupported(
                "E02_ZERO_UNSUPPORTED: acceptance inconclusive") from exc
        return exc
    except ArithmeticError as exc:
        return exc


def assert_invalid_rejected(invoke, encoding, factor):
    try:
        invoke(encoding, factor)
    except (ValueError, TypeError):
        return
    raise AssertionError("invalid factor was accepted")


def geometry(encoding, controlled=False):
    ns, na = operator.index(encoding.num_system), operator.index(
        encoding.num_ancilla)
    assert ns >= 1 and na >= 1, "requires positive system and ancilla widths"
    if ns + na + int(controlled) > 7:
        raise ProbeLimit("E02_PROBE_LIMIT: more than seven total qubits")
    return ns, na, 1 << (ns + na + int(controlled))


def _probe_kernel(encoding, factory):
    import cudaq
    controlled = factory.startswith("controlled_")
    ns, na, dimension = geometry(encoding, controlled)
    signal_width = na + int(controlled)
    operation = getattr(encoding, factory)()

    @cudaq.kernel
    def probe(state: cudaq.State):
        register = cudaq.qvector(state)
        system = register.front(ns)
        signal = register.back(signal_width)
        operation(signal, system)

    return probe, dimension


def apply_state(encoding, factory, state):
    import cudaq
    kernel, dimension = _probe_kernel(encoding, factory)
    state = np.asarray(state, dtype=np.complex128)
    assert state.shape == (dimension, )
    return np.asarray(cudaq.get_state(kernel, cudaq.State.from_data(state)),
                      dtype=np.complex128)


def full_matrix(encoding, factory):
    import cudaq
    kernel, dimension = _probe_kernel(encoding, factory)
    columns = []
    for index in range(dimension):
        state = np.zeros(dimension, dtype=np.complex128)
        state[index] = 1.
        columns.append(
            np.asarray(cudaq.get_state(kernel, cudaq.State.from_data(state)),
                       dtype=np.complex128))
    matrix = np.column_stack(columns)
    _close(matrix.conj().T @ matrix, np.eye(dimension))
    return matrix


def prepared_walk_block(encoding):
    """Probe P-dagger W P; raw W is in the prepared-ancilla basis."""
    import cudaq
    ns, na, _ = geometry(encoding)
    prepare = encoding.prepare_kernel()
    walk = encoding.walk_step_kernel()
    unprepare = encoding.unprepare_kernel()

    @cudaq.kernel
    def probe(state: cudaq.State):
        system = cudaq.qvector(state)
        ancilla = cudaq.qvector(na)
        prepare(ancilla)
        walk(ancilla, system)
        unprepare(ancilla)

    dim = 1 << ns
    columns = []
    for index in range(dim):
        ket = np.zeros(dim, complex)
        ket[index] = 1.
        output = np.asarray(cudaq.get_state(probe, cudaq.State.from_data(ket)),
                            complex)
        columns.append(output[:dim])
    return np.column_stack(columns)


class ForeignEncoding:
    """Only original protocol data and minted handles; no backing built-in."""
    __slots__ = ("num_system", "num_ancilla", "alpha", "_handles",
                 "_observable")

    def __init__(self, inner):
        self.num_system, self.num_ancilla, self.alpha = (inner.num_system,
                                                         inner.num_ancilla,
                                                         inner.alpha)
        self._handles = {
            name: getattr(inner, name)()
            for name in ("prepare_kernel", "unprepare_kernel", "apply_kernel",
                         "controlled_apply_kernel", "walk_step_kernel",
                         "adjoint_walk_step_kernel",
                         "controlled_walk_step_kernel",
                         "controlled_adjoint_walk_step_kernel")
        }
        self._observable = inner.select_observable()

    def prepare_kernel(self):
        return self._handles["prepare_kernel"]

    def unprepare_kernel(self):
        return self._handles["unprepare_kernel"]

    def apply_kernel(self):
        return self._handles["apply_kernel"]

    def controlled_apply_kernel(self):
        return self._handles["controlled_apply_kernel"]

    def walk_step_kernel(self):
        return self._handles["walk_step_kernel"]

    def adjoint_walk_step_kernel(self):
        return self._handles["adjoint_walk_step_kernel"]

    def controlled_walk_step_kernel(self):
        return self._handles["controlled_walk_step_kernel"]

    def controlled_adjoint_walk_step_kernel(self):
        return self._handles["controlled_adjoint_walk_step_kernel"]

    def select_observable(self):
        return self._observable


def make_input(case, foreign=False):
    from cudaq_algorithms import PauliLCU
    inner = PauliLCU(case[0])
    return ForeignEncoding(inner) if foreign else inner


def _close(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, "wrong matrix shape"
    assert np.isfinite(actual).all(), "nonfinite matrix"
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)


def assert_scaled_block(block, hamiltonian, factor, alpha):
    assert np.ndim(alpha) == 0 and np.isreal(
        alpha), "normalization must be real"
    assert np.isfinite(
        alpha) and alpha > 0, "normalization must be finite and positive"
    expected = factor * np.asarray(hamiltonian, dtype=np.complex128)
    _close(block, expected / alpha)
    # Absolute operator check prevents huge alpha from hiding a zero/wrong block.
    _close(alpha * np.asarray(block), expected)


def assert_controlled(controlled, unitary, num_system):
    unitary = np.asarray(unitary)
    dim = len(unitary)
    expected = np.eye(2 * dim, dtype=np.complex128)
    # Physical register order: system, external control, encoding ancillas.
    enabled = [((i >> num_system) <<
                (num_system + 1)) + (1 << num_system) + (i % (1 << num_system))
               for i in range(dim)]
    expected[np.ix_(enabled, enabled)] = unitary
    _close(controlled,
           expected)  # Includes relative phase between both branches.


def assert_inverse(forward, adjoint):
    forward, adjoint = np.asarray(forward), np.asarray(adjoint)
    _close(adjoint, forward.conj().T)
    _close(adjoint @ forward, np.eye(len(forward)))


def assert_walk_block(block, hamiltonian, alpha):
    # P† W P = R U, with R=-1 on the good subspace.
    assert_scaled_block(block, hamiltonian, -1., alpha)


def assert_qsvt_output(output, hamiltonian, alpha, ket, *, degree):
    signal = np.asarray(hamiltonian) / alpha
    if degree == 1:
        expected = -signal @ ket
    elif degree == 2:
        expected = (2 * signal @ signal - np.eye(len(signal))) @ ket
    else:
        raise ValueError(
            "only independently specified degrees one and two are supported")
    _close(output, expected)


def assert_consumers(encoding, hamiltonian):
    """Original public consumers; their source binding must be reviewed/frozen."""
    import cudaq
    from cudaq_algorithms import Walk, QSVT, PhaseSequence
    hamiltonian = np.asarray(hamiltonian)
    dim = len(hamiltonian)
    ket = np.arange(1, dim + 1) + 1j * np.arange(dim, 0, -1)
    ket = ket / np.linalg.norm(ket)
    signal = hamiltonian / encoding.alpha
    walk = Walk(encoding)
    for order, polynomial in enumerate(
        (np.eye(dim), signal, 2 * signal @ signal - np.eye(dim))):
        _close(walk.moment(ket, order), np.vdot(ket, polynomial @ ket).real)
    transformer = QSVT(encoding)
    for degree in (1, 2):
        kernel = transformer.kernel(PhaseSequence([0.] * (degree + 1)))
        output = np.asarray(
            cudaq.get_state(kernel, cudaq.State.from_data(ket)), complex)
        assert_qsvt_output(output[:dim],
                           hamiltonian,
                           encoding.alpha,
                           ket,
                           degree=degree)
