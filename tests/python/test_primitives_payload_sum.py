# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Large-payload SELECT semantics: sum over superposed addresses.

The walker's other tests use one-gate bodies. Here each address carries a
random multi-gate Pauli sequence on a 6-qubit target — non-commuting,
phase-carrying (Y contributes ``+/-i``) — and the address register enters
in a phased superposition, so one application must produce

    sum_k  a_k |k> (x) |ladder=0> (x) U_k |psi>

with every ``U_k`` built independently in NumPy as a dense 64x64 product
of embedded Paulis. Any leaf-line error, ordering error, or phase error
in the walk misassembles the sum; any ladder residue leaks amplitude out
of the ``ladder=0`` block.

The Toffoli contract rides along: Pauli payloads are leaf-controlled
two-qubit gates, so the walk's Toffoli count must NOT grow with payload
size — the structural formula holds and the compiler agrees.

The file parametrizes over every walker importable on the current
branch: the coherent fused walk always, the measured (MBU) walk when its
module is present. One file, both branches, no rebase conflict.
"""

import functools

import numpy as np
import pytest

import cudaq

from cudaq_algorithms.primitives import unary_iteration_kernels

_WALKERS = [("coherent", unary_iteration_kernels)]
try:
    from cudaq_algorithms.primitives import measured_unary_iteration_kernels
    _WALKERS.append(("measured", measured_unary_iteration_kernels))
except ImportError:
    pass

_NUM_TARGET = 6
_PAULI = {
    "x": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}
_ID = np.eye(2, dtype=np.complex128)


def _random_payloads(num_items, rng):
    """Per-address random Pauli sequences on the 6-qubit target."""
    payloads = []
    for _ in range(num_items):
        length = int(rng.integers(3, 7))
        payloads.append([(("x", "y", "z")[int(rng.integers(3))],
                          int(rng.integers(_NUM_TARGET)))
                         for _ in range(length)])
    return payloads


def _embed(gate, qubit):
    """Single-qubit Pauli embedded on the 6-qubit target, qubit 0 = LSB."""
    factors = [_PAULI[gate] if i == qubit else _ID for i in range(_NUM_TARGET)]
    return functools.reduce(np.kron, reversed(factors))


def _dense_unitary(payload):
    """Dense 64x64 product of the payload, applied in list order."""
    unitary = np.eye(1 << _NUM_TARGET, dtype=np.complex128)
    for gate, qubit in payload:
        unitary = _embed(gate, qubit) @ unitary
    return unitary


def _target_state(rng):
    """Random product state on the target, complex amplitudes."""
    state = np.ones(1, dtype=np.complex128)
    angles = []
    for _ in range(_NUM_TARGET):
        theta = float(rng.uniform(0.2, 2.9))
        phi = float(rng.uniform(0.1, 6.2))
        angles.append((theta, phi))
        # cudaq rz(phi) = diag(e^{-i phi/2}, e^{+i phi/2}) — keep the
        # half-angle phases so the comparison is exact, not up-to-phase.
        qubit = np.array([
            np.exp(-0.5j * phi) * np.cos(theta / 2.0),
            np.exp(+0.5j * phi) * np.sin(theta / 2.0)
        ])
        state = np.kron(qubit, state)  # new qubit is the higher bit
    return state, angles


@pytest.mark.parametrize("walker_name,walker",
                         _WALKERS,
                         ids=[name for name, _ in _WALKERS])
@pytest.mark.parametrize("num_bits,num_items", [(3, 8), (4, 16), (4, 11)])
def test_superposed_address_sum_with_multiqubit_payloads(
        num_bits, num_items, walker_name, walker):
    rng = np.random.default_rng(1000 * num_bits + num_items)
    payloads = _random_payloads(num_items, rng)
    walk = walker(num_bits,
                  num_items,
                  lambda k: payloads[k],
                  include_adjoint=False)
    kernel = walk.kernel
    target_state, target_angles = _target_state(rng)
    address_phases = [float(rng.uniform(0.1, 3.0)) for _ in range(num_bits)]
    thetas = [a[0] for a in target_angles]
    phis = [a[1] for a in target_angles]

    @cudaq.kernel
    def harness():
        address_reg = cudaq.qvector(num_bits)
        ladder = cudaq.qvector(num_bits)
        target = cudaq.qvector(_NUM_TARGET)
        for i in range(num_bits):
            h(address_reg[i])
            rz(address_phases[i], address_reg[i])
        for i in range(_NUM_TARGET):
            ry(thetas[i], target[i])
            rz(phis[i], target[i])
        kernel(address_reg, ladder, target)

    state = np.array(cudaq.get_state(harness))

    # Reference: address amplitude a_k from the phased uniform prep
    # (rz(phi)|+> = (e^{-i phi/2}|0> + e^{+i phi/2}|1>)/sqrt(2)); the
    # target picks up U_k on the address-k branch; ladder ends |0>.
    dim = 1 << (num_bits + num_bits + _NUM_TARGET)
    expected = np.zeros(dim, dtype=np.complex128)
    for k in range(1 << num_bits):
        amplitude = 1.0 + 0.0j
        for i in range(num_bits):
            bit = (k >> i) & 1
            sign = 1.0 if bit else -1.0
            amplitude *= np.exp(sign * 0.5j * address_phases[i]) / np.sqrt(2)
        unitary = _dense_unitary(payloads[k]) if k < num_items else np.eye(
            1 << _NUM_TARGET, dtype=np.complex128)
        block = amplitude * (unitary @ target_state)
        base = k  # address bits are the global LSBs; ladder block is 0
        stride = 1 << (2 * num_bits)
        for t in range(1 << _NUM_TARGET):
            expected[base + t * stride] = block[t]

    np.testing.assert_allclose(state, expected, atol=1e-12)


@pytest.mark.parametrize("walker_name,walker",
                         _WALKERS,
                         ids=[name for name, _ in _WALKERS])
def test_payload_size_does_not_change_toffoli_count(walker_name, walker):
    # Leaf-controlled Paulis are two-qubit gates: the Toffoli count is a
    # property of the walk structure alone. A 6-gate-per-address payload
    # must cost exactly as many Toffolis as a 1-gate payload — pinned
    # against both the emitter and the compiler.
    rng = np.random.default_rng(7)
    payloads = _random_payloads(8, rng)
    small = walker(3, 8, lambda k: [("x", 0)], include_adjoint=False)
    large = walker(3, 8, lambda k: payloads[k], include_adjoint=False)
    assert large.toffoli_count == small.toffoli_count

    if not hasattr(cudaq, "estimate_resources"):
        pytest.skip("cudaq.estimate_resources is not available")
    kernel = large.kernel

    @cudaq.kernel
    def harness():
        address_reg = cudaq.qvector(3)
        ladder = cudaq.qvector(3)
        target = cudaq.qvector(_NUM_TARGET)
        kernel(address_reg, ladder, target)

    resources = cudaq.estimate_resources(harness)
    assert resources.count_controls("x", 2) == large.toffoli_count
