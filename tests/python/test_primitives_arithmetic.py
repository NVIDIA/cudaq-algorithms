# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exhaustive tests for the primitives arithmetic device kernels.

Every operation is checked against classical integer arithmetic over *all*
inputs for register widths up to 5, and every inverse is pinned by an
op-then-inverse == identity test. Exhaustiveness comes from a superposition
harness: the input registers are prepared in a uniform superposition, so a
single statevector comparison validates the operation's action on every
computational-basis input at once (the operations are classical
permutations, so distinct inputs cannot interfere).

The resource-contract tests at the bottom hold the module's documented
gate prices against the compiler: ``cudaq.estimate_resources`` counts the
operations actually synthesized, so the closed forms (``2 n`` Toffolis for
every CDKM operation, zero Toffolis and the exact ``r1`` budgets for the
Draper QFT family) are compiler facts, not emitter claims.
"""

import numpy as np
import pytest

import cudaq

from cudaq_algorithms.primitives import _arithmetic as arith

WIDTHS = [1, 2, 3, 4, 5]


def _basis(index: int, num_qubits: int) -> np.ndarray:
    ket = np.zeros(1 << num_qubits, dtype=np.complex128)
    ket[index] = 1.0
    return ket


# ----------------------------------------------------------------------
# Register-register add / subtract (CDKM)
# ----------------------------------------------------------------------
#
# Layout: a at bits [0, n), b at [n, 2n), carry at bit 2n.


@cudaq.kernel
def _run_add_register(n: int, subtract: int):
    a = cudaq.qvector(n)
    b = cudaq.qvector(n)
    carry = cudaq.qvector(1)
    for k in range(n):
        h(a[k])
        h(b[k])
    if subtract == 0:
        arith.add_register(a, b, carry)
    else:
        arith.subtract_register(a, b, carry)


@cudaq.kernel
def _run_add_then_subtract(n: int):
    a = cudaq.qvector(n)
    b = cudaq.qvector(n)
    carry = cudaq.qvector(1)
    for k in range(n):
        h(a[k])
        h(b[k])
    arith.add_register(a, b, carry)
    arith.subtract_register(a, b, carry)


def _register_op_expected(n: int, op) -> np.ndarray:
    """Uniform superposition over (a, b) mapped through (a, op(a, b))."""
    expected = np.zeros(1 << (2 * n + 1), dtype=np.complex128)
    norm = 1.0 / (1 << n)
    for a in range(1 << n):
        for b in range(1 << n):
            expected[a + (op(a, b) % (1 << n)) * (1 << n)] += norm
    return expected


@pytest.mark.parametrize("n", WIDTHS)
def test_add_register_all_inputs(n):
    state = np.array(cudaq.get_state(_run_add_register, n, 0))
    np.testing.assert_allclose(state,
                               _register_op_expected(n, lambda a, b: a + b),
                               atol=1e-12)


@pytest.mark.parametrize("n", WIDTHS)
def test_subtract_register_all_inputs(n):
    state = np.array(cudaq.get_state(_run_add_register, n, 1))
    np.testing.assert_allclose(state,
                               _register_op_expected(n, lambda a, b: b - a),
                               atol=1e-12)


@pytest.mark.parametrize("n", WIDTHS)
def test_subtract_register_inverts_add_register(n):
    state = np.array(cudaq.get_state(_run_add_then_subtract, n))
    np.testing.assert_allclose(state,
                               _register_op_expected(n, lambda a, b: b),
                               atol=1e-12)


# ----------------------------------------------------------------------
# Constant add / subtract (CDKM and Draper QFT)
# ----------------------------------------------------------------------
#
# CDKM layout: target at [0, n), work at [n, 2n), carry at 2n.
# QFT layout: target only.


@cudaq.kernel
def _run_add_constant(n: int, bits: list[int], subtract: int, roundtrip: int):
    target = cudaq.qvector(n)
    work = cudaq.qvector(n)
    carry = cudaq.qvector(1)
    for k in range(n):
        h(target[k])
    if subtract == 0:
        arith.add_constant(target, bits, work, carry)
    else:
        arith.subtract_constant(target, bits, work, carry)
    if roundtrip == 1:
        if subtract == 0:
            arith.subtract_constant(target, bits, work, carry)
        else:
            arith.add_constant(target, bits, work, carry)


@cudaq.kernel
def _run_add_constant_qft(n: int, constant: int, subtract: int,
                          roundtrip: int):
    target = cudaq.qvector(n)
    for k in range(n):
        h(target[k])
    if subtract == 0:
        arith.add_constant_qft(target, constant)
    else:
        arith.subtract_constant_qft(target, constant)
    if roundtrip == 1:
        if subtract == 0:
            arith.subtract_constant_qft(target, constant)
        else:
            arith.add_constant_qft(target, constant)


def _constant_op_expected(n: int, shift: int, extra_qubits: int) -> np.ndarray:
    expected = np.zeros(1 << (n + extra_qubits), dtype=np.complex128)
    norm = 1.0 / np.sqrt(1 << n)
    for value in range(1 << n):
        expected[(value + shift) % (1 << n)] += norm
    return expected


def _bits(value: int, n: int) -> list[int]:
    return [(value >> k) & 1 for k in range(n)]


@pytest.mark.parametrize("n", WIDTHS)
def test_add_and_subtract_constant_all_inputs(n):
    for constant in range(1 << n):
        bits = _bits(constant, n)
        added = np.array(cudaq.get_state(_run_add_constant, n, bits, 0, 0))
        np.testing.assert_allclose(added,
                                   _constant_op_expected(n, constant, n + 1),
                                   atol=1e-12)
        subtracted = np.array(cudaq.get_state(_run_add_constant, n, bits, 1,
                                              0))
        np.testing.assert_allclose(subtracted,
                                   _constant_op_expected(n, -constant, n + 1),
                                   atol=1e-12)
        roundtrip = np.array(cudaq.get_state(_run_add_constant, n, bits, 0, 1))
        np.testing.assert_allclose(roundtrip,
                                   _constant_op_expected(n, 0, n + 1),
                                   atol=1e-12)


@pytest.mark.parametrize("n", WIDTHS)
def test_add_and_subtract_constant_qft_all_inputs(n):
    for constant in range(1 << n):
        added = np.array(
            cudaq.get_state(_run_add_constant_qft, n, constant, 0, 0))
        np.testing.assert_allclose(added,
                                   _constant_op_expected(n, constant, 0),
                                   atol=1e-10)
        subtracted = np.array(
            cudaq.get_state(_run_add_constant_qft, n, constant, 1, 0))
        np.testing.assert_allclose(subtracted,
                                   _constant_op_expected(n, -constant, 0),
                                   atol=1e-10)
        roundtrip = np.array(
            cudaq.get_state(_run_add_constant_qft, n, constant, 0, 1))
        np.testing.assert_allclose(roundtrip,
                                   _constant_op_expected(n, 0, 0),
                                   atol=1e-10)


# ----------------------------------------------------------------------
# >= comparators (CDKM and Draper QFT)
# ----------------------------------------------------------------------


@cudaq.kernel
def _run_cmp_ge_constant(n: int, bits: list[int], k_is_zero: int):
    x_reg = cudaq.qvector(n)
    work = cudaq.qvector(n)
    carry = cudaq.qvector(1)
    out = cudaq.qvector(1)
    for k in range(n):
        h(x_reg[k])
    arith.cmp_ge_constant(x_reg, bits, k_is_zero, work, carry, out)


@cudaq.kernel
def _run_cmp_ge_constant_qft(n: int, constant: int, invert: int,
                             uncompute: int):
    x_reg = cudaq.qvector(n)
    out = cudaq.qvector(1)
    for k in range(n):
        h(x_reg[k])
    arith.cmp_ge_constant_qft(x_reg, out, constant, invert)
    if uncompute == 1:
        arith.cmp_ge_constant_qft_adj(x_reg, out, constant, invert)


def _cmp_expected(n: int, predicate, extra_before_out: int) -> np.ndarray:
    """|x> (work/carry |0>) |out = predicate(x)> over superposed x."""
    total = n + extra_before_out + 1
    expected = np.zeros(1 << total, dtype=np.complex128)
    norm = 1.0 / np.sqrt(1 << n)
    for value in range(1 << n):
        expected[value +
                 (1 << (n + extra_before_out)) * int(predicate(value))] += norm
    return expected


@pytest.mark.parametrize("n", WIDTHS)
def test_cmp_ge_constant_all_inputs(n):
    for constant in range(1 << n):
        complement = _bits((1 << n) - constant, n) if constant else [0] * n
        state = np.array(
            cudaq.get_state(_run_cmp_ge_constant, n, complement,
                            int(constant == 0)))
        np.testing.assert_allclose(state,
                                   _cmp_expected(n, lambda v: v >= constant,
                                                 n + 1),
                                   atol=1e-12)


@pytest.mark.parametrize("n", WIDTHS)
def test_cmp_ge_constant_qft_all_inputs(n):
    # Between the compute/adjoint pair the register is shifted by -K (a
    # documented contract), so the compute-only check reads out through the
    # shifted basis; the roundtrip check pins full restoration.
    for constant in range(1 << n):
        for invert in (0, 1):
            state = np.array(
                cudaq.get_state(_run_cmp_ge_constant_qft, n, constant, invert,
                                0))
            expected = np.zeros(1 << (n + 1), dtype=np.complex128)
            norm = 1.0 / np.sqrt(1 << n)
            for value in range(1 << n):
                flag = (value >= constant) if invert == 0 else (value
                                                                < constant)
                shifted = (value - constant) % (1 << n)
                expected[shifted + (int(flag) << n)] += norm
            np.testing.assert_allclose(state, expected, atol=1e-10)
            roundtrip = np.array(
                cudaq.get_state(_run_cmp_ge_constant_qft, n, constant, invert,
                                1))
            np.testing.assert_allclose(roundtrip,
                                       _constant_op_expected(n, 0, 1),
                                       atol=1e-10)


# ----------------------------------------------------------------------
# Direct basis-state spot checks (readable, non-superposed)
# ----------------------------------------------------------------------


@cudaq.kernel
def _spot_add(n: int, aval: int, bval: int):
    a = cudaq.qvector(n)
    b = cudaq.qvector(n)
    carry = cudaq.qvector(1)
    for k in range(n):
        if ((aval >> k) & 1) == 1:
            x(a[k])
        if ((bval >> k) & 1) == 1:
            x(b[k])
    arith.add_register(a, b, carry)


def test_add_register_basis_spot_checks():
    for n, aval, bval in [(2, 1, 2), (3, 5, 6), (5, 21, 27)]:
        state = np.array(cudaq.get_state(_spot_add, n, aval, bval))
        index = aval + (((aval + bval) % (1 << n)) << n)
        np.testing.assert_allclose(state, _basis(index, 2 * n + 1), atol=1e-12)


# ----------------------------------------------------------------------
# Compiler-pinned resource contracts
# ----------------------------------------------------------------------
#
# The harnesses below pass the operation parameters as *runtime kernel
# arguments* (never source literals): constant-folded literals let the
# compiler specialize the circuit, and the pinned counts would then
# depend on the folding rather than the construction.
#
# CDKM derivations (n-bit registers, one Toffoli per MAJ and per UMA):
# - add_register / subtract_register: the MAJ sweep is one Toffoli at
#   the carry step plus n - 1 in the ripple = n; the UMA sweep mirrors
#   it = n. Total exactly 2 n.
# - add_constant / subtract_constant: the constant load/unload is X-only
#   (free), so the price is the inner register adder's 2 n.
# - cmp_ge_constant (K >= 1): a MAJ sweep (n) plus its literal reversal
#   (n) with a free CNOT carry-copy in between = 2 n; K = 0 is a bare X.
#
# Draper QFT derivations: no Toffolis at all. add_constant_qft is
# qft + phases + iqft = 2 * (n(n-1)/2) controlled-r1, n free r1 and 2 n
# H; each side of the cmp_ge_constant_qft pair is one extended
# (n+1)-bit QFT sandwich = n(n+1) controlled-r1, n + 1 free r1 and
# 2 (n + 1) H (K >= 1; K = 0 emits no rotations).

_RESOURCES = pytest.mark.skipif(
    not hasattr(cudaq, "estimate_resources"),
    reason="cudaq.estimate_resources is not available in this CUDA-Q")


@cudaq.kernel
def _res_add_register(n: int, subtract: int):
    a = cudaq.qvector(n)
    b = cudaq.qvector(n)
    carry = cudaq.qvector(1)
    if subtract == 0:
        arith.add_register(a, b, carry)
    else:
        arith.subtract_register(a, b, carry)


@cudaq.kernel
def _res_add_constant(n: int, bits: list[int], subtract: int):
    target = cudaq.qvector(n)
    work = cudaq.qvector(n)
    carry = cudaq.qvector(1)
    if subtract == 0:
        arith.add_constant(target, bits, work, carry)
    else:
        arith.subtract_constant(target, bits, work, carry)


@cudaq.kernel
def _res_cmp_ge_constant(n: int, bits: list[int], k_is_zero: int):
    x_reg = cudaq.qvector(n)
    work = cudaq.qvector(n)
    carry = cudaq.qvector(1)
    out = cudaq.qvector(1)
    arith.cmp_ge_constant(x_reg, bits, k_is_zero, work, carry, out)


@cudaq.kernel
def _res_add_constant_qft(n: int, constant: int):
    target = cudaq.qvector(n)
    arith.add_constant_qft(target, constant)


@cudaq.kernel
def _res_cmp_ge_constant_qft(n: int, constant: int, invert: int):
    x_reg = cudaq.qvector(n)
    out = cudaq.qvector(1)
    arith.cmp_ge_constant_qft(x_reg, out, constant, invert)


def _toffolis(kernel, *args) -> int:
    # A Toffoli is an x with two controls; ``count_controls`` is the
    # arity-aware accessor (``count("ccx")`` matches nothing — the
    # display name is not the lookup key and returns 0).
    return cudaq.estimate_resources(kernel, *args).count_controls("x", 2)


@_RESOURCES
@pytest.mark.parametrize("n", WIDTHS)
def test_cdkm_register_ops_cost_exactly_2n_toffolis(n):
    assert _toffolis(_res_add_register, n, 0) == 2 * n
    assert _toffolis(_res_add_register, n, 1) == 2 * n


@_RESOURCES
@pytest.mark.parametrize("n", WIDTHS)
def test_cdkm_constant_ops_cost_exactly_2n_toffolis(n):
    bits = _bits((1 << n) - 1, n)  # worst-case load: every bit set
    assert _toffolis(_res_add_constant, n, bits, 0) == 2 * n
    assert _toffolis(_res_add_constant, n, bits, 1) == 2 * n


@_RESOURCES
@pytest.mark.parametrize("n", WIDTHS)
def test_cdkm_comparator_costs_exactly_2n_toffolis(n):
    complement = _bits((1 << n) - 1, n)  # K = 1
    assert _toffolis(_res_cmp_ge_constant, n, complement, 0) == 2 * n
    # K = 0 short-circuits to a single X: no Toffolis.
    assert _toffolis(_res_cmp_ge_constant, n, [0] * n, 1) == 0


@_RESOURCES
def test_cdkm_toffoli_cost_grows_linearly_per_doubling():
    # The cost is exactly linear (2 n): each width doubling doubles it.
    compiled = {n: _toffolis(_res_add_register, n, 0) for n in (2, 4, 8, 16)}
    for n in (2, 4, 8):
        assert compiled[2 * n] == 2 * compiled[n], compiled


@_RESOURCES
@pytest.mark.parametrize("n", WIDTHS)
def test_qft_add_constant_costs_rotations_not_toffolis(n):
    resources = cudaq.estimate_resources(_res_add_constant_qft, n, 1)
    assert resources.count_controls("x", 2) == 0
    assert resources.count_controls("r1", 1) == n * (n - 1)
    assert resources.count_controls("r1", 0) == n
    assert resources.count("h") == 2 * n


@_RESOURCES
@pytest.mark.parametrize("n", WIDTHS)
def test_qft_comparator_costs_rotations_not_toffolis(n):
    for invert in (0, 1):
        resources = cudaq.estimate_resources(_res_cmp_ge_constant_qft, n, 1,
                                             invert)
        assert resources.count_controls("x", 2) == 0
        assert resources.count_controls("r1", 1) == n * (n + 1)
        assert resources.count_controls("r1", 0) == n + 1
        assert resources.count("h") == 2 * (n + 1)
        assert resources.count("x") == (1 if invert == 0 else 0)
    # K = 0 emits no rotations at all: the constant-true comparator is a
    # bare X on the out qubit.
    trivial = cudaq.estimate_resources(_res_cmp_ge_constant_qft, n, 0, 0)
    assert trivial.count("r1") == 0
    assert trivial.count("x") == 1
