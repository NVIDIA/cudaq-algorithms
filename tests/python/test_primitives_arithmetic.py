# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the primitives arithmetic device kernels.

Every operation is checked against classical integer arithmetic over *all*
inputs (register widths up to 5 for the register ops, exhaustive at small n
for the single-register constant ops), and every inverse is pinned by an
op-then-inverse == identity test. Each harness is chosen so the check is
*operation-pinning*, not merely a permutation check: the register add/subtract
fix one operand and superpose the other (a uniform superposition of *both*
operands maps to the same uniform output for add, subtract or identity, so it
proves nothing); the constant ops use basis-input truth tables (the constant K
is invisible to a uniform-input check); the comparators superpose the input and
verify the correlated output flag.

The resource-contract tests at the bottom hold the module's documented gate
prices against the compiler via ``cudaq.estimate_resources``: the CDKM adders
cost ``2n - 2`` Toffolis (the unused top carry-out's Toffoli pair cancels;
Cuccaro et al., arXiv:quant-ph/0410184, Section 4.1 reaches ``2n - 3``, not
adopted) and the CDKM comparators ``2n - 1`` (Cuccaro's comparator count),
while the Draper QFT family uses only ``r1`` rotations. Each number is stated
on the kernel's own docstring; the tests pin it.
"""

import re

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


# The input register ``a`` is a uniform superposition while ``b`` is a fixed
# classical value (looped over all of them). This is exhaustive over every
# (a, b) and, crucially, *operation-pinning*: for a fixed ``b`` the maps
# ``a -> a + b`` and ``a -> b - a`` are different permutations of ``a``, so the
# output statevector distinguishes add from subtract. Superposing *both* a and
# b does not -- any bijection sends a uniform input to the same uniform output,
# so that collapsed check stays green even if add and subtract are swapped.

@cudaq.kernel
def _run_register_op_fixed_b(n: int, bval: int, subtract: int):
    a = cudaq.qvector(n)
    b = cudaq.qvector(n)
    carry = cudaq.qvector(1)
    for k in range(n):
        h(a[k])
        if ((bval >> k) & 1) == 1:
            x(b[k])
    if subtract == 0:
        arith.add_register(a, b, carry)
    else:
        arith.subtract_register(a, b, carry)


@cudaq.kernel
def _run_add_then_subtract_fixed_b(n: int, bval: int):
    a = cudaq.qvector(n)
    b = cudaq.qvector(n)
    carry = cudaq.qvector(1)
    for k in range(n):
        h(a[k])
        if ((bval >> k) & 1) == 1:
            x(b[k])
    arith.add_register(a, b, carry)
    arith.subtract_register(a, b, carry)


def _fixed_b_expected(n: int, bval: int, op) -> np.ndarray:
    """Uniform superposition over ``a`` mapped through ``(a, op(a, b))``."""
    expected = np.zeros(1 << (2 * n + 1), dtype=np.complex128)
    amplitude = 1.0 / np.sqrt(1 << n)
    for a in range(1 << n):
        expected[a + ((op(a, bval) % (1 << n)) << n)] = amplitude
    return expected


_REGISTER_ADD_OPS = [
    ("add", 0, lambda a, b: a + b),
    ("subtract", 1, lambda a, b: b - a),
]


@pytest.mark.parametrize("n", WIDTHS)
@pytest.mark.parametrize("name,subtract,op", _REGISTER_ADD_OPS)
def test_register_add_subtract_all_inputs(n, name, subtract, op):
    for bval in range(1 << n):
        state = np.array(
            cudaq.get_state(_run_register_op_fixed_b, n, bval, subtract))
        np.testing.assert_allclose(state, _fixed_b_expected(n, bval, op),
                                   atol=1e-12)


@pytest.mark.parametrize("n", WIDTHS)
def test_subtract_register_inverts_add_register(n):
    for bval in range(1 << n):
        state = np.array(
            cudaq.get_state(_run_add_then_subtract_fixed_b, n, bval))
        np.testing.assert_allclose(state,
                                   _fixed_b_expected(n, bval, lambda a, b: b),
                                   atol=1e-12)


# ----------------------------------------------------------------------
# Constant add / subtract (CDKM and Draper QFT)
# ----------------------------------------------------------------------
#
# CDKM layout: target at [0, n), work at [n, 2n), carry at 2n.
# QFT layout: target only.


# A single register carries no second operand to leave fixed, so pinning the
# constant K requires a definite input value: these truth tables prepare
# ``target`` in each basis state (exhaustive at small n). A uniform-target
# superposition, by contrast, maps to the same uniform output for every K, so
# it checks only that work/carry are restored -- not that K was added.

@cudaq.kernel
def _run_add_constant_basis(n: int, bits: list[int], tval: int, subtract: int,
                            roundtrip: int):
    target = cudaq.qvector(n)
    work = cudaq.qvector(n)
    carry = cudaq.qvector(1)
    for k in range(n):
        if ((tval >> k) & 1) == 1:
            x(target[k])
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
def _run_add_constant_qft_basis(n: int, constant: int, tval: int, subtract: int,
                                roundtrip: int):
    target = cudaq.qvector(n)
    for k in range(n):
        if ((tval >> k) & 1) == 1:
            x(target[k])
    if subtract == 0:
        arith.add_constant_qft(target, constant)
    else:
        arith.subtract_constant_qft(target, constant)
    if roundtrip == 1:
        if subtract == 0:
            arith.subtract_constant_qft(target, constant)
        else:
            arith.add_constant_qft(target, constant)


def _bits(value: int, n: int) -> list[int]:
    return [(value >> k) & 1 for k in range(n)]


@pytest.mark.parametrize("n", [1, 2, 3])
def test_add_and_subtract_constant_all_inputs(n):
    for constant in range(1 << n):
        bits = _bits(constant, n)
        for tval in range(1 << n):
            added = np.array(
                cudaq.get_state(_run_add_constant_basis, n, bits, tval, 0, 0))
            np.testing.assert_allclose(
                added, _basis((tval + constant) % (1 << n), 2 * n + 1),
                atol=1e-12)
            subtracted = np.array(
                cudaq.get_state(_run_add_constant_basis, n, bits, tval, 1, 0))
            np.testing.assert_allclose(
                subtracted, _basis((tval - constant) % (1 << n), 2 * n + 1),
                atol=1e-12)
            roundtrip = np.array(
                cudaq.get_state(_run_add_constant_basis, n, bits, tval, 0, 1))
            np.testing.assert_allclose(roundtrip, _basis(tval, 2 * n + 1),
                                       atol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_add_and_subtract_constant_qft_all_inputs(n):
    for constant in range(1 << n):
        for tval in range(1 << n):
            added = np.array(cudaq.get_state(
                _run_add_constant_qft_basis, n, constant, tval, 0, 0))
            np.testing.assert_allclose(
                added, _basis((tval + constant) % (1 << n), n), atol=1e-10)
            subtracted = np.array(cudaq.get_state(
                _run_add_constant_qft_basis, n, constant, tval, 1, 0))
            np.testing.assert_allclose(
                subtracted, _basis((tval - constant) % (1 << n), n), atol=1e-10)
            roundtrip = np.array(cudaq.get_state(
                _run_add_constant_qft_basis, n, constant, tval, 0, 1))
            np.testing.assert_allclose(roundtrip, _basis(tval, n), atol=1e-10)


# ----------------------------------------------------------------------
# >= comparators (CDKM and Draper QFT)
# ----------------------------------------------------------------------


@cudaq.kernel
def _run_cmp_ge_constant(n: int, bits: list[int], k_is_zero: int,
                         flag_init: int):
    x_reg = cudaq.qvector(n)
    work = cudaq.qvector(n)
    carry = cudaq.qvector(1)
    out = cudaq.qvector(1)
    for k in range(n):
        h(x_reg[k])
    if flag_init == 1:
        x(out[0])
    arith.cmp_ge_constant(x_reg, bits, k_is_zero, work, carry, out)


@cudaq.kernel
def _run_cmp_ge_constant_qft_shift(n: int, constant: int, invert: int,
                             uncompute: int):
    x_reg = cudaq.qvector(n)
    out = cudaq.qvector(1)
    for k in range(n):
        h(x_reg[k])
    arith.cmp_ge_constant_qft_shift(x_reg, out, constant, invert)
    if uncompute == 1:
        arith.cmp_ge_constant_qft_shift_adj(x_reg, out, constant, invert)


def _cmp_expected(n: int, predicate, extra_before_out: int,
                  flag_init: int = 0) -> np.ndarray:
    """|x> (work/carry |0>) |out = flag_init ^ predicate(x)> over superposed x."""
    total = n + extra_before_out + 1
    expected = np.zeros(1 << total, dtype=np.complex128)
    norm = 1.0 / np.sqrt(1 << n)
    for value in range(1 << n):
        out_bit = flag_init ^ int(predicate(value))
        expected[value + (out_bit << (n + extra_before_out))] += norm
    return expected


@pytest.mark.parametrize("n", WIDTHS)
def test_cmp_ge_constant_all_inputs(n):
    # K over 0 .. 2^n inclusive (both boundaries: K = 0 always true, K = 2^n
    # always false), and out prepared to 0 and 1 (the flag is XOR-loaded).
    for constant in list(range(1 << n)) + [1 << n]:
        if constant == 0:
            complement, k_is_zero = [0] * n, 1           # K = 0: always true
        elif constant == (1 << n):
            complement, k_is_zero = [0] * n, 0           # K = 2^n: always false
        else:
            complement, k_is_zero = _bits((1 << n) - constant, n), 0
        for flag_init in (0, 1):
            state = np.array(
                cudaq.get_state(_run_cmp_ge_constant, n, complement, k_is_zero,
                                flag_init))
            np.testing.assert_allclose(
                state, _cmp_expected(n, lambda v: v >= constant, n + 1,
                                     flag_init),
                atol=1e-12)


@pytest.mark.parametrize("n", WIDTHS)
def test_cmp_ge_constant_qft_shift_all_inputs(n):
    # Between the compute/adjoint pair the register is shifted by -K (a
    # documented contract), so the compute-only check reads out through the
    # shifted basis; the roundtrip check pins full restoration.
    for constant in range(1 << n):
        for invert in (0, 1):
            state = np.array(
                cudaq.get_state(_run_cmp_ge_constant_qft_shift, n, constant, invert,
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
                cudaq.get_state(_run_cmp_ge_constant_qft_shift, n, constant, invert,
                                1))
            # compute then adjoint restores the input: uniform x, out |0>.
            restored = np.zeros(1 << (n + 1), dtype=np.complex128)
            for value in range(1 << n):
                restored[value] += norm
            np.testing.assert_allclose(roundtrip, restored, atol=1e-10)


# ----------------------------------------------------------------------
# Register-register >= / > comparators (CDKM)
# ----------------------------------------------------------------------
#
# Layout: a at bits [0, n), b at [n, 2n), carry at 2n, out at 2n + 1.


@cudaq.kernel
def _run_cmp_register_basis(n: int, aval: int, bval: int, flag_init: int,
                            strict: int):
    a = cudaq.qvector(n)
    b = cudaq.qvector(n)
    carry = cudaq.qvector(1)
    out = cudaq.qvector(1)
    for k in range(n):
        if ((aval >> k) & 1) == 1:
            x(a[k])
        if ((bval >> k) & 1) == 1:
            x(b[k])
    if flag_init == 1:
        x(out[0])
    if strict == 0:
        arith.cmp_ge_register(a, b, carry, out)
    else:
        arith.cmp_gt_register(a, b, carry, out)


@cudaq.kernel
def _run_cmp_register_superposed(n: int, strict: int):
    a = cudaq.qvector(n)
    b = cudaq.qvector(n)
    carry = cudaq.qvector(1)
    out = cudaq.qvector(1)
    for k in range(n):
        h(a[k])
        h(b[k])
    if strict == 0:
        arith.cmp_ge_register(a, b, carry, out)
    else:
        arith.cmp_gt_register(a, b, carry, out)


@cudaq.kernel
def _run_cmp_register_twice(n: int, angles: list[float], strict: int,
                            apply_ops: int):
    a = cudaq.qvector(n)
    b = cudaq.qvector(n)
    carry = cudaq.qvector(1)
    out = cudaq.qvector(1)
    for k in range(n):
        ry(angles[k], a[k])
        ry(angles[n + k], b[k])
        cx(a[k], b[k])  # entangle a with b
    ry(angles[2 * n], out[0])
    if apply_ops == 1:
        if strict == 0:
            arith.cmp_ge_register(a, b, carry, out)
            arith.cmp_ge_register(a, b, carry, out)
        else:
            arith.cmp_gt_register(a, b, carry, out)
            arith.cmp_gt_register(a, b, carry, out)


_CMP_REGISTER_OPS = [
    ("ge", 0, lambda a, b: a >= b),
    ("gt", 1, lambda a, b: a > b),
]


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("name,strict,predicate", _CMP_REGISTER_OPS)
def test_cmp_register_truth_table_exhaustive(n, name, strict, predicate):
    # All (a, b) pairs, flag initially 0 and 1: XOR semantics, operands
    # (and carry) verified unchanged by the full statevector equality.
    for aval in range(1 << n):
        for bval in range(1 << n):
            for flag_init in (0, 1):
                state = np.array(
                    cudaq.get_state(_run_cmp_register_basis, n, aval, bval,
                                    flag_init, strict))
                flag = flag_init ^ int(predicate(aval, bval))
                index = aval + (bval << n) + (flag << (2 * n + 1))
                np.testing.assert_allclose(state,
                                           _basis(index, 2 * n + 2),
                                           atol=1e-12)


@pytest.mark.parametrize("n", WIDTHS)
@pytest.mark.parametrize("name,strict,predicate", _CMP_REGISTER_OPS)
def test_cmp_register_all_inputs_superposed(n, name, strict, predicate):
    state = np.array(cudaq.get_state(_run_cmp_register_superposed, n, strict))
    expected = np.zeros(1 << (2 * n + 2), dtype=np.complex128)
    norm = 1.0 / (1 << n)
    for aval in range(1 << n):
        for bval in range(1 << n):
            index = aval + (bval << n) + (int(predicate(aval, bval)) <<
                                          (2 * n + 1))
            expected[index] += norm
    np.testing.assert_allclose(state, expected, atol=1e-12)


@pytest.mark.parametrize("n", WIDTHS)
@pytest.mark.parametrize("name,strict,predicate", _CMP_REGISTER_OPS)
def test_cmp_register_twice_is_identity(n, name, strict, predicate):
    # Self-inverse contract: the operand action is compute-copy-uncompute
    # and the flag is XOR-accumulated, so two applications are the
    # identity — checked on a random entangled state (superposed flag
    # included: XOR-loading is a permutation, so linearity carries it).
    rng = np.random.default_rng(20260911 + 8 * n + strict)
    angles = rng.uniform(0.1, 3.0, size=2 * n + 1).tolist()
    twice = np.array(
        cudaq.get_state(_run_cmp_register_twice, n, angles, strict, 1))
    reference = np.array(
        cudaq.get_state(_run_cmp_register_twice, n, angles, strict, 0))
    np.testing.assert_allclose(twice, reference, atol=1e-12)


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
# - cmp_ge_register / cmp_gt_register: the same MAJ sweep + reversal
#   = 2 n; the b-complement, carry-in set and flag copy are X/CNOT-only.
#
# Draper QFT derivations: no Toffolis at all. add_constant_qft is
# qft + phases + iqft = 2 * (n(n-1)/2) controlled-r1, n free r1 and 2 n
# H; each side of the cmp_ge_constant_qft_shift pair is one extended
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
def _res_cmp_register(n: int, strict: int):
    a = cudaq.qvector(n)
    b = cudaq.qvector(n)
    carry = cudaq.qvector(1)
    out = cudaq.qvector(1)
    if strict == 0:
        arith.cmp_ge_register(a, b, carry, out)
    else:
        arith.cmp_gt_register(a, b, carry, out)


@cudaq.kernel
def _res_add_constant_qft(n: int, constant: int):
    target = cudaq.qvector(n)
    arith.add_constant_qft(target, constant)


@cudaq.kernel
def _res_cmp_ge_constant_qft_shift(n: int, constant: int, invert: int):
    x_reg = cudaq.qvector(n)
    out = cudaq.qvector(1)
    arith.cmp_ge_constant_qft_shift(x_reg, out, constant, invert)


def _toffolis(kernel, *args) -> int:
    # A Toffoli is an x with two controls; ``count_controls`` is the
    # arity-aware accessor (``count("ccx")`` matches nothing — the
    # display name is not the lookup key and returns 0).
    return cudaq.estimate_resources(kernel, *args).count_controls("x", 2)


# The mod-2^n adder never reads the top carry-out, so its MAJ/UMA Toffoli pair
# cancels -> 2n - 2 Toffolis (n = 1 needs none). Cuccaro et al.
# (arXiv:quant-ph/0410184) Section 4.1 reaches 2n - 3; not adopted here.

@_RESOURCES
@pytest.mark.parametrize("n", WIDTHS)
def test_cdkm_register_ops_cost_2n_minus_2_toffolis(n):
    assert _toffolis(_res_add_register, n, 0) == 2 * n - 2
    assert _toffolis(_res_add_register, n, 1) == 2 * n - 2


@_RESOURCES
@pytest.mark.parametrize("n", WIDTHS)
def test_cdkm_constant_ops_cost_2n_minus_2_toffolis(n):
    bits = _bits((1 << n) - 1, n)  # worst-case load: every bit set
    assert _toffolis(_res_add_constant, n, bits, 0) == 2 * n - 2
    assert _toffolis(_res_add_constant, n, bits, 1) == 2 * n - 2


@_RESOURCES
@pytest.mark.parametrize("n", WIDTHS)
def test_cdkm_comparator_cost_2n_minus_1_toffolis(n):
    # The comparator does read the top carry (it is the result), so only the
    # compute/copy/uncompute of that one carry collapses: 2n - 1 Toffolis.
    complement = _bits((1 << n) - 1, n)  # K = 1
    assert _toffolis(_res_cmp_ge_constant, n, complement, 0) == 2 * n - 1
    # K = 0 short-circuits to a single X: no Toffolis.
    assert _toffolis(_res_cmp_ge_constant, n, [0] * n, 1) == 0


@_RESOURCES
@pytest.mark.parametrize("n", WIDTHS + [8])
def test_cmp_register_cost_2n_minus_1_toffolis(n):
    # Widths past the truth-table range (5 and 8) included: the count is
    # a function of the runtime width argument, never a folded constant.
    assert _toffolis(_res_cmp_register, n, 0) == 2 * n - 1
    assert _toffolis(_res_cmp_register, n, 1) == 2 * n - 1


@_RESOURCES
def test_cdkm_toffoli_cost_is_affine_with_slope_two():
    # 2n - 2 is affine with slope 2: doubling the width from n to 2n adds
    # exactly 2n Toffolis. (An exact "doubling doubles the count" law would
    # require the 2n form, i.e. keeping the unused top-carry Toffoli.)
    compiled = {n: _toffolis(_res_add_register, n, 0) for n in (2, 4, 8, 16)}
    for n in (2, 4, 8):
        assert compiled[2 * n] - compiled[n] == 2 * n, compiled


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
        resources = cudaq.estimate_resources(_res_cmp_ge_constant_qft_shift, n, 1,
                                             invert)
        assert resources.count_controls("x", 2) == 0
        assert resources.count_controls("r1", 1) == n * (n + 1)
        assert resources.count_controls("r1", 0) == n + 1
        assert resources.count("h") == 2 * (n + 1)
        assert resources.count("x") == (1 if invert == 0 else 0)
    # K = 0 emits no rotations at all: the constant-true comparator is a
    # bare X on the out qubit.
    trivial = cudaq.estimate_resources(_res_cmp_ge_constant_qft_shift, n, 0, 0)
    assert trivial.count("r1") == 0
    assert trivial.count("x") == 1


# ----------------------------------------------------------------------
# Controlled composition (cudaq.control of a kernel that calls a sub-kernel)
# ----------------------------------------------------------------------
#
# cudaq.control of a device kernel that calls another @cudaq.kernel regressed
# on CUDA-Q 0.15.x (kernel-specialization failure: "Unhandled controlled
# quantum kernel call"); it is correct on 0.14.2 and again on 0.16.0. Since the
# library targets 0.16, this pins the controlled-composition behaviour there and
# skips on the affected 0.15.x line.


def _cudaq_version():
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", cudaq.__version__)
    return tuple(int(part) for part in match.groups()) if match else (0, 0, 0)


_CONTROLLED_COMPOSITION_OK = _cudaq_version() >= (0, 16, 0)


@cudaq.kernel
def _run_controlled_add_constant_qft(n: int, tval: int, constant: int):
    ctrl = cudaq.qubit()
    target = cudaq.qvector(n)
    h(ctrl)                                  # control in a uniform superposition
    for k in range(n):
        if ((tval >> k) & 1) == 1:
            x(target[k])
    # add_constant_qft calls qft / iqft, so this exercises a controlled call.
    cudaq.control(arith.add_constant_qft, ctrl, target, constant)


@pytest.mark.skipif(
    not _CONTROLLED_COMPOSITION_OK,
    reason="cudaq.control of a composed kernel regressed on 0.15.x "
           "(specialization failure); correct on 0.14.2 and 0.16.0")
@pytest.mark.parametrize("n", [2, 3])
def test_controlled_add_constant_qft_superposed_control(n):
    # A superposed control: the branch that fires must, and the other must not.
    # If control silently applied on both branches, the ctrl = 0 component would
    # be shifted too and this would fail.
    for tval in range(1 << n):
        for constant in range(1 << n):
            state = np.array(cudaq.get_state(
                _run_controlled_add_constant_qft, n, tval, constant))
            expected = np.zeros(1 << (n + 1), dtype=np.complex128)
            amp = 1.0 / np.sqrt(2)
            expected[tval << 1] += amp                                # ctrl = 0
            expected[1 + (((tval + constant) % (1 << n)) << 1)] += amp  # ctrl = 1
            np.testing.assert_allclose(state, expected, atol=1e-10)
