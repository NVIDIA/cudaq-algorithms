#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# [Begin Documentation]
"""Reversible modular arithmetic, comparison flags, and clean workspace.

Run with: python3 reversible_arithmetic.py
Register bit zero is least significant. All views passed to an operation
are disjoint. Arithmetic wraps modulo 2**n; it does not report overflow.
"""

import os
import time

import numpy as np

import cudaq
from cudaq_algorithms import state_from
from cudaq_algorithms.primitives import (
    add_constant, add_constant_qft, add_register, cmp_ge_constant,
    cmp_ge_constant_qft_shift, cmp_ge_constant_qft_shift_adj, cmp_ge_register,
    subtract_constant, subtract_constant_qft, subtract_register)

ATOL = 1e-10


def check(label, actual, expected):
    error = float(np.max(np.abs(actual - expected)))
    print(f"CHECK {label}: {error:.3e} <= {ATOL:.1e} -> "
          f"{'PASS' if error <= ATOL else 'FAIL'}")
    if not np.isfinite(error) or error > ATOL:
        raise AssertionError(label)


def complex_ket(size, rng):
    ket = rng.normal(size=size) + 1j * rng.normal(size=size)
    return ket / np.linalg.norm(ket)


def embed(ket, extra_bits):
    expected = np.zeros(len(ket) << extra_bits, dtype=complex)
    expected[:len(ket)] = ket
    return expected


def permute(ket, mapping):
    result = np.zeros_like(ket)
    for index, amplitude in enumerate(ket):
        result[mapping(index)] += amplitude
    return result


def simulate(kernel, ket, *args):
    return np.asarray(cudaq.get_state(kernel, state_from(ket), *args))


@cudaq.kernel
def register_arithmetic(initial: cudaq.State, n: int, inverse: bool,
                        roundtrip: bool):
    operands = cudaq.qvector(initial)  # [a(n) | b(n)]
    carry = cudaq.qvector(1)  # clean on entry and exit
    if inverse:
        subtract_register(operands[0:n], operands[n:2 * n], carry)
    else:
        add_register(operands[0:n], operands[n:2 * n], carry)
    if roundtrip:
        if inverse:
            add_register(operands[0:n], operands[n:2 * n], carry)
        else:
            subtract_register(operands[0:n], operands[n:2 * n], carry)


@cudaq.kernel
def constant_cdkm(initial: cudaq.State, bits: list[int], inverse: bool,
                  roundtrip: bool):
    target = cudaq.qvector(initial)
    work = cudaq.qvector(target.size())
    carry = cudaq.qvector(1)
    if inverse:
        subtract_constant(target, bits, work, carry)
    else:
        add_constant(target, bits, work, carry)
    if roundtrip:
        if inverse:
            add_constant(target, bits, work, carry)
        else:
            subtract_constant(target, bits, work, carry)


@cudaq.kernel
def constant_qft(initial: cudaq.State, constant: int, inverse: bool,
                 roundtrip: bool):
    target = cudaq.qvector(initial)
    if inverse:
        subtract_constant_qft(target, constant)
    else:
        add_constant_qft(target, constant)
    if roundtrip:
        if inverse:
            add_constant_qft(target, constant)
        else:
            subtract_constant_qft(target, constant)


@cudaq.kernel
def register_comparison(initial: cudaq.State, n: int):
    logical = cudaq.qvector(initial)  # [a(n) | b(n) | flag(1)]
    carry = cudaq.qvector(1)
    cmp_ge_register(logical[0:n], logical[n:2 * n], carry,
                    logical[2 * n:2 * n + 1])


@cudaq.kernel
def threshold_cdkm(initial: cudaq.State, bits: list[int], k_is_zero: int,
                   phase: bool):
    logical = cudaq.qvector(initial)  # [x(n) | flag(1)]
    n = logical.size() - 1
    work = cudaq.qvector(n)
    carry = cudaq.qvector(1)
    cmp_ge_constant(logical[0:n], bits, k_is_zero, work, carry,
                    logical[n:n + 1])
    if phase:
        z(logical[n])
        cmp_ge_constant(logical[0:n], bits, k_is_zero, work, carry,
                        logical[n:n + 1])


@cudaq.kernel
def threshold_qft(initial: cudaq.State, constant: int, phase: bool,
                  roundtrip: bool):
    logical = cudaq.qvector(initial)
    n = logical.size() - 1
    cmp_ge_constant_qft_shift(logical[0:n], constant, 0, logical[n:n + 1])
    if phase:
        z(logical[n])
    if roundtrip:
        cmp_ge_constant_qft_shift_adj(logical[0:n], constant, 0,
                                      logical[n:n + 1])


@cudaq.kernel
def resource_add(n: int):
    a = cudaq.qvector(n)
    b = cudaq.qvector(n)
    carry = cudaq.qvector(1)
    add_register(a, b, carry)


@cudaq.kernel
def resource_compare(n: int):
    a = cudaq.qvector(n)
    b = cudaq.qvector(n)
    carry = cudaq.qvector(1)
    flag = cudaq.qvector(1)
    cmp_ge_register(a, b, carry, flag)


@cudaq.kernel
def resource_qft(n: int):
    target = cudaq.qvector(n)
    add_constant_qft(target, 1)


def main():
    start = time.perf_counter()
    cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))
    rng = np.random.default_rng(7)
    n = 3
    modulus = 1 << n
    mask = modulus - 1
    print("Three-bit unsigned operands; arithmetic is modulo 8.")
    print("All carry/work qubits start at zero and must return to zero.")
    print("\n a  b   (a+b) mod 8  (b-a) mod 8")
    for a, b in ((1, 2), (3, 7), (7, 7)):
        basis = np.zeros(modulus * modulus, dtype=complex)
        basis[a + (b << n)] = 1
        expected = np.zeros_like(basis)
        expected[a + (((a + b) & mask) << n)] = 1
        check(f"basis addition {a}+{b}",
              simulate(register_arithmetic, basis, n, False, False),
              embed(expected, 1))
        print(f"{a:2d} {b:2d} {((a+b) & mask):13d} {((b-a) & mask):12d}")

    # A nonuniform complex state detects swapped operations and phase errors.
    operands = complex_ket(modulus * modulus, rng)
    for inverse in (False, True):

        def mapping(index):
            a, b = index & mask, index >> n
            output = (b - a if inverse else b + a) & mask
            return a + (output << n)

        expected = embed(permute(operands, mapping), 1)
        name = "subtract" if inverse else "add"
        check(f"register {name}, a preserved, carry clean",
              simulate(register_arithmetic, operands, n, inverse, False),
              expected)
    check("register add/subtract round trip",
          simulate(register_arithmetic, operands, n, False, True),
          embed(operands, 1))

    constant = 3
    bits = [(constant >> bit) & 1 for bit in range(n)]
    target = complex_ket(modulus, rng)
    for inverse in (False, True):
        shift = -constant if inverse else constant
        expected = np.roll(target, shift)
        cdkm = simulate(constant_cdkm, target, bits, inverse, False)
        qft = simulate(constant_qft, target, constant, inverse, False)
        name = "subtract" if inverse else "add"
        check(f"CDKM constant {name}, workspace clean", cdkm,
              embed(expected, n + 1))
        check(f"QFT constant {name}", qft, expected)
        check(f"CDKM/QFT {name} agree", cdkm[:modulus], qft)
    check("CDKM constant round trip",
          simulate(constant_cdkm, target, bits, False, True),
          embed(target, n + 1))
    check("QFT constant round trip",
          simulate(constant_qft, target, constant, False, True), target)

    print("\nComparison flags are XOR-loaded, not assigned.")
    for flag_value in (0, 1):
        logical = np.zeros(2 * len(operands), dtype=complex)
        logical[flag_value * len(operands):(flag_value + 1) *
                len(operands)] = operands

        def comparison(index):
            a, b = index & mask, (index >> n) & mask
            return index ^ (int(a >= b) << (2 * n))

        check(f"register comparison, input flag={flag_value}",
              simulate(register_comparison, logical, n),
              embed(permute(logical, comparison), 1))

    # Both flag amplitudes are nonzero: the XOR contract holds coherently too.
    logical = complex_ket(2 * modulus, rng)
    phase_input = np.zeros_like(logical)
    phase_input[:modulus] = target
    for threshold in (0, constant, modulus):
        complement = (modulus - threshold) & mask
        complement_bits = [(complement >> bit) & 1 for bit in range(n)]
        expected = permute(logical, lambda i: i ^ (int(
            (i & mask) >= threshold) << n))
        check(
            f"CDKM threshold K={threshold}, coherent flag",
            simulate(threshold_cdkm, logical, complement_bits,
                     int(threshold == 0), False), embed(expected, n + 1))
        marked = phase_input.copy()
        for value in range(modulus):
            if value >= threshold:
                marked[value] *= -1
        check(
            f"CDKM compute/phase/uncompute K={threshold}",
            simulate(threshold_cdkm, phase_input, complement_bits,
                     int(threshold == 0), True), embed(marked, n + 1))

    print("\nThe QFT comparator temporarily shifts x; use only its flag.")
    shifted = permute(
        logical, lambda i: (((i & mask) - constant) & mask) + (((i >> n) ^ int(
            (i & mask) >= constant)) << n))
    check("QFT comparator intermediate shift and flag",
          simulate(threshold_qft, logical, constant, False, False), shifted)
    check("QFT comparator adjoint restores x and flag",
          simulate(threshold_qft, logical, constant, False, True), logical)
    marked = phase_input.copy()
    marked[constant:modulus] *= -1
    check("QFT compute/phase/uncompute restores workspace",
          simulate(threshold_qft, phase_input, constant, True, True), marked)
    unchanged_x = permute(logical, lambda i: i ^ modulus)
    check("QFT K=0 is unshifted and constant-true",
          simulate(threshold_qft, logical, 0, False, False), unchanged_x)

    print("\nResource-only width sweep (no input-state preparation):")
    print("width operation       qubits CCX controlled-R1 single-R1")
    for width in (2, 3, 5):
        for name, kernel, qubits, expected_ccx in (("register add",
                                                    resource_add,
                                                    2 * width + 1,
                                                    2 * width - 2),
                                                   ("register cmp",
                                                    resource_compare,
                                                    2 * width + 2,
                                                    2 * width - 1),
                                                   ("QFT add K=1",
                                                    resource_qft, width, 0)):
            resources = cudaq.estimate_resources(kernel, width)
            ccx = resources.count_controls("x", 2)
            if ccx != expected_ccx:
                raise AssertionError(f"{name}: resource contract changed")
            print(f"{width:5d} {name:15s} {qubits:6d} {ccx:3d} "
                  f"{resources.count_controls('r1', 1):13d} "
                  f"{resources.count_controls('r1', 0):9d}")
    print(
        "Zero Toffolis in QFT arithmetic does not imply zero synthesis cost.")
    print(f"Maximum executed qubits: {2*n+2}; "
          f"elapsed: {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
