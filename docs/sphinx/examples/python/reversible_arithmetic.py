#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# [Begin Documentation]
"""Reversible modular arithmetic, comparison flags, and clean workspace.

Many quantum algorithms make decisions from numbers stored in qubits: add an
offset, compare a value with a threshold, or mark the values that satisfy a
condition. This example builds those operations as reversible circuits. Its
threshold demonstration computes a yes/no flag, uses that flag to place a
relative minus sign on the matching input, and then undoes the comparison so
the flag and scratch qubits can be reused.

The lessons are to track what every register means, account for wraparound in
fixed-width arithmetic, and undo temporary calculations after using their
result. Addition and subtraction, two constant-arithmetic constructions, and
unsigned comparisons are checked against classical predictions on
superpositions. You should be able to distinguish a correct numerical output
from a complete reversible operation that also preserves phases and restores
its workspace, and read the resource counts as the cost of those requirements.

Run with: python3 reversible_arithmetic.py
Add --verbose to display the numerical error for every validation check.
Register bit zero is least significant. All views passed to an operation
are disjoint. Arithmetic wraps modulo 2**n; it does not report overflow.
"""

import argparse
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
def threshold_oracle_stages(initial: cudaq.State, complement_bits: list[int],
                            stage: int):
    """Run a prefix of compare, phase mark, uncompute; no measurements."""
    logical = cudaq.qvector(initial)  # [x(n) | flag(1)]
    n = logical.size() - 1
    work = cudaq.qvector(n)
    carry = cudaq.qvector(1)
    if stage >= 1:
        cmp_ge_constant(logical[0:n], complement_bits, 0, work, carry,
                        logical[n:n + 1])
    if stage >= 2:
        z(logical[n])
    if stage >= 3:
        cmp_ge_constant(logical[0:n], complement_bits, 0, work, carry,
                        logical[n:n + 1])


def show_threshold_oracle(verbose=False):
    """Display actual simulated amplitudes for a two-value threshold oracle."""
    n = 3
    modulus = 1 << n
    initial = np.zeros(2 * modulus, dtype=complex)
    initial[[2, 5]] = 1 / np.sqrt(2)
    # cmp_ge_constant takes the n low bits of 2**n - K, here 8 - 3 = 5.
    complement_bits = [1, 0, 1]
    snapshots = (
        ("Input", ((2, 0, 1), (5, 0, 1))),
        ("Compare x >= 3", ((2, 0, 1), (5, 1, 1))),
        ("Z on flag", ((2, 0, 1), (5, 1, -1))),
        ("Uncompute", ((2, 0, 1), (5, 0, -1))),
    )
    print("\nA useful comparison: a threshold phase oracle")
    print("Kets below are |x>|flag>; work/carry are shown separately.")
    print("Each row inspects a fresh simulator run of the circuit prefix.")
    for stage, (label, branches) in enumerate(snapshots):
        actual = simulate(threshold_oracle_stages, initial, complement_bits,
                          stage)
        expected = np.zeros_like(actual)
        for value, flag, sign in branches:
            expected[value + (flag << n)] = sign / np.sqrt(2)
        # Compare all eight qubits: every unlisted amplitude must be zero,
        # including any population in the work or carry registers.
        np.testing.assert_allclose(actual,
                                   expected,
                                   rtol=0,
                                   atol=ATOL,
                                   err_msg=f"threshold oracle: {label}")
        terms = []
        for index in np.flatnonzero(np.abs(actual) > ATOL):
            value, flag = index & (modulus - 1), (index >> n) & 1
            sign = "-" if actual[index].real < 0 else "+"
            terms.append(f"{sign} |{value}>|{flag}>")
        expression = " ".join(terms).removeprefix("+ ")
        print(f"  {label:15s}: ({expression}) / sqrt(2)   work/carry=|0000>")
    probabilities = np.abs(actual[:modulus])**2
    np.testing.assert_allclose(probabilities[[2, 5]], [.5, .5],
                               rtol=0,
                               atol=ATOL)
    overlap = abs(np.vdot(embed(initial, n + 1), actual))
    np.testing.assert_allclose(overlap, 0., rtol=0, atol=ATOL)
    print("The values still have probability 1/2 each; only their relative "
          "phase changed.")
    print("The final state is orthogonal to the input, so interference can "
          "distinguish them.")
    print("The flag and all work/carry qubits return to zero; the oracle "
          "can be reused.")
    if verbose:
        print(
            f"All four full-state references verified at tolerance {ATOL:.1e}."
        )


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


def main(verbose=False):

    def check(label, actual, expected):
        error = float(np.max(np.abs(actual - expected)))
        passed = np.isfinite(error) and error <= ATOL
        if verbose or not passed:
            print(f"CHECK {label}: {error:.3e} <= {ATOL:.1e} -> "
                  f"{'PASS' if passed else 'FAIL'}")
        if not passed:
            raise AssertionError(label)

    start = time.perf_counter()
    cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))
    rng = np.random.default_rng(7)
    n = 3
    modulus = 1 << n
    mask = modulus - 1
    print("Three-bit unsigned operands; arithmetic is modulo 8.")
    print("All carry/work qubits start at zero and must return to zero.")
    show_threshold_oracle(verbose)
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
    print("All arithmetic, comparison, and workspace checks passed.")
    print(f"Maximum executed qubits: {2*n+2}; "
          f"elapsed: {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show the numerical error for every validation check")
    main(verbose=parser.parse_args().verbose)
