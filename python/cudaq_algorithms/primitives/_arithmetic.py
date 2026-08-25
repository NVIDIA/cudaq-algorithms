# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reversible in-place integer arithmetic device kernels.

Two families, both little-endian (``register[0]`` is the least-significant
bit, matching ``docs/conventions.md``):

- **CDKM/Cuccaro ripple-carry** (`arXiv:quant-ph/0410184`):
  ``add_register`` / ``subtract_register`` (in-place ``b <- b +/- a``),
  ``add_constant`` / ``subtract_constant`` (constant loaded into a caller
  provided work register), and ``cmp_ge_constant`` (a ``x >= K`` comparator
  writing into an out qubit, leaving ``x`` untouched: MAJ sweep, copy the
  carry, reverse MAJ sweep). Toffoli prices (documented contracts, pinned
  by the resource tests in ``tests/python/test_primitives_arithmetic.py``):
  ``add_register`` / ``subtract_register`` cost exactly ``2 n`` Toffolis
  on ``n``-bit registers (``n`` in the MAJ sweep, ``n`` in the UMA sweep),
  ``add_constant`` / ``subtract_constant`` inherit the same ``2 n`` (the
  constant load is X-only), and ``cmp_ge_constant`` costs ``2 n`` for
  ``K >= 1`` (MAJ sweep plus its reversal) and ``0`` for ``K = 0``.
- **Draper QFT arithmetic** (`arXiv:quant-ph/0008033`): ``qft`` / ``iqft``
  and ``add_constant_qft``, plus the ``cmp_ge_constant_qft`` /
  ``cmp_ge_constant_qft_adj`` comparator pair. The QFT family needs *no*
  work qubits — on a statevector simulator every extra work qubit doubles
  the memory, and the CDKM constant ops need ``n + 1`` of them — and *no*
  Toffolis: its price is rotations, ``n (n - 1)`` controlled-``r1`` plus
  ``n`` free ``r1`` for ``add_constant_qft`` on ``n`` bits, and
  ``n (n + 1)`` controlled-``r1`` plus ``n + 1`` free ``r1`` per side of
  the ``cmp_ge_constant_qft`` pair (``K >= 1``; zero rotations at
  ``K = 0``), also pinned by the resource tests.

Registers passed to any one op must be pairwise disjoint: the kernels
never alias-check, and overlapping views (e.g. ``add_register(a, a,
carry)``, or a ``work`` view sharing qubits with ``target``) are
undefined.

Every inverse is hand-written (``cudaq.adjoint`` is off-limits as of
CUDA-Q 0.15: cuda-quantum#4897/#4898, still unresolved as of the 0.16
pre-release) and pinned by op-then-inverse identity tests in
``tests/python/test_primitives_arithmetic.py``, alongside exhaustive
value checks against classical integer arithmetic for widths up to 5.

Kernel-language notes (see CLAUDE.md): guards are positive ``if`` blocks
(kernel ``return`` is silently ignored, cuda-quantum#4845), and all
classical indices are recomputed from loop variables rather than carried
as mutating accumulators.
"""

from __future__ import annotations

import cudaq

__all__ = [
    "add_register",
    "subtract_register",
    "add_constant",
    "subtract_constant",
    "cmp_ge_constant",
    "qft",
    "iqft",
    "phase_add_constant",
    "add_constant_qft",
    "subtract_constant_qft",
    "cmp_ge_constant_qft",
    "cmp_ge_constant_qft_adj",
]

# ============================================================================
# CDKM / Cuccaro ripple-carry family
# ============================================================================
#
# MAJ(x, y, z) = cx(z, y); cx(z, x); ccx(x, y, z)
# UMA(x, y, z) = ccx(x, y, z); cx(z, x); cx(x, y)
#
# The adder chains MAJ(carry, b0, a0), MAJ(a0, b1, a1), ...; after the MAJ
# sweep the ripple carry-out sits on a[n-1]; the UMA sweep (in reverse)
# restores ``a`` and the carry ancilla and completes ``b <- a + b mod 2^n``.


@cudaq.kernel
def add_register(a: cudaq.qview, b: cudaq.qview, carry: cudaq.qview):
    """``b <- (a + b) mod 2^n`` (CDKM). ``a`` and ``carry`` are restored.

    ``a`` and ``b`` have equal size ``n``; ``carry`` is a one-qubit view
    that must be |0> on entry (it is returned to |0>).
    """
    n = a.size()
    if n > 0:
        # MAJ sweep.
        cx(a[0], b[0])
        cx(a[0], carry[0])
        x.ctrl(carry[0], b[0], a[0])
        for i in range(1, n):
            cx(a[i], b[i])
            cx(a[i], a[i - 1])
            x.ctrl(a[i - 1], b[i], a[i])
        # UMA sweep (reverse).
        for k in range(1, n):
            i = n - k
            x.ctrl(a[i - 1], b[i], a[i])
            cx(a[i], a[i - 1])
            cx(a[i - 1], b[i])
        x.ctrl(carry[0], b[0], a[0])
        cx(a[0], carry[0])
        cx(carry[0], b[0])


@cudaq.kernel
def subtract_register(a: cudaq.qview, b: cudaq.qview, carry: cudaq.qview):
    """``b <- (b - a) mod 2^n``: the exact gate-reversal of ``add_register``.

    Hand-written inverse (no ``cudaq.adjoint``); every gate of the adder is
    self-inverse, so the reversed sequence is the inverse circuit.
    """
    n = a.size()
    if n > 0:
        cx(carry[0], b[0])
        cx(a[0], carry[0])
        x.ctrl(carry[0], b[0], a[0])
        for i in range(1, n):
            cx(a[i - 1], b[i])
            cx(a[i], a[i - 1])
            x.ctrl(a[i - 1], b[i], a[i])
        for k in range(1, n):
            i = n - k
            x.ctrl(a[i - 1], b[i], a[i])
            cx(a[i], a[i - 1])
            cx(a[i], b[i])
        x.ctrl(carry[0], b[0], a[0])
        cx(a[0], carry[0])
        cx(a[0], b[0])


@cudaq.kernel
def add_constant(target: cudaq.qview, constant_bits: list[int],
                 work: cudaq.qview, carry: cudaq.qview):
    """``target <- (target + K) mod 2^n`` (CDKM).

    ``constant_bits[k]`` is bit ``k`` of ``K`` (little-endian). Precondition:
    ``constant_bits`` has length exactly ``n = target.size()``, i.e.
    ``0 <= K < 2^n`` — a shorter or longer list is not truncated or padded
    and the kernel is undefined. ``work`` (``n`` qubits) and ``carry`` (1
    qubit) must be |0> on entry and are returned to |0>: the constant is
    X-loaded into ``work``, ripple-added, and X-unloaded.
    """
    n = target.size()
    for k in range(n):
        if constant_bits[k] == 1:
            x(work[k])
    add_register(work, target, carry)
    for k in range(n):
        if constant_bits[k] == 1:
            x(work[k])


@cudaq.kernel
def subtract_constant(target: cudaq.qview, constant_bits: list[int],
                      work: cudaq.qview, carry: cudaq.qview):
    """``target <- (target - K) mod 2^n``: inverse of ``add_constant``."""
    n = target.size()
    for k in range(n):
        if constant_bits[k] == 1:
            x(work[k])
    subtract_register(work, target, carry)
    for k in range(n):
        if constant_bits[k] == 1:
            x(work[k])


@cudaq.kernel
def cmp_ge_constant(x_reg: cudaq.qview, complement_bits: list[int],
                    k_is_zero: int, work: cudaq.qview, carry: cudaq.qview,
                    out: cudaq.qview):
    """``out[0] ^= (x >= K)`` (CDKM), leaving ``x_reg`` unchanged.

    ``complement_bits`` are the little-endian bits of ``2^n - K``.
    Precondition: ``complement_bits`` has length exactly ``n =
    x_reg.size()`` and ``0 <= K <= 2^n`` (so ``2^n - K`` fits in ``n``
    bits; pass ``k_is_zero = 1`` and all-zero bits for ``K = 0``, which is
    always true, and all-zero bits with ``k_is_zero = 0`` for ``K = 2^n``,
    which is always false). Uses the ripple identity ``x >= K <=> carry_out(x + (2^n
    - K))`` for ``K >= 1``: MAJ sweep, copy the carry-out (which sits on
    the top work qubit) into ``out``, then reverse the MAJ sweep so
    ``x_reg``, ``work`` and ``carry`` are all restored.
    """
    if k_is_zero == 1:
        x(out[0])
    else:
        n = x_reg.size()
        for k in range(n):
            if complement_bits[k] == 1:
                x(work[k])
        # MAJ sweep (a = work, b = x_reg).
        cx(work[0], x_reg[0])
        cx(work[0], carry[0])
        x.ctrl(carry[0], x_reg[0], work[0])
        for i in range(1, n):
            cx(work[i], x_reg[i])
            cx(work[i], work[i - 1])
            x.ctrl(work[i - 1], x_reg[i], work[i])
        cx(work[n - 1], out[0])
        # Reverse MAJ sweep (restores x_reg, work, carry).
        for k in range(1, n):
            i = n - k
            x.ctrl(work[i - 1], x_reg[i], work[i])
            cx(work[i], work[i - 1])
            cx(work[i], x_reg[i])
        x.ctrl(carry[0], x_reg[0], work[0])
        cx(work[0], carry[0])
        cx(work[0], x_reg[0])
        for k in range(n):
            if complement_bits[k] == 1:
                x(work[k])


# ============================================================================
# Draper QFT family (no work qubits)
# ============================================================================
#
# ``qft`` is the textbook circuit without the final bit-reversal swaps;
# after it, qubit t holds (|0> + exp(2 pi i x / 2^(t+1)) |1>)/sqrt(2), so a
# constant K is added by the single-qubit phases r1(2 pi K / 2^(t+1)).


@cudaq.kernel
def qft(reg: cudaq.qview):
    """Quantum Fourier transform (no bit-reversal swaps; see module doc)."""
    n = reg.size()
    for j in range(n):
        t = n - 1 - j
        h(reg[t])
        for c in range(t):
            r1.ctrl(3.141592653589793 / (1 << (t - c)), reg[c], reg[t])


@cudaq.kernel
def iqft(reg: cudaq.qview):
    """Inverse QFT: hand-written reversal of ``qft``."""
    n = reg.size()
    for t in range(n):
        for k in range(t):
            c = t - 1 - k
            r1.ctrl(-3.141592653589793 / (1 << (t - c)), reg[c], reg[t])
        h(reg[t])


@cudaq.kernel
def phase_add_constant(reg: cudaq.qview, constant: int):
    """``reg <- reg + K mod 2^n`` in the Fourier basis (between qft/iqft)."""
    n = reg.size()
    for t in range(n):
        r1(6.283185307179586 * constant / (1 << (t + 1)), reg[t])


@cudaq.kernel
def add_constant_qft(reg: cudaq.qview, constant: int):
    """``reg <- (reg + K) mod 2^n`` (Draper; no work qubits)."""
    qft(reg)
    phase_add_constant(reg, constant)
    iqft(reg)


@cudaq.kernel
def subtract_constant_qft(reg: cudaq.qview, constant: int):
    """``reg <- (reg - K) mod 2^n``: inverse of ``add_constant_qft``."""
    qft(reg)
    phase_add_constant(reg, -constant)
    iqft(reg)


@cudaq.kernel
def _qft_extended(x_reg: cudaq.qview, msb: cudaq.qview):
    """QFT over the (n+1)-bit register ``[x_reg, msb]`` (msb = bit n)."""
    n = x_reg.size()
    h(msb[0])
    for c in range(n):
        r1.ctrl(3.141592653589793 / (1 << (n - c)), x_reg[c], msb[0])
    qft(x_reg)


@cudaq.kernel
def _iqft_extended(x_reg: cudaq.qview, msb: cudaq.qview):
    """Inverse of ``_qft_extended``."""
    n = x_reg.size()
    iqft(x_reg)
    for k in range(n):
        c = n - 1 - k
        r1.ctrl(-3.141592653589793 / (1 << (n - c)), x_reg[c], msb[0])
    h(msb[0])


@cudaq.kernel
def cmp_ge_constant_qft(x_reg: cudaq.qview, out: cudaq.qview, constant: int,
                        invert: int):
    """Compute ``out[0] = (x >= K)`` (or ``x < K`` with ``invert = 1``).

    Precondition: ``0 <= K <= 2^n`` (``n = x_reg.size()``). For ``K >
    2^n`` the result is silently wrong: the subtraction acts on the
    (n+1)-bit extension, so the borrow wraps mod ``2^(n+1)`` and ``out``
    no longer encodes ``x < K``.

    Draper-style: subtract ``K`` on the (n+1)-bit register ``[x_reg,
    out]``; the MSB (``out``) becomes the borrow, i.e. ``x < K``. ``out``
    must be |0> on entry. Between this kernel and
    ``cmp_ge_constant_qft_adj`` the low bits hold ``(x - K) mod 2^n`` —
    callers may read ``out`` but must not consume ``x_reg`` until the
    adjoint restores it. ``K = 0`` (with ``invert = 0``) yields the
    constant-true comparator.
    """
    n = x_reg.size()
    if constant > 0:
        _qft_extended(x_reg, out)
        for t in range(n):
            r1(-6.283185307179586 * constant / (1 << (t + 1)), x_reg[t])
        r1(-6.283185307179586 * constant / (1 << (n + 1)), out[0])
        _iqft_extended(x_reg, out)
    if invert == 0:
        x(out[0])


@cudaq.kernel
def cmp_ge_constant_qft_adj(x_reg: cudaq.qview, out: cudaq.qview,
                            constant: int, invert: int):
    """Hand-written inverse of ``cmp_ge_constant_qft``."""
    n = x_reg.size()
    if invert == 0:
        x(out[0])
    if constant > 0:
        _qft_extended(x_reg, out)
        r1(6.283185307179586 * constant / (1 << (n + 1)), out[0])
        for k in range(n):
            t = n - 1 - k
            r1(6.283185307179586 * constant / (1 << (t + 1)), x_reg[t])
        _iqft_extended(x_reg, out)
