# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reversible in-place integer arithmetic device kernels.

Two families: CDKM/Cuccaro ripple-carry (``arXiv:quant-ph/0410184``) and Draper
QFT arithmetic (``arXiv:quant-ph/0008033``). Both are little-endian --
``register[0]`` is the least-significant bit (see ``docs/conventions.md``).
Per-operation gate costs are documented on each kernel and pinned by the
resource-contract tests in ``tests/python/test_primitives_arithmetic.py``.

Registers passed to one op must be pairwise disjoint: the kernels never
alias-check, and overlapping views (e.g. ``add_register(a, a, carry)``, or a
``work`` view sharing qubits with ``target``) are undefined.

Every inverse is hand-written -- ``cudaq.adjoint`` is off-limits as of CUDA-Q
0.15 (cuda-quantum#4897/#4898) -- and pinned by op-then-inverse identity tests.
"""

from __future__ import annotations

import cudaq

__all__ = [
    "add_register",
    "subtract_register",
    "add_constant",
    "subtract_constant",
    "cmp_ge_constant",
    "cmp_ge_register",
    "cmp_gt_register",
    "qft",
    "iqft",
    "phase_add_constant",
    "add_constant_qft",
    "subtract_constant_qft",
    "cmp_ge_constant_qft_shift",
    "cmp_ge_constant_qft_shift_adj",
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

    Cost: ``2n - 2`` Toffolis. The mod-2^n adder never reads the top carry-out,
    so its MAJ/UMA Toffoli pair (and the surrounding CX pair) cancels;
    ``n = 1`` needs none. Cuccaro et al. (arXiv:quant-ph/0410184) Section 4.1
    reaches ``2n - 3`` via an ``(n-1)``-bit adder plus a final CNOT -- one
    further Toffoli, not taken here to keep the plain MAJ/UMA structure.
    """
    n = a.size()
    if n == 1:
        # b <- a XOR b; the carry-out is discarded (mod 2), so no Toffoli.
        cx(a[0], b[0])
    elif n > 1:
        # MAJ sweep over the lower bits.
        cx(a[0], b[0])
        cx(a[0], carry[0])
        x.ctrl(carry[0], b[0], a[0])
        for i in range(1, n - 1):
            cx(a[i], b[i])
            cx(a[i], a[i - 1])
            x.ctrl(a[i - 1], b[i], a[i])
        # Top bit: the carry-out is never read for a mod-2^n adder, so the
        # MAJ Toffoli at i = n-1 and its UMA reverse cancel, and the CX pair
        # around them with it -- what survives is two CX into b[n-1].
        cx(a[n - 1], b[n - 1])
        cx(a[n - 2], b[n - 1])
        # UMA sweep (reverse) over the lower bits.
        for k in range(2, n):
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

    Cost: ``2n - 2`` Toffolis.
    """
    n = a.size()
    if n == 1:
        cx(a[0], b[0])
    elif n > 1:
        cx(carry[0], b[0])
        cx(a[0], carry[0])
        x.ctrl(carry[0], b[0], a[0])
        for i in range(1, n - 1):
            cx(a[i - 1], b[i])
            cx(a[i], a[i - 1])
            x.ctrl(a[i - 1], b[i], a[i])
        cx(a[n - 2], b[n - 1])
        cx(a[n - 1], b[n - 1])
        for i in range(n - 2, 0, -1):
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

    Cost: ``2n - 2`` Toffolis (the constant load/unload is X-only). An
    ancilla-light constant adder (Haener et al., arXiv:1611.07995) that avoids
    the ``n``-qubit ``work`` register is a possible follow-up.
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
    """``target <- (target - K) mod 2^n``: inverse of ``add_constant``.

    Cost: ``2n - 2`` Toffolis.
    """
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
    - K))`` for ``K >= 1``: MAJ sweep, write the carry-out into ``out``, then
    reverse the MAJ sweep so ``x_reg``, ``work`` and ``carry`` are all
    restored.

    This is the ``x``-preserving, ``out``-XOR-ing, complement-bit comparator;
    contrast ``cmp_ge_constant_qft_shift``, which shifts ``x``, sets ``out``
    from |0>, and takes ``K`` as an integer.

    Cost: ``2n - 1`` Toffolis for ``K >= 1`` (the top carry is written straight
    into ``out``, saving one Toffoli over compute/copy/uncompute); ``0`` for
    ``K = 0``.
    """
    if k_is_zero == 1:
        x(out[0])
    else:
        n = x_reg.size()
        for k in range(n):
            if complement_bits[k] == 1:
                x(work[k])
        if n == 1:
            # carry_out(x + (2 - K)) = x AND (2 - K), and 2 - K is X-loaded
            # into work[0], so the result is a single Toffoli.
            x.ctrl(x_reg[0], work[0], out[0])
        else:
            # MAJ sweep (a = work, b = x_reg) over the lower bits.
            cx(work[0], x_reg[0])
            cx(work[0], carry[0])
            x.ctrl(carry[0], x_reg[0], work[0])
            for i in range(1, n - 1):
                cx(work[i], x_reg[i])
                cx(work[i], work[i - 1])
                x.ctrl(work[i - 1], x_reg[i], work[i])
            # Top bit: write the carry-out straight into ``out`` rather than
            # computing it into work[n-1], copying it, and uncomputing it --
            # one Toffoli instead of two, and work[n-1] is left untouched.
            cx(work[n - 1], x_reg[n - 1])
            cx(work[n - 1], work[n - 2])
            cx(work[n - 1], out[0])
            x.ctrl(work[n - 2], x_reg[n - 1], out[0])
            cx(work[n - 1], work[n - 2])
            cx(work[n - 1], x_reg[n - 1])
            # Reverse MAJ sweep over the lower bits (restores x_reg, work, carry).
            for k in range(2, n):
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


@cudaq.kernel
def cmp_ge_register(a: cudaq.qview, b: cudaq.qview, carry: cudaq.qview,
                    out: cudaq.qview):
    """``out[0] ^= (a >= b)`` (CDKM, unsigned), leaving ``a``, ``b`` unchanged.

    ``a`` and ``b`` have equal size ``n >= 1``; ``carry`` is a one-qubit
    view that must be |0> on entry (it is returned to |0>); ``out`` is the
    one-qubit flag, XOR-loaded (any input state is allowed). ``a``, ``b``,
    ``carry`` and ``out`` must be pairwise disjoint (module precondition).

    Uses the ripple identity ``a >= b <=> carry_out(a + (2^n - b))`` with
    ``2^n - b = ~b + 1``: complement ``b`` in place and set the carry-in
    (so ``b`` doubles as the constant-load work register — no extra
    ``n``-qubit work is needed, unlike ``cmp_ge_constant``), MAJ sweep,
    write the carry-out into ``out``, reverse the MAJ sweep and undo the
    complement so ``a``, ``b`` and ``carry`` are all restored. Because the
    flag is XOR-accumulated and the operand action is compute-uncompute,
    applying the kernel twice with the same arguments is the identity
    (self-inverse).

    Cost: ``2n - 1`` Toffolis (the top carry is written straight into ``out``,
    as in ``cmp_ge_constant``; ``n = 1`` needs one).
    """
    n = a.size()
    if n == 1:
        # a >= b  ==  NOT(~a AND b): one Toffoli.
        x(out[0])
        x(a[0])
        x.ctrl(a[0], b[0], out[0])
        x(a[0])
    elif n > 1:
        # 2^n - b = ~b + 1: complement b in place, set the carry-in to 1.
        for k in range(n):
            x(b[k])
        x(carry[0])
        # MAJ sweep of a + ~b + 1 over the lower bits (b accumulates carries).
        cx(b[0], a[0])
        cx(b[0], carry[0])
        x.ctrl(carry[0], a[0], b[0])
        for i in range(1, n - 1):
            cx(b[i], a[i])
            cx(b[i], b[i - 1])
            x.ctrl(b[i - 1], a[i], b[i])
        # Top bit: write the carry-out straight into ``out`` (one Toffoli
        # instead of compute-into-b / copy / uncompute); b[n-1] is untouched.
        cx(b[n - 1], a[n - 1])
        cx(b[n - 1], b[n - 2])
        cx(b[n - 1], out[0])
        x.ctrl(b[n - 2], a[n - 1], out[0])
        cx(b[n - 1], b[n - 2])
        cx(b[n - 1], a[n - 1])
        # Reverse MAJ sweep over the lower bits (restores a, ~b, carry-in).
        for k in range(2, n):
            i = n - k
            x.ctrl(b[i - 1], a[i], b[i])
            cx(b[i], b[i - 1])
            cx(b[i], a[i])
        x.ctrl(carry[0], a[0], b[0])
        cx(b[0], carry[0])
        cx(b[0], a[0])
        x(carry[0])
        for k in range(n):
            x(b[k])


@cudaq.kernel
def cmp_gt_register(a: cudaq.qview, b: cudaq.qview, carry: cudaq.qview,
                    out: cudaq.qview):
    """``out[0] ^= (a > b)`` (CDKM, unsigned), leaving ``a``, ``b`` unchanged.

    Free strict variant of ``cmp_ge_register`` via ``a > b <=> not
    (b >= a)``: an X on ``out`` plus the ``>=`` comparator with the roles
    swapped. Same preconditions, same ``2n - 1`` Toffoli price, and likewise
    self-inverse.
    """
    x(out[0])
    cmp_ge_register(b, a, carry, out)


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
    k = constant % (1 << n)
    for t in range(n):
        # Only ``K mod 2^(t+1)`` affects reg[t]'s phase. Reduce it as an integer
        # first so the rotation argument stays in [0, 2*pi) and never loses
        # precision to a large float -- exact for any K, including K >= 2^53.
        kt = k % (1 << (t + 1))
        r1(6.283185307179586 * kt / (1 << (t + 1)), reg[t])


@cudaq.kernel
def add_constant_qft(reg: cudaq.qview, constant: int):
    """``reg <- (reg + K) mod 2^n`` (Draper; no work qubits).

    Cost: no Toffolis -- ``n(n-1)`` controlled-``r1`` plus ``n`` single-qubit
    ``r1`` on ``n`` bits.
    """
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
def cmp_ge_constant_qft_shift(x_reg: cudaq.qview, out: cudaq.qview,
                              constant: int, invert: int):
    """Compute ``out[0] = (x >= K)`` and **leave ``x_reg`` shifted to
    ``(x - K) mod 2^n``** (hence the ``_shift`` -- this is the compute half of a
    pair). ``invert = 1`` gives ``x < K``.

    Unlike ``cmp_ge_constant`` (which restores ``x``), this Draper comparator
    does *not* restore ``x_reg``: it must be paired with
    ``cmp_ge_constant_qft_shift_adj`` to undo the shift, and the caller may read
    ``out`` but must **not** consume ``x_reg`` in between. ``out`` must be |0>
    on entry.

    Precondition: ``0 <= K <= 2^n`` (``n = x_reg.size()``). For ``K > 2^n`` the
    result is silently wrong: the subtraction acts on the (n+1)-bit extension,
    so the borrow wraps mod ``2^(n+1)`` and ``out`` no longer encodes
    ``x < K``. Mechanically: subtract ``K`` on the (n+1)-bit register
    ``[x_reg, out]``; the MSB (``out``) becomes the borrow ``x < K``. ``K = 0``
    (with ``invert = 0``) yields the constant-true comparator.

    Cost: no Toffolis -- ``n(n+1)`` controlled-``r1`` plus ``n+1`` single-qubit
    ``r1`` per side of the compute/adjoint pair (``K >= 1``; zero at ``K = 0``).
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
def cmp_ge_constant_qft_shift_adj(x_reg: cudaq.qview, out: cudaq.qview,
                            constant: int, invert: int):
    """Hand-written inverse of ``cmp_ge_constant_qft_shift``."""
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
