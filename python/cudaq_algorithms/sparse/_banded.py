# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefab sparse-access oracles for banded matrices.

``banded_oracles`` compiles a band description — diagonal offsets and one
value per band — into an ``OracleKernels`` bundle for
``SparseOracleEncoding``: the location oracle is a slot-multiplexed
constant adder (Draper QFT form: no work qubits), and the value oracle
selects each band's precomputed fixed-point angle behind a slot match and
a boundary comparator (bands are *not* periodic: elements that would wrap
around the matrix edge are encoded as zero).

The kernels are deliberately *flat* — the QFT and comparator circuits of
``cudaq_algorithms.primitives._arithmetic`` are inlined rather than
called: ``from_general_oracles`` wraps these oracles in
``cudaq.control``, and CUDA-Q's control-variant generation rejects
kernels that call other kernels ("Unhandled controlled quantum kernel
call"). The inlined circuits are the same ones pinned exhaustively by
``tests/python/test_primitives_arithmetic.py``.

For ``hermitian=True`` (direct encoding) the offsets must be closed under
negation with symmetric band values and a non-negative diagonal (the
encoding's sign convention cannot represent negative diagonals — see
``_sparse_oracle``); ``+o/-o`` pairs are laid out on bit-0-adjacent slots
so the reverse-slot involution is synthesizable. With ``hermitian=False``
the bands may describe any real matrix ``A``; the returned bundle has
``slot_flip=None`` and is meant for
``SparseOracleEncoding.from_general_oracles`` (Hermitian dilation).

These oracles implement the banded construction of Camps, Lin, Van
Beeumen and Yang, "Explicit quantum circuits for block encodings of
certain sparse matrices" (arXiv:2203.10236, Sec. 4.2): the location
oracle is their shift/adder ``O_c`` and the value oracle their Ry-rotation
``O_A``, feeding the Hermitian ``T-dagger S T`` assembler of
``SparseOracleEncoding`` (that paper's Thm. 8.1). Bands are here encoded
non-periodically (edge elements clipped to zero) rather than as the
paper's circulant.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import cudaq

from ._sparse_oracle import OracleKernels, _retain

__all__ = ["banded_oracles", "quantized_angle"]


def quantized_angle(value: float, h: float, value_bits: int) -> int:
    """Fixed-point angle for ``|value|``: ``round(theta / (pi/2) * 2^vb)``.

    ``theta = arcsin(sqrt(|value| / h))``; the result is clamped to
    ``2^value_bits - 1`` (the ``|value| = h`` edge loses one ulp). Shared
    by the oracle factory and the dense-reference tests, so the tests pin
    the exact encoded (quantized) operator at simulator precision.
    """
    theta = math.asin(math.sqrt(min(1.0, abs(value) / h)))
    return min((1 << value_bits) - 1,
               int(round(theta / (0.5 * math.pi) * (1 << value_bits))))


def banded_oracles(offsets: Sequence[int],
                   band_values: Sequence[float],
                   num_system: int,
                   value_bits: int,
                   *,
                   h: float | None = None,
                   hermitian: bool = True) -> OracleKernels:
    """Oracles for the banded matrix ``H[j + o, j] = band_values[o]``.

    Parameters
    ----------
    offsets
        Distinct band offsets ``o`` (``|o| < 2^num_system``); band ``o``
        holds the elements ``H[j + o, j]`` for all in-range ``j`` (no
        periodic wrap-around).
    band_values
        One real value per offset.
    num_system
        System register width ``n`` (the matrix is ``2^n x 2^n``).
    value_bits
        Fixed-point bits for the angle register.
    h
        Value normalization (defaults to ``max |band_values|``); must
        satisfy ``h >= max |band_values|``.
    hermitian
        Validate and lay out for direct encoding (see the module
        docstring); ``False`` produces a dilation-only bundle
        (``slot_flip=None``).
    """
    offsets = [int(o) for o in offsets]
    values = [float(v) for v in band_values]
    if len(offsets) == 0:
        raise ValueError("offsets must be non-empty")
    if len(offsets) != len(values):
        raise ValueError("offsets and band_values must have the same length")
    if len(set(offsets)) != len(offsets):
        raise ValueError("offsets must be distinct")
    if int(num_system) != num_system or num_system < 1:
        raise ValueError("num_system must be a positive integer")
    if int(value_bits) != value_bits or value_bits < 1:
        raise ValueError("value_bits must be a positive integer")
    dimension = 1 << int(num_system)
    if any(abs(o) >= dimension for o in offsets):
        raise ValueError(
            f"band offsets must satisfy |offset| < 2^num_system = {dimension}")
    if any(not math.isfinite(v) for v in values):
        raise ValueError("band values must be finite")
    max_value = max(abs(v) for v in values)
    if max_value == 0.0:
        raise ValueError("band values are all zero: nothing to encode")
    if h is None:
        h = max_value
    h = float(h)
    if not (h > 0.0) or not math.isfinite(h):
        raise ValueError("h must be positive and finite (h > 0)")
    if h < max_value:
        raise ValueError(f"h={h} is smaller than max |band value| "
                         f"{max_value}: the encoding requires h >= max|H_ij|")

    by_offset = dict(zip(offsets, values))
    if hermitian:
        for o, v in by_offset.items():
            if -o not in by_offset:
                raise ValueError(
                    f"hermitian banded oracles need offsets closed under "
                    f"negation: offset {o} has no partner {-o} (pass "
                    f"hermitian=False for a dilation-only bundle)")
            if abs(v - by_offset[-o]) > 1e-12:
                raise ValueError(
                    f"hermitian banded oracles need symmetric values: "
                    f"band {o} has {v} but band {-o} has {by_offset[-o]}")
        if by_offset.get(0, 0.0) < 0.0:
            raise ValueError(
                "negative diagonal band values are not representable by the "
                "symmetric sparse-oracle construction (the diagonal phase "
                "contribution is always +1); shift the diagonal or encode "
                "the Hermitian dilation via from_general_oracles")
        # Slot layout: (+o, -o) pairs on bit-0-adjacent slots, diagonal
        # (if present) self-paired after them.
        ordered = []
        for o in sorted(o for o in by_offset if o > 0):
            ordered.extend([o, -o])
        if 0 in by_offset:
            ordered.append(0)
    else:
        ordered = list(offsets)

    d = len(ordered)
    m = max(1, (d - 1).bit_length())
    d_padded = 1 << m
    if hermitian:
        slot_flip = []
        for s in range(d_padded):
            if s < d and ordered[s] != 0:
                slot_flip.append(s ^ 1)
            else:
                slot_flip.append(s)
    else:
        slot_flip = None

    # Per padded slot classical tables (pads: offset 0, value 0, skipped).
    slot_offsets = [ordered[s] if s < d else 0 for s in range(d_padded)]
    slot_values = [
        by_offset[ordered[s]] if s < d else 0.0 for s in range(d_padded)
    ]
    angle_bits: list[int] = []
    signs: list[int] = []
    uppers: list[int] = []
    cmp_constants: list[int] = []
    cmp_inverts: list[int] = []
    skips: list[int] = []
    for s in range(d_padded):
        o = slot_offsets[s]
        v = slot_values[s]
        a = quantized_angle(v, h, value_bits) if v != 0.0 else 0
        angle_bits.extend((a >> k) & 1 for k in range(value_bits))
        signs.append(1 if v < 0.0 else 0)
        uppers.append(1 if (hermitian and o < 0) else 0)
        if o > 0:
            cmp_constants.append(o)
            cmp_inverts.append(0)
        elif o < 0:
            cmp_constants.append(dimension - abs(o))
            cmp_inverts.append(1)
        else:
            cmp_constants.append(0)  # constant-true comparator
            cmp_inverts.append(0)
        skips.append(1 if (a == 0 and v >= 0.0) else 0)

    n = int(num_system)
    vb = int(value_bits)
    big_d = d_padded

    @cudaq.kernel
    def banded_o_loc(slot: cudaq.qview, system: cudaq.qview,
                     work: cudaq.qview):
        """Slot-multiplexed Draper constant adder: |s>|j> -> |s>|j + o_s>."""
        # QFT (inlined; see the module docstring).
        for j in range(n):
            t = n - 1 - j
            h(system[t])
            for c in range(t):
                r1.ctrl(3.141592653589793 / (1 << (t - c)), system[c],
                        system[t])
        for s in range(big_d):
            offset = slot_offsets[s]
            if offset != 0:
                for b in range(m):
                    if ((s >> b) & 1) == 0:
                        x(slot[b])
                for t in range(n):
                    r1.ctrl(6.283185307179586 * offset / (1 << (t + 1)), slot,
                            system[t])
                for b in range(m):
                    if ((s >> b) & 1) == 0:
                        x(slot[b])
        # Inverse QFT (inlined).
        for t in range(n):
            for k in range(t):
                c = t - 1 - k
                r1.ctrl(-3.141592653589793 / (1 << (t - c)), system[c],
                        system[t])
            h(system[t])

    @cudaq.kernel
    def banded_o_loc_adj(slot: cudaq.qview, system: cudaq.qview,
                         work: cudaq.qview):
        """Hand-written inverse of ``banded_o_loc`` (negated phases)."""
        for j in range(n):
            t = n - 1 - j
            h(system[t])
            for c in range(t):
                r1.ctrl(3.141592653589793 / (1 << (t - c)), system[c],
                        system[t])
        for s in range(big_d):
            offset = slot_offsets[s]
            if offset != 0:
                for b in range(m):
                    if ((s >> b) & 1) == 0:
                        x(slot[b])
                for t in range(n):
                    r1.ctrl(-6.283185307179586 * offset / (1 << (t + 1)), slot,
                            system[t])
                for b in range(m):
                    if ((s >> b) & 1) == 0:
                        x(slot[b])
        for t in range(n):
            for k in range(t):
                c = t - 1 - k
                r1.ctrl(-3.141592653589793 / (1 << (t - c)), system[c],
                        system[t])
            h(system[t])

    @cudaq.kernel
    def banded_o_val(slot: cudaq.qview, system: cudaq.qview,
                     value_and_sign: cudaq.qview, work: cudaq.qview):
        """Comparator-gated angle/sign/upper loads per band.

        Per slot: a slot match into ``work[0]``, the boundary comparator
        (inlined Draper form of ``cmp_ge_constant_qft``) into ``work[1]``,
        XOR loads controlled on the (match, valid) pair, then explicit
        uncomputation. Each per-slot block is its own inverse and blocks
        for distinct slots commute, so this kernel is an involution and
        doubles as ``o_val_adj``.
        """
        for s in range(big_d):
            if skips[s] == 0:
                for b in range(m):
                    if ((s >> b) & 1) == 0:
                        x(slot[b])
                x.ctrl(slot, work[0])
                for b in range(m):
                    if ((s >> b) & 1) == 0:
                        x(slot[b])
                # Comparator compute (inlined cmp_ge_constant_qft on
                # [system, work[1]]).
                constant = cmp_constants[s]
                if constant > 0:
                    h(work[1])
                    for c in range(n):
                        r1.ctrl(3.141592653589793 / (1 << (n - c)), system[c],
                                work[1])
                    for j in range(n):
                        t = n - 1 - j
                        h(system[t])
                        for c in range(t):
                            r1.ctrl(3.141592653589793 / (1 << (t - c)),
                                    system[c], system[t])
                    for t in range(n):
                        r1(-6.283185307179586 * constant / (1 << (t + 1)),
                           system[t])
                    r1(-6.283185307179586 * constant / (1 << (n + 1)), work[1])
                    for t in range(n):
                        for k in range(t):
                            c = t - 1 - k
                            r1.ctrl(-3.141592653589793 / (1 << (t - c)),
                                    system[c], system[t])
                        h(system[t])
                    for k in range(n):
                        c = n - 1 - k
                        r1.ctrl(-3.141592653589793 / (1 << (n - c)), system[c],
                                work[1])
                    h(work[1])
                if cmp_inverts[s] == 0:
                    x(work[1])
                # XOR loads on the (match, valid) pair.
                for k in range(vb):
                    if angle_bits[s * vb + k] == 1:
                        x.ctrl(work[0], work[1], value_and_sign[k])
                if signs[s] == 1:
                    x.ctrl(work[0], work[1], value_and_sign[vb])
                if uppers[s] == 1:
                    x.ctrl(work[0], work[1], value_and_sign[vb + 1])
                # Comparator uncompute (inlined cmp_ge_constant_qft_adj).
                if cmp_inverts[s] == 0:
                    x(work[1])
                if constant > 0:
                    h(work[1])
                    for c in range(n):
                        r1.ctrl(3.141592653589793 / (1 << (n - c)), system[c],
                                work[1])
                    for j in range(n):
                        t = n - 1 - j
                        h(system[t])
                        for c in range(t):
                            r1.ctrl(3.141592653589793 / (1 << (t - c)),
                                    system[c], system[t])
                    r1(6.283185307179586 * constant / (1 << (n + 1)), work[1])
                    for k in range(n):
                        t = n - 1 - k
                        r1(6.283185307179586 * constant / (1 << (t + 1)),
                           system[t])
                    for t in range(n):
                        for k in range(t):
                            c = t - 1 - k
                            r1.ctrl(-3.141592653589793 / (1 << (t - c)),
                                    system[c], system[t])
                        h(system[t])
                    for k in range(n):
                        c = n - 1 - k
                        r1.ctrl(-3.141592653589793 / (1 << (n - c)), system[c],
                                work[1])
                    h(work[1])
                # Slot-match uncompute.
                for b in range(m):
                    if ((s >> b) & 1) == 0:
                        x(slot[b])
                x.ctrl(slot, work[0])
                for b in range(m):
                    if ((s >> b) & 1) == 0:
                        x(slot[b])

    _retain(banded_o_loc, banded_o_loc_adj, banded_o_val)
    return OracleKernels(o_loc=banded_o_loc,
                         o_loc_adj=banded_o_loc_adj,
                         o_val=banded_o_val,
                         o_val_adj=banded_o_val,
                         d=d,
                         h=h,
                         value_bits=vb,
                         num_work=2,
                         slot_flip=slot_flip)
