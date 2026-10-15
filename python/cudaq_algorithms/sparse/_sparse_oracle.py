# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse-access block encoding from user-supplied structure oracles.

``SparseOracleEncoding`` block-encodes a real symmetric ``d``-sparse
matrix ``H`` from two oracles (plus hand-written inverses): a *location*
oracle giving the column structure and a *value* oracle giving fixed-point
matrix-element angles. The encoded block is ``H / alpha`` with the honest
padded normalization ``alpha = d_padded * h`` (``d_padded = 2^ceil(log2
d)``, ``h >= max |H_ij|``).

This construction follows Camps, Lin, Van Beeumen and Yang, "Explicit
quantum circuits for block encodings of certain sparse matrices"
(arXiv:2203.10236). ``T`` is their diffusion-``O_A``-``O_c`` sandwich
(Sec. 4, ``D_s = H^m`` over the ``slot`` register, ``O_A`` the value
rotation, ``O_c`` the location oracle), ``S`` is the ``O-dagger . SWAP .
O`` symmetrization that makes the encoding Hermitian (their Thm. 5.1 /
8.1), and ``alpha = d_padded * h`` is their sparsity normalization ``s``.
We diverge from the paper only in the real-only sign convention (they
carry the sign in a complex ``sqrt(A_ij)``; here a ``sign``/``upper`` bit
pair drives ``+i``/``-i`` phases — negative *diagonals* are
unrepresentable either way, as ``|sqrt(A_ii)|^2 = +1``) and in the
hand-written oracle inverses (no ``cudaq.adjoint``; cuda-quantum#4897/
#4898). The underlying sparse-access oracle input model (in-place
location + value oracles for a d-sparse matrix) is the standard one of
Berry, Childs and Kothari (arXiv:1501.01715); the block-encoding / QSVT
context is Low and Chuang (arXiv:1610.06546) and Gilyen, Su, Low and
Wiebe (arXiv:1806.01838). ``banded_oracles`` (see ``_banded``) supplies
the paper's Sec. 4.2 banded oracles as explicit flat circuits.

Construction (T . S . T-dagger)
-------------------------------

``U_A = T-dagger S T`` over the registers ``system(n)`` and::

    ancilla = [slot(m) | flag_a | flag_b | dual(n) | value(value_bits) |
               sign | upper | work]


- ``T``: diffusion (Hadamards) over the slot register; CNOT-copy of the
  system index into the dual register; ``o_loc`` in place on the dual;
  ``o_val`` loading the angle/sign/upper scratch bits; the linear angle
  cascade ``ry(pi * 2^(k - value_bits))`` per value bit onto ``flag_b``
  plus the sign phase (below) and a flag flip; explicit un-load
  (``o_val_adj``).
- ``S``: SWAP of system and dual, SWAP of the flag pair, and the slot
  *flip* involution pairing each slot with its reverse-direction slot.
- ``T-dagger``: the explicit hand-written reversal of ``T`` (ending in
  ``o_loc_adj`` and the closing diffusion). No ``cudaq.adjoint``
  (cuda-quantum#4897/#4898).

``S`` is Hermitian and ``T`` unitary, so ``U_A`` is exactly self-adjoint
and involutory — the property Walk's Chebyshev powers and QSVT's
forward/adjoint step reuse of ``apply_kernel`` rely on. It also makes the
controlled variants cheap: ``controlled-U_A = T-dagger (controlled-S) T``
(the uncontrolled ``T`` pair cancels at control ``|0>``), so user oracles
never need controlled versions.

Sign convention
---------------

A plain sign-controlled Z cannot carry signs through this symmetric
construction (the two ``T`` factors contribute the sign twice and it
squares away). Instead ``o_val`` emits a ``sign`` bit and an ``upper``
bit (row < column), and the encoding applies ``+i`` / ``-i`` phases
(controlled S / S-dagger) keyed on them; the bra- and ket-side phases then
multiply to ``-1`` exactly for negative off-diagonal elements. Negative
*diagonal* elements are not representable (the diagonal phase contribution
is ``|chi|^2`` = +1 for any convention); encode such operators via
``from_general_oracles`` (Hermitian dilation) or shift the diagonal.

Oracle contract (``OracleKernels``)
-----------------------------------

All registers little-endian (``docs/conventions.md``). For slots ``s <
d``, ``x -> c(s, x)`` must be a permutation; padded slots ``s >= d`` must
act as the identity with zero value. Signatures::

    o_loc(slot: qview, system: qview, work: qview)          # in place
    o_loc_adj(slot, system, work)                           # its inverse
    o_val(slot: qview, system: qview, value_and_sign: qview, work: qview)
    o_val_adj(...)                                          # its inverse

``o_val`` XOR-loads, for slot ``s`` and register content ``x`` (the *row*
index, i.e. after ``o_loc``): the ``value_bits``-bit fixed point of
``theta = arcsin(sqrt(|H_{x, cinv(s, x)}| / h))`` (as ``round(theta /
(pi/2) * 2^value_bits)``, clamped to ``2^value_bits - 1``) into
``value_and_sign[0:value_bits]``, the sign bit into
``value_and_sign[value_bits]``, and the upper bit (row < column) into
``value_and_sign[value_bits + 1]``. ``work`` provides ``num_work`` scratch
qubits that must be returned to ``|0>``.

``slot_flip`` is the reverse-slot involution: ``c(slot_flip[s], c(s, x)) =
x`` for all ``x``. Non-trivial pairs must be bit-0 adjacent (``2k <->
2k+1``) so the encoding can synthesize the flip circuit; padded slots must
be self-paired. Oracles without a valid ``slot_flip`` (``None``) cannot be
encoded directly — use ``from_general_oracles``.

Register accounting
-------------------

``num_ancilla`` counts *all* ancillas, including the deterministic
scratch (value/sign/upper/work). The design goal of reflecting only over
the block ancillas is not implementable on CUDA-Q today: qubits allocated
inside a kernel are never deallocated mid-circuit, so per-step scratch
allocation would grow the register with every walk step. Folding the
scratch into the reflected ancilla register is *exactly* equivalent:
scratch is deterministically ``|0>`` at every reflection point, and
``I - 2|0><0|_(block+scratch)`` restricted to the scratch-``|0>`` subspace
equals ``I - 2|0><0|_block``. ``num_block_ancilla`` and ``num_scratch`` expose the
split for hardware-cost accounting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cudaq

from ..common_kernels import (_validate_power, controlled_reflect_about_zero,
                              reflect_about_zero)

if TYPE_CHECKING:
    from ..block_encoding import Kernel

__all__ = ["OracleKernels", "SparseOracleEncoding"]

_HALF_PI = 1.5707963267948966

# Keep-alive registry for factory-minted kernels. CUDA-Q identifies a
# kernel by ``<function name>..<hex(id(decorator))>`` and retains compiled
# modules under that key without unretaining on deallocation; when a
# same-named factory kernel (every encoding instance mints a ``t_iso``,
# ``apply_u``, ...) is created at a recycled ``id()`` of a dead one, the
# stale module collides with the new kernel and compilation fails with
# arbitrary errors ("too many values"). Pinning every minted kernel for
# the process lifetime keeps the ids from recycling. The cost is a few KB
# per encoding instance.
_LIVE_KERNELS: list = []


def _retain(*kernels) -> None:
    _LIVE_KERNELS.extend(kernels)


@cudaq.kernel
def _noop(register: cudaq.qview):
    """Identity on the register (PREPARE/UNPREPARE of this encoding).

    The diffusion lives inside ``U_A`` itself, so the protocol's
    PREPARE/UNPREPARE hooks are trivial.
    """
    pass


@dataclass
class OracleKernels:
    """The user-supplied oracle bundle (see the module docstring).

    Attributes
    ----------
    o_loc, o_loc_adj, o_val, o_val_adj
        ``@cudaq.kernel`` device kernels with the documented signatures;
        every inverse is explicit (``cudaq.adjoint`` is off-limits).
    d
        Number of structure slots (>= 1); padded to ``2^ceil(log2 d)``.
    h
        Value normalization, ``h >= max |H_ij|`` (> 0).
    value_bits
        Fixed-point bits of the angle register (>= 1).
    num_work
        Work qubits the oracles need (returned to ``|0>``).
    slot_flip
        Reverse-slot involution over the *padded* slot range, or ``None``
        if unavailable (then only ``from_general_oracles`` applies).
    """

    o_loc: Any
    o_loc_adj: Any
    o_val: Any
    o_val_adj: Any
    d: int
    h: float
    value_bits: int
    num_work: int = 0
    slot_flip: list[int] | None = field(default=None)


class SparseOracleEncoding:
    """Block encoding of ``H / (d_padded * h)`` from sparse-access oracles.

    Parameters
    ----------
    oracles
        The ``OracleKernels`` bundle (contract in the module docstring).
    num_system
        System register width ``n`` (the encoded matrix is ``2^n x 2^n``).

    Satisfies the ``BlockEncoding`` protocol except ``select_observable``
    (LCU-specific odd-moment hook), which raises ``NotImplementedError``.
    Every kernel factory works unchanged; measure Chebyshev data through
    ``Walk(...).walk_kernel`` powers rather than ``Walk.moment`` — the
    even-moment reflection observable expands to ``2^num_ancilla`` Pauli
    terms, astronomical at this encoding's ancilla widths (``Walk``
    refuses fast rather than hanging).
    """

    def __init__(self, oracles: OracleKernels, num_system: int) -> None:
        if int(num_system) != num_system or num_system < 1:
            raise ValueError("num_system must be a positive integer")
        if int(oracles.d) != oracles.d or oracles.d < 1:
            raise ValueError("oracles.d must be a positive integer (d >= 1)")
        if not (oracles.h > 0.0) or not math.isfinite(oracles.h):
            raise ValueError("oracles.h must be positive and finite (h > 0)")
        if int(oracles.value_bits) != oracles.value_bits or \
                oracles.value_bits < 1:
            raise ValueError("oracles.value_bits must be a positive integer "
                             "(value_bits >= 1)")
        if int(oracles.num_work) != oracles.num_work or oracles.num_work < 0:
            raise ValueError("oracles.num_work must be a non-negative integer")
        for name in ("o_loc", "o_loc_adj", "o_val", "o_val_adj"):
            if getattr(oracles, name) is None:
                raise ValueError(f"oracles.{name} is required")

        d = int(oracles.d)
        m = max(1, (d - 1).bit_length())
        d_padded = 1 << m
        flip = oracles.slot_flip
        if flip is None:
            raise ValueError(
                "oracles.slot_flip is required for a direct encoding: it is "
                "the reverse-slot involution with c(slot_flip[s], c(s, x)) = "
                "x. For general (non-symmetric or flip-less) oracles use "
                "SparseOracleEncoding.from_general_oracles, whose dilated "
                "slots are self-paired.")
        flip = [int(s) for s in flip]
        if len(flip) != d_padded:
            raise ValueError(
                f"slot_flip must cover the padded slot range: expected "
                f"{d_padded} entries, got {len(flip)}")
        for s, t in enumerate(flip):
            if t < 0 or t >= d_padded or flip[t] != s:
                raise ValueError("slot_flip must be an involution on "
                                 f"[0, {d_padded})")
            if t != s and t != (s ^ 1):
                raise ValueError(
                    "non-trivial slot_flip pairs must be bit-0 adjacent "
                    "(2k <-> 2k+1); relabel the slots")
            if s >= d and t != s:
                raise ValueError("padded slots (s >= d) must be self-paired "
                                 "in slot_flip")

        self._oracles = oracles
        self._num_system = int(num_system)
        self._d = d
        self._m = m
        self._d_padded = d_padded
        self._h = float(oracles.h)
        self._value_bits = int(oracles.value_bits)
        self._num_work = max(1, int(oracles.num_work))  # >= 1: an empty
        # work view must never cross the kernel boundary (the same padding
        # rationale as cuda-quantum#4847's empty-list restriction).
        self._flip_patterns = [
            s >> 1 for s in range(0, d_padded, 2) if flip[s] == s + 1
        ]

        self._num_block_ancilla = m + 2 + self._num_system
        self._num_scratch = self._value_bits + 2 + self._num_work
        self._build_kernels()

    # ------------------------------------------------------------------
    # Kernel construction (all data captured here, factories are data-free)
    # ------------------------------------------------------------------

    def _build_kernels(self) -> None:
        # Unpack everything into scalar/list locals: tuples (and self)
        # cannot be closure-captured into kernels.
        m = self._m
        n_sys = self._num_system
        vb = self._value_bits
        nw = self._num_work
        fa = m
        fb = m + 1
        d0 = m + 2
        v0 = m + 2 + n_sys
        sign_q = v0 + vb
        upper_q = v0 + vb + 1
        w0 = v0 + vb + 2
        n_anc = self.num_ancilla
        patterns = list(self._flip_patterns)
        num_patterns = len(patterns)
        if not patterns:
            patterns = [0]  # padding only; the count above is the guard
            # (empty lists cannot cross the kernel boundary,
            # cuda-quantum#4847).
        o_loc = self._oracles.o_loc
        o_loc_adj = self._oracles.o_loc_adj
        o_val = self._oracles.o_val
        o_val_adj = self._oracles.o_val_adj

        @cudaq.kernel
        def t_iso(ancilla: cudaq.qview, system: cudaq.qview):
            """The T factor: diffusion, copy, o_loc, angle load onto fb."""
            for k in range(m):
                h(ancilla[k])
            for k in range(n_sys):
                cx(system[k], ancilla[d0 + k])
            o_loc(ancilla.front(m), ancilla[d0:d0 + n_sys],
                  ancilla[w0:w0 + nw])
            o_val(ancilla.front(m), ancilla[d0:d0 + n_sys],
                  ancilla[v0:v0 + vb + 2], ancilla[w0:w0 + nw])
            for k in range(vb):
                ry.ctrl(3.141592653589793 * (1 << k) / (1 << vb),
                        ancilla[v0 + k], ancilla[fb])
            r1.ctrl(_HALF_PI, ancilla[sign_q], ancilla[upper_q], ancilla[fb])
            x(ancilla[upper_q])
            r1.ctrl(-_HALF_PI, ancilla[sign_q], ancilla[upper_q], ancilla[fb])
            x(ancilla[upper_q])
            x(ancilla[fb])
            o_val_adj(ancilla.front(m), ancilla[d0:d0 + n_sys],
                      ancilla[v0:v0 + vb + 2], ancilla[w0:w0 + nw])

        @cudaq.kernel
        def t_iso_adj(ancilla: cudaq.qview, system: cudaq.qview):
            """Hand-written exact reversal of ``t_iso``."""
            o_val(ancilla.front(m), ancilla[d0:d0 + n_sys],
                  ancilla[v0:v0 + vb + 2], ancilla[w0:w0 + nw])
            x(ancilla[fb])
            x(ancilla[upper_q])
            r1.ctrl(_HALF_PI, ancilla[sign_q], ancilla[upper_q], ancilla[fb])
            x(ancilla[upper_q])
            r1.ctrl(-_HALF_PI, ancilla[sign_q], ancilla[upper_q], ancilla[fb])
            for j in range(vb):
                k = vb - 1 - j
                ry.ctrl(-3.141592653589793 * (1 << k) / (1 << vb),
                        ancilla[v0 + k], ancilla[fb])
            o_val_adj(ancilla.front(m), ancilla[d0:d0 + n_sys],
                      ancilla[v0:v0 + vb + 2], ancilla[w0:w0 + nw])
            o_loc_adj(ancilla.front(m), ancilla[d0:d0 + n_sys],
                      ancilla[w0:w0 + nw])
            for k in range(n_sys):
                cx(system[k], ancilla[d0 + k])
            for k in range(m):
                h(ancilla[k])

        @cudaq.kernel
        def s_swap(ancilla: cudaq.qview, system: cudaq.qview):
            """The Hermitian S factor: register swap + flag swap + flip."""
            for k in range(n_sys):
                swap(system[k], ancilla[d0 + k])
            swap(ancilla[fa], ancilla[fb])
            for p in range(num_patterns):
                pattern = patterns[p]
                if m == 1:
                    x(ancilla[0])
                else:
                    for b in range(m - 1):
                        if ((pattern >> b) & 1) == 0:
                            x(ancilla[1 + b])
                    x.ctrl(ancilla[1:m], ancilla[0])
                    for b in range(m - 1):
                        if ((pattern >> b) & 1) == 0:
                            x(ancilla[1 + b])

        @cudaq.kernel
        def controlled_s_swap(control_and_ancilla: cudaq.qview,
                              system: cudaq.qview):
            """S controlled by qubit 0 of ``control_and_ancilla``.

            The pattern match for the slot flip is computed into a (then
            ``|0>``) work qubit so every gate has an individual-qubit control
            set (a CUDA-Q control set cannot mix a qview with a qubit).
            """
            for k in range(n_sys):
                swap.ctrl(control_and_ancilla[0], system[k],
                          control_and_ancilla[1 + d0 + k])
            swap.ctrl(control_and_ancilla[0], control_and_ancilla[1 + fa],
                      control_and_ancilla[1 + fb])
            for p in range(num_patterns):
                pattern = patterns[p]
                if m == 1:
                    x.ctrl(control_and_ancilla[0], control_and_ancilla[1])
                else:
                    for b in range(m - 1):
                        if ((pattern >> b) & 1) == 0:
                            x(control_and_ancilla[2 + b])
                    x.ctrl(control_and_ancilla[2:1 + m],
                           control_and_ancilla[1 + w0])
                    x.ctrl(control_and_ancilla[0], control_and_ancilla[1 + w0],
                           control_and_ancilla[1])
                    x.ctrl(control_and_ancilla[2:1 + m],
                           control_and_ancilla[1 + w0])
                    for b in range(m - 1):
                        if ((pattern >> b) & 1) == 0:
                            x(control_and_ancilla[2 + b])

        @cudaq.kernel
        def apply_u(ancilla: cudaq.qview, system: cudaq.qview):
            """U_A = T-dagger S T (exactly self-adjoint and involutory)."""
            t_iso(ancilla, system)
            s_swap(ancilla, system)
            t_iso_adj(ancilla, system)

        @cudaq.kernel
        def controlled_apply_u(control_and_ancilla: cudaq.qview,
                               system: cudaq.qview):
            """Controlled U_A = T-dagger (controlled-S) T.

            The uncontrolled T pair cancels at control ``|0>``, so only the S
            factor carries the external control.
            """
            t_iso(control_and_ancilla.back(n_anc), system)
            controlled_s_swap(control_and_ancilla, system)
            t_iso_adj(control_and_ancilla.back(n_anc), system)

        @cudaq.kernel
        def walk_step(ancilla: cudaq.qview, system: cudaq.qview):
            """W = R U_A (block encodes -H/alpha)."""
            apply_u(ancilla, system)
            reflect_about_zero(ancilla)

        @cudaq.kernel
        def adjoint_walk_step(ancilla: cudaq.qview, system: cudaq.qview):
            """W-dagger = U_A R (U_A is self-adjoint)."""
            reflect_about_zero(ancilla)
            apply_u(ancilla, system)

        @cudaq.kernel
        def controlled_walk_step(control_and_ancilla: cudaq.qview,
                                 system: cudaq.qview):
            controlled_apply_u(control_and_ancilla, system)
            controlled_reflect_about_zero(control_and_ancilla)

        @cudaq.kernel
        def controlled_adjoint_walk_step(control_and_ancilla: cudaq.qview,
                                         system: cudaq.qview):
            controlled_reflect_about_zero(control_and_ancilla)
            controlled_apply_u(control_and_ancilla, system)

        _retain(t_iso, t_iso_adj, s_swap, controlled_s_swap, apply_u,
                controlled_apply_u, walk_step, adjoint_walk_step,
                controlled_walk_step, controlled_adjoint_walk_step)
        self._apply = apply_u
        self._controlled_apply = controlled_apply_u
        self._walk_step = walk_step
        self._adjoint_walk_step = adjoint_walk_step
        self._controlled_walk_step = controlled_walk_step
        self._controlled_adjoint_walk_step = controlled_adjoint_walk_step

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def num_system(self) -> int:
        return self._num_system

    @property
    def num_ancilla(self) -> int:
        """All ancillas: block ancillas plus deterministic scratch.

        See the module docstring for why the scratch is folded in (CUDA-Q
        cannot deallocate mid-circuit) and why reflecting over it is
        exactly equivalent to a block-only reflection.
        """
        return self._num_block_ancilla + self._num_scratch

    @property
    def num_block_ancilla(self) -> int:
        """Slot + flag pair + dual register (the entangled ancillas)."""
        return self._num_block_ancilla

    @property
    def num_scratch(self) -> int:
        """Value/sign/upper/work qubits, deterministically ``|0>`` outside
        the oracle-load window inside U_A."""
        return self._num_scratch

    @property
    def d(self) -> int:
        """Requested sparsity (number of structure slots)."""
        return self._d

    @property
    def d_padded(self) -> int:
        """Slots after power-of-two padding (what alpha honestly pays)."""
        return self._d_padded

    @property
    def h(self) -> float:
        """Value normalization (h >= max ``|H_ij|``)."""
        return self._h

    @property
    def value_bits(self) -> int:
        return self._value_bits

    @property
    def alpha(self) -> float:
        """Honest padded normalization: alpha = d_padded * h."""
        return self._d_padded * self._h

    @property
    def oracles(self) -> OracleKernels:
        return self._oracles

    def __repr__(self) -> str:
        return (f"SparseOracleEncoding(d={self.d} (padded {self.d_padded}), "
                f"system_qubits={self.num_system}, "
                f"ancilla_qubits={self.num_ancilla} "
                f"(block {self.num_block_ancilla} + "
                f"scratch {self.num_scratch}), "
                f"value_bits={self.value_bits}, "
                f"alpha={self.alpha:.6g})")

    # ------------------------------------------------------------------
    # Convenience factories (mirroring PauliLCU / the DF example)
    # ------------------------------------------------------------------

    def encode_kernel(self, state_prep: Kernel | None = None) -> Kernel:
        """A kernel applying the full block encoding.

        Without ``state_prep``: a ``@cudaq.kernel(state)`` allocating the
        system register from ``state`` and the ancilla register (in
        ``|0...0>``) after it. With ``state_prep`` (a ``(qubits: qview)``
        kernel): a zero-argument kernel that allocates the system register
        in ``|0...0>``, runs ``state_prep`` on it, then applies the encoding.
        """
        apply_u = self._apply
        n_anc = self.num_ancilla
        n_sys = self.num_system

        if state_prep is not None:

            @cudaq.kernel
            def sparse_prep_encoded():
                system = cudaq.qvector(n_sys)
                state_prep(system)
                ancilla = cudaq.qvector(n_anc)
                apply_u(ancilla, system)

            _retain(sparse_prep_encoded)
            return sparse_prep_encoded

        @cudaq.kernel
        def sparse_encoded(state: cudaq.State):
            system = cudaq.qvector(state)
            ancilla = cudaq.qvector(n_anc)
            apply_u(ancilla, system)

        _retain(sparse_encoded)
        return sparse_encoded

    def walk_kernel(self,
                    power: int = 1,
                    state_prep: Kernel | None = None) -> Kernel:
        """A kernel applying ``power`` qubitization walk steps.

        The all-zero-ancilla block of the result is T_power(-H/alpha)
        applied to the input state (this encoding's PREPARE is trivial, so
        no PREPARE/UNPREPARE sandwich is needed). Input modes as in
        ``encode_kernel``.
        """
        step = self._walk_step
        n_anc = self.num_ancilla
        n_sys = self.num_system
        steps = _validate_power(power)

        if state_prep is not None:

            @cudaq.kernel
            def sparse_prep_walked():
                system = cudaq.qvector(n_sys)
                state_prep(system)
                ancilla = cudaq.qvector(n_anc)
                for _ in range(steps):
                    step(ancilla, system)

            _retain(sparse_prep_walked)
            return sparse_prep_walked

        @cudaq.kernel
        def sparse_walked(state: cudaq.State):
            system = cudaq.qvector(state)
            ancilla = cudaq.qvector(n_anc)
            for _ in range(steps):
                step(ancilla, system)

        _retain(sparse_walked)
        return sparse_walked

    # ------------------------------------------------------------------
    # BlockEncoding protocol: data-free kernel factories
    # ------------------------------------------------------------------

    def prepare_kernel(self) -> Kernel:
        """``(ancilla: qview)``: trivial (the diffusion lives inside U_A)."""
        return _noop

    def unprepare_kernel(self) -> Kernel:
        """``(ancilla: qview)``: trivial (see ``prepare_kernel``)."""
        return _noop

    def apply_kernel(self) -> Kernel:
        """``(ancilla, system)``: the full block encoding U_A."""
        return self._apply

    def controlled_apply_kernel(self) -> Kernel:
        """``(control_and_ancilla, system)``: U_A controlled by qubit 0."""
        return self._controlled_apply

    def walk_step_kernel(self) -> Kernel:
        """``(ancilla, system)``: one qubitization walk step W = R U_A."""
        return self._walk_step

    def adjoint_walk_step_kernel(self) -> Kernel:
        """``(ancilla, system)``: one adjoint walk step W-dagger."""
        return self._adjoint_walk_step

    def controlled_walk_step_kernel(self) -> Kernel:
        """``(control_and_ancilla, system)``: controlled walk step."""
        return self._controlled_walk_step

    def controlled_adjoint_walk_step_kernel(self) -> Kernel:
        """``(control_and_ancilla, system)``: controlled adjoint step."""
        return self._controlled_adjoint_walk_step

    # ------------------------------------------------------------------
    # Observable hooks
    # ------------------------------------------------------------------

    def select_observable(self) -> Any:
        """Unavailable for this encoding (LCU-specific hook)."""
        raise NotImplementedError(
            "SparseOracleEncoding does not provide select_observable: its "
            "SELECT structure is oracle-defined, not a computational-frame "
            "Pauli-word LCU; odd Chebyshev moments via the observable trick "
            "are unavailable for this encoding")

    # ------------------------------------------------------------------
    # Hermitian dilation of general (non-symmetric) oracles
    # ------------------------------------------------------------------

    @classmethod
    def from_general_oracles(cls, oracles: OracleKernels,
                             num_system: int) -> "SparseOracleEncoding":
        """Encode the Hermitian dilation ``[[0, A], [A^T, 0]]`` of ``A``.

        ``oracles`` describes a general real ``2^n x 2^n`` matrix ``A``
        (per-slot column permutations ``c(s, .)`` with explicit inverses;
        the oracle kernels must be flat — through CUDA-Q 0.15 the dilation
        wraps them in ``cudaq.control``, which rejects kernels that call
        kernels, so ``qrom_oracles`` bundles cannot feed this path;
        the value convention of the module docstring, with ``o_val``
        writing *only* the angle and sign bits — the dilation supplies the
        ``upper`` bit itself, since row < column reduces to the halves
        bit). ``slot_flip`` is ignored: every dilated slot is its own
        reverse (``c_H(s, .)`` is an involution by construction), so the
        flip is the identity.

        The result acts on ``num_system + 1`` system qubits (the halves
        bit is the most significant) and encodes ``[[0, A], [A^T, 0]] /
        alpha`` with the same ``d``, ``h``, ``value_bits`` and honest
        padded ``alpha`` as a direct encoding of ``A`` would report.
        """
        if int(num_system) != num_system or num_system < 1:
            raise ValueError("num_system must be a positive integer")
        for name in ("o_loc", "o_loc_adj", "o_val", "o_val_adj"):
            if getattr(oracles, name) is None:
                raise ValueError(f"oracles.{name} is required")
        if int(oracles.d) != oracles.d or oracles.d < 1:
            raise ValueError("oracles.d must be a positive integer (d >= 1)")

        o_loc_a = oracles.o_loc
        o_loc_a_adj = oracles.o_loc_adj
        o_val_a = oracles.o_val
        o_val_a_adj = oracles.o_val_adj
        vb = int(oracles.value_bits)

        @cudaq.kernel
        def o_loc_h(slot: cudaq.qview, system: cudaq.qview, work: cudaq.qview):
            # c_H(s, (b, x)) = (1 - b, b == 1 ? c(s, x) : cinv(s, x)).
            n1 = system.size()
            cudaq.control(o_loc_a, system[n1 - 1], slot, system.front(n1 - 1),
                          work)
            x(system[n1 - 1])
            cudaq.control(o_loc_a_adj, system[n1 - 1], slot,
                          system.front(n1 - 1), work)

        @cudaq.kernel
        def o_loc_h_adj(slot: cudaq.qview, system: cudaq.qview,
                        work: cudaq.qview):
            n1 = system.size()
            cudaq.control(o_loc_a, system[n1 - 1], slot, system.front(n1 - 1),
                          work)
            x(system[n1 - 1])
            cudaq.control(o_loc_a_adj, system[n1 - 1], slot,
                          system.front(n1 - 1), work)

        @cudaq.kernel
        def o_val_h(slot: cudaq.qview, system: cudaq.qview,
                    value_and_sign: cudaq.qview, work: cudaq.qview):
            n1 = system.size()
            low = system.front(n1 - 1)
            # upper = (row < column) = (halves bit of the row is 0).
            x(system[n1 - 1])
            cx(system[n1 - 1], value_and_sign[vb + 1])
            x(system[n1 - 1])
            # Bottom half (rows (0, x), columns (1, .)): A's row-side value
            # oracle applies directly.
            x(system[n1 - 1])
            cudaq.control(o_val_a, system[n1 - 1], slot, low, value_and_sign,
                          work)
            x(system[n1 - 1])
            # Top half (rows (1, x), columns (0, .)): the element is
            # A[c(s, x), x] — conjugate A's value oracle by its location
            # oracle.
            cudaq.control(o_loc_a, system[n1 - 1], slot, low, work)
            cudaq.control(o_val_a, system[n1 - 1], slot, low, value_and_sign,
                          work)
            cudaq.control(o_loc_a_adj, system[n1 - 1], slot, low, work)

        @cudaq.kernel
        def o_val_h_adj(slot: cudaq.qview, system: cudaq.qview,
                        value_and_sign: cudaq.qview, work: cudaq.qview):
            n1 = system.size()
            low = system.front(n1 - 1)
            cudaq.control(o_loc_a, system[n1 - 1], slot, low, work)
            cudaq.control(o_val_a_adj, system[n1 - 1], slot, low,
                          value_and_sign, work)
            cudaq.control(o_loc_a_adj, system[n1 - 1], slot, low, work)
            x(system[n1 - 1])
            cudaq.control(o_val_a_adj, system[n1 - 1], slot, low,
                          value_and_sign, work)
            x(system[n1 - 1])
            x(system[n1 - 1])
            cx(system[n1 - 1], value_and_sign[vb + 1])
            x(system[n1 - 1])

        _retain(o_loc_h, o_loc_h_adj, o_val_h, o_val_h_adj)
        d = int(oracles.d)
        d_padded = 1 << max(1, (d - 1).bit_length())
        dilated = OracleKernels(o_loc=o_loc_h,
                                o_loc_adj=o_loc_h_adj,
                                o_val=o_val_h,
                                o_val_adj=o_val_h_adj,
                                d=d,
                                h=oracles.h,
                                value_bits=oracles.value_bits,
                                num_work=oracles.num_work,
                                slot_flip=list(range(d_padded)))
        return cls(dilated, num_system=int(num_system) + 1)
