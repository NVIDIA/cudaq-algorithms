# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unary iteration: apply a per-address body once for each address value.

Implements unary iteration (Babbush et al., arXiv:1805.03662, Fig. 7)
without measurement-based uncomputation of the temporary logical ANDs —
this corresponds to the strictly unitary select(V) walk of Childs et
al., arXiv:1711.10980, Appendix G.4 (Lemma G.7): a binary-tree walk with
one clean ladder ancilla per address bit, whose sibling and deeper
transitions are fused into in-place rewrites of the active line. The
caller's body gates for address ``k`` fire controlled on the leaf line,
exactly when the address register equals ``k``, so the minted kernel
implements ``sum_k |k><k| (x) U_k``.

Toffoli cost: exactly ``3 N / 2 - 5`` uncontrolled for a full tree of
``N = 2^num_address_bits`` addresses (``N >= 8``; 2 at ``N = 4``, 0 at
``N = 2``) and ``3 N / 2 - 1`` controlled; partial trees
(``num_items < N``) cost less, and the emitter reports the exact number
as ``toffoli_count``. Babbush et al. reach ``N - 1`` Toffolis by pricing
the AND uncomputation at zero via measurement-and-fixup (Gidney,
arXiv:1709.06648); the coherent primitives in this module stay strictly
unitary (statevector-testable, inverse-composable, ``cudaq.control``-safe),
and under that constraint the fused walk is the standard construction. The
measured walk in ``_unary_iteration_measured`` trades those properties for
the ``N - 1`` count. The naive compute/uncompute walk costs ``2 N - 4``.

T-count caveat (so the comparison stays honest under either metric):
roughly a third of this walk's Toffolis are the fused ones, whose
target is a *dirty* ladder line and therefore cost the generic 7 T,
while a compute-from-``|0>`` AND costs 4 T (Gidney, arXiv:1709.06648).
In T gates the fused walk is therefore ~7.5 N against the naive
unitary walk's ~8 N — a ~6% advantage, not the 25% the Toffoli count
suggests. The 7 T figure is the ancilla-free, measurement-free Toffoli;
with Jones-style measured Toffolis the fused walk is about 6 N T, still
above the 4 N T of the measured walk of Babbush et al.

The callback pattern (READ THIS — it is the template for QROM and for
factory-composed SELECTs)
------------------------------------------------------------------------

CUDA-Q kernels are compiled from fixed Python source, so a callback can
never execute *inside* a kernel. The callback therefore runs at **factory
time**: ``body(k)`` is called once per address on the host and must return
the gate list for that address as data. The emitter flattens the whole
tree walk (ladder gadgets + body gates) into parallel opcode/operand
integer lists, and the minted kernel is a single flat interpreter loop
over those captured lists, pruned to the dispatch arms whose opcodes the
tape actually uses (see ``_mint_interpreter``). Flatness is load-bearing:
the QROM builders rewrite walk tapes operand by operand, and a flat tape
keeps every coherent walk ``cudaq.control``-compatible on every CUDA-Q
release this library supports (0.15 rejected control variants of kernels
that call other kernels; 0.16 accepts them), so anything built this way
can sit inside a controlled SELECT.

Body instruction set. Each item is a tuple whose head names the gate;
``target``/``work`` operands index the target/work registers. The three
original gates are implicitly controlled on the address-``k`` leaf line:

- ``("x", t)`` / ``("y", t)`` / ``("z", t)`` — leaf-controlled Pauli on
  ``target[t]``.

The extended vocabulary (for SELECTs whose term unitaries are
multi-controlled bit flips and phases) has two families. *Free* gates
carry **no** leaf control and execute for every address — they are only
sound as conjugation pairs closed inside the same body (pre-ops,
leaf-controlled core, reversed pre-ops), where they cancel exactly on
inactive addresses:

- ``("free_x", t)`` — ``x(target[t])``;
- ``("free_cx", a, b)`` — ``cx(target[a], target[b])``;
- ``("and_tt", a, b, w)`` — Toffoli ``target[a], target[b] -> work[w]``;
- ``("and_wt", v, t, w)`` — Toffoli ``work[v], target[t] -> work[w]``;
- ``("copy_tw", t, w)`` — ``cx(target[t], work[w])``.

Leaf-referencing core gates:

- ``("x_w", w, t)`` — ``x.ctrl(leaf, work[w], target[t])``;
- ``("z_w", w)`` — ``z.ctrl(leaf, work[w])``;
- ``("sign",)`` — ``z(leaf)``: a ``-1`` phase exactly on the
  address-``k`` branch (term-sign carrier).

Any body that uses ``work`` operands makes the minted kernels take a
trailing ``work: qview`` (clean, |0> in / |0> out — each body must
uncompute its own work usage); ``num_work`` on the result records the
required width (0 means no work view in the signature).

Kernel signatures (all registers little-endian, qubit 0 = LSB, per
``docs/conventions.md``):

- ``kernel(address: qview, ladder: qview, target: qview[, work: qview])``
  — walk with the body applied at each active address (the ``work`` view
  appears only when ``num_work > 0``).
- with ``controlled=True``: the same signatures with a leading
  ``control: qview``, a one-qubit view rooting the tree — the whole walk
  acts as the identity when the control is ``|0>``. (A separate one-qubit
  view, not a leading qubit of a combined register, so callers never mix
  a qview with a bare qubit in one control set. Free body gates still
  execute at control ``|0>``, but their conjugation pairing cancels them.)

``kernel_adj`` is the hand-written inverse (``cudaq.adjoint`` is
off-limits, cuda-quantum#4897/#4898): every emitted gate is self-inverse
(X/CX/CCX and controlled Paulis), so the inverse is the same interpreter
over the reversed instruction list. When the body gates at every address
mutually commute and square to identity (e.g. the X-only QROM write), the
walk itself is an involution and ``include_adjoint=False`` skips minting
the redundant inverse.

The walk circuit
----------------

Level ``r`` (1-based, from the root) examines address bit
``address[num_address_bits - r]`` (level 1 = MSB) and owns ladder line
``ladder[r - 1]``; the invariant at leaf ``k`` is that ``ladder[r - 1]``
holds the indicator "top ``r`` address bits equal those of ``k``" (AND
the external control, if any), so the leaf/body line is always
``ladder[num_address_bits - 1]``. The transition from leaf ``k`` to
``k + 1`` (with ``t`` trailing ones in ``k``, flip level ``d = n - t``):

- ``t = 0``: one CNOT from the parent line onto the leaf line — exactly
  one of the two siblings is active given the parent, so their XOR is
  the parent itself.
- ``t >= 1``: unclamp levels ``n .. d + 2`` with Toffolis against the
  still-old parents, XOR the fused difference
  ``guard AND (bit_d XOR bit_{d+1})`` into level ``d + 1`` with a single
  Toffoli on a CNOT-conjugated address wire (the old and new level-
  ``(d+1)`` indicators are disjoint given the guard, so their XOR
  factors), CNOT-flip level ``d``, and reclamp levels ``d + 2 .. n``
  against the new parents: ``2 t - 1`` Toffolis. When the flip level is
  the unguarded root, the fused difference is linear (free CNOTs) and
  the first reclamp pair collapses into one Toffoli on two conjugated
  address wires: ``max(2 t - 3, 0)`` Toffolis.
"""

from __future__ import annotations

from ._unary_iteration_ir import (
    _BASE_OPS,
    _BODY_OPCODES,
    _CONTROL_OPS,
    _EXTENDED_BODY_GATES,
    _MEASURED_CONTROL_OPS,
    _MEASURED_OPS,
    _OP_AND_TT,
    _OP_AND_WT,
    _OP_BODY_X,
    _OP_BODY_X_W,
    _OP_BODY_Y,
    _OP_BODY_Z,
    _OP_BODY_Z_W,
    _OP_CCX,
    _OP_CCX_ADDR_ADDR,
    _OP_CCX_CTRL,
    _OP_COPY_TW,
    _OP_CX_ADDR_ADDR,
    _OP_CX_ADDR_LADDER,
    _OP_CX_CTRL_LADDER,
    _OP_CX_LADDER_LADDER,
    _OP_CX_LADDER_TARGET,
    _OP_DESCRIPTIONS,
    _OP_FREE_CX,
    _OP_FREE_X,
    _OP_MEAS_UNCOMPUTE,
    _OP_MEAS_UNCOMPUTE_CTRL,
    _OP_X_ADDR,
    _OP_X_LADDER,
    _OP_Z_LADDER,
    _TOFFOLI_OPCODES,
    _WORK_OPERANDS,
    _WORK_OPS,
    _emit_walk,
    _trailing_ones,
    _walk_toffoli_count,
    validate_unary_inputs,
    materialize_bodies,
    infer_num_work,
)

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import cudaq

# Bind the measured-arm intrinsics here: CUDA-Q otherwise searches caller
# frames for these names, even when a coherent tape never uses those arms.
from cudaq.qis import h, mz, x, y, z, cx

__all__ = ["UnaryIterationKernels", "unary_iteration_kernels"]

# Keep-alive registry for factory-minted kernels. CUDA-Q identifies a
# kernel by ``<function name>..<hex(id(decorator))>`` and retains compiled
# modules under that key without unretaining on deallocation; when a
# same-named factory kernel is created at a recycled ``id()`` of a dead
# one, the stale module collides with the new kernel and compilation fails
# with arbitrary errors. Pinning every minted kernel for the process
# lifetime keeps the ids from recycling. The cost is real: each pinned
# kernel keeps roughly 0.1 MB of Python objects plus about 130 bytes per
# tape instruction, and because CUDA-Q's decorator keeps its defining
# frame, it also pins the caller's stack frames and their locals for the
# process lifetime (build lookups in a short-lived helper, or expect that
# retention). The kernel names below are unique to this subpackage so they
# can never collide with other subpackages' registries.
_LIVE_KERNELS: list = []


def _retain(*kernels) -> None:
    _LIVE_KERNELS.extend(kernels)


# Interpreter opcodes (operands a, b, c index the signature's registers).
@dataclass(frozen=True)
class UnaryIterationKernels:
    """The minted walk (see the module docstring for signatures).

    Attributes
    ----------
    kernel, kernel_adj
        The walk and its hand-written inverse (``kernel_adj`` is ``None``
        when minted with ``include_adjoint=False``).
    num_address, num_ladder, num_items
        Register widths (``num_ladder == num_address`` clean ancillas)
        and the number of iterated addresses.
    controlled
        Whether ``kernel`` takes the leading one-qubit control view.
    num_work
        Width of the trailing clean ``work`` view (0: no work view in the
        signature).
    toffoli_count
        Toffolis in one application (cost accounting).
    """

    kernel: Any
    kernel_adj: Any
    num_address: int
    num_ladder: int
    num_items: int
    controlled: bool
    toffoli_count: int
    num_work: int = 0
    ops: Any = ()

    def describe(self) -> str:
        """Decode the instruction tape into a human-readable gate listing.

        One line per gate, in execution order — exactly what the
        interpreter kernel replays. Lines tagged ``# Toffoli`` are the
        gates ``toffoli_count`` counts. This is the intended way to
        *read* a minted walk; the kernel source itself is a generic
        interpreter and shows nothing about any particular circuit.
        """
        control = ", controlled" if self.controlled else ""
        header = (f"unary-iteration walk: {self.num_items} addresses over "
                  f"{self.num_address} bits{control}; "
                  f"{self.toffoli_count} Toffolis")
        lines = [
            _OP_DESCRIPTIONS[op].format(a=a, b=b, c=c)
            for op, a, b, c in self.ops
        ]
        return "\n".join([header] + lines)


def unary_iteration_kernels(
        num_address_bits: int,
        num_items: int,
        body: Callable[[int], Sequence[tuple]],
        *,
        controlled: bool = False,
        include_adjoint: bool = True,
        num_work: int | None = None) -> UnaryIterationKernels:
    """Mint the unary-iteration walk for a factory-time body callback.

    Parameters
    ----------
    num_address_bits
        Address register width (>= 1).
    num_items
        Number of iterated addresses (``1 <= num_items <=
        2^num_address_bits``); addresses ``k >= num_items`` are never
        entered and the walk acts as the identity on them.
    body
        The payload table of the SELECT: a factory-time callback mapping
        each address ``k`` to the gate list ``U_k`` that the walk applies
        when the address register holds ``k`` — i.e. the minted kernel
        implements ``sum_k |k><k| (x) U_k``. Each entry is an opcode
        tuple such as ``("x", t)`` / ``("y", t)`` / ``("z", t)``
        (leaf-controlled Pauli on ``target[t]``) or one of the extended
        gates in the module docstring; an empty list means ``U_k = I``.
        Called exactly once per address, in address order, during this
        factory call — the gates are baked into the kernel, so ``body``
        itself never runs at circuit time.
    controlled
        Mint the externally controlled walk (leading one-qubit view).
    include_adjoint
        Mint ``kernel_adj``; pass ``False`` when the walk is known to be
        an involution (commuting self-inverse body, e.g. X-only QROM).
    num_work
        Width of the trailing clean ``work`` view. ``None`` (default)
        infers the width from the body's work usage (0 keeps the original
        work-less signatures); an explicit value must cover that usage
        and forces the work view into the signature even when unused.
    """
    num_address_bits, num_items = validate_unary_inputs(
        num_address_bits, num_items)
    bodies = materialize_bodies(body, num_items)
    ops = _emit_walk(num_address_bits, num_items, bool(controlled),
                     lambda k: bodies[k])
    num_work = infer_num_work(ops, num_work)

    toffolis = sum(1 for op in ops if op[0] in _TOFFOLI_OPCODES)
    kernel = _mint_interpreter(ops, bool(controlled), num_work > 0)
    kernel_adj = None
    if include_adjoint:
        kernel_adj = _mint_interpreter(list(reversed(ops)), bool(controlled),
                                       num_work > 0)
    return UnaryIterationKernels(kernel=kernel,
                                 kernel_adj=kernel_adj,
                                 num_address=num_address_bits,
                                 num_ladder=num_address_bits,
                                 num_items=num_items,
                                 controlled=bool(controlled),
                                 toffoli_count=toffolis,
                                 num_work=num_work,
                                 ops=tuple(ops))


def _mint_interpreter(ops: list,
                      controlled: bool,
                      has_work: bool,
                      *,
                      num_results: int = 0,
                      parity_masks: Sequence[Sequence[int]] = ()):
    """Mint the flat interpreter kernel over a flattened instruction list.

    Every emitted gate is self-inverse, so the inverse walk is this same
    interpreter over the reversed list — how ``kernel_adj`` is minted.

    The tape is validated against the variant's supported opcode set
    (``_BASE_OPS`` plus the control/work bundles the signature carries):
    the dispatch loop would silently skip an unknown opcode — the worst
    failure mode available — so an out-of-set tape is a mint-time error.

    Dispatch pruning: each dispatch arm is guarded by a captured bool
    (``use_*``) that is True only when the opcode occurs in this tape.
    Captured Python constants are folded at kernel-compile time, so a
    False guard statically removes the arm from the minted module. This
    matters twice: an arm whose opcode never occurs is classically dead
    code, and CUDA-Q's CSE can alias a dead multi-control arm's wires
    into illegal IR (cuda-quantum#5280 — an infinite compiler loop
    through 0.15, a hard "linear-typed result is used more than once"
    compile error once fixed); it also shrinks every minted module to
    the arms it actually replays.

    Measured gadgets. The two measured-uncompute opcodes
    (``_OP_MEAS_UNCOMPUTE`` and its control-rooted twin) are dispatched here as
    well, behind the same captured guards; a coherent tape never contains them,
    so their arms are pruned from every coherent module, and the measured walk
    in ``_unary_iteration_measured`` is their only producer. On CUDA-Q 0.16 the
    captured guards are lifted into kernel arguments, so the generic kernel
    source still mentions ``mz`` for coherent walks; the runtime's
    measurement-feedback checks look at entry kernels, and the tests pin that
    ``cudaq.sample``, ``cudaq.control`` and ``cudaq.adjoint`` of coherent tapes
    keep working.
    """
    from . import _unary_iteration_ir as ir

    supported = _BASE_OPS | _MEASURED_OPS | ir._FEEDBACK_OPS
    if controlled:
        supported = supported | _CONTROL_OPS | _MEASURED_CONTROL_OPS
    if has_work:
        supported = supported | _WORK_OPS
    present = {op[0] for op in ops}
    unsupported = sorted(present - supported)
    if unsupported:
        raise ValueError(
            f"instruction tape contains opcodes {unsupported} outside the "
            f"interpreter variant's dispatch set (controlled={controlled}, "
            f"has_work={has_work}); the dispatch loop would silently skip "
            "them")
    # Infer only the minimum wire bounds here; public plans validate against
    # their actual geometry. This also validates result lifetimes and masks.
    def width(roles):
        return max(
            (op[1 + slot] + 1 for op in ops for slot in roles.get(op[0], ())),
            default=0)

    ir.validate_feedback_ops(ops,
                             parity_masks=parity_masks,
                             num_results=num_results,
                             num_address=width(ir._ADDRESS_OPERANDS),
                             num_ladder=width(ir._LADDER_OPERANDS),
                             num_output=width(ir._TARGET_OPERANDS),
                             controlled=controlled,
                             num_work=max(int(has_work),
                                          width(ir._WORK_OPERANDS)))
    mask_offsets = [0]
    mask_slots = []
    for mask in parity_masks:
        mask_slots.extend(mask)
        mask_offsets.append(len(mask_slots))
    # Empty captured vectors are unsupported by the SDK.
    mask_slots = mask_slots or [0]
    result_capacity = max(1, num_results)
    use_measure_reset = ir._OP_MEASURE_RESET in present
    use_measure_reset_output = ir._OP_MEASURE_RESET_OUTPUT in present
    use_parity_z = ir._OP_PARITY_Z_LADDER in present
    use_parity_cz = ir._OP_PARITY_CZ_LADDER in present
    op_measure_reset = ir._OP_MEASURE_RESET
    op_measure_reset_output = ir._OP_MEASURE_RESET_OUTPUT
    op_parity_z = ir._OP_PARITY_Z_LADDER
    op_parity_cz = ir._OP_PARITY_CZ_LADDER
    num_ops = len(ops)
    opcodes = [op[0] for op in ops]
    ops_a = [op[1] for op in ops]
    ops_b = [op[2] for op in ops]
    ops_c = [op[3] for op in ops]
    if num_ops == 0:
        # An empty list must never cross the kernel boundary
        # (cuda-quantum#4847): pad with one never-dispatched entry.
        opcodes, ops_a, ops_b, ops_c = [-1], [0], [0], [0]

    # Kernel-visible opcode names: kernel code cannot reference Python
    # enums or module globals, but it does capture enclosing-scope ints,
    # so the module's _OP_* constants are bound as locals here to keep
    # the dispatch below readable.
    op_x_addr = _OP_X_ADDR
    op_x_ladder = _OP_X_LADDER
    op_cx_addr_ladder = _OP_CX_ADDR_LADDER
    op_cx_ladder_ladder = _OP_CX_LADDER_LADDER
    op_ccx = _OP_CCX
    op_body_x = _OP_BODY_X
    op_body_y = _OP_BODY_Y
    op_body_z = _OP_BODY_Z
    op_cx_ctrl_ladder = _OP_CX_CTRL_LADDER
    op_ccx_ctrl = _OP_CCX_CTRL
    op_free_x = _OP_FREE_X
    op_free_cx = _OP_FREE_CX
    op_and_tt = _OP_AND_TT
    op_and_wt = _OP_AND_WT
    op_copy_tw = _OP_COPY_TW
    op_body_x_w = _OP_BODY_X_W
    op_body_z_w = _OP_BODY_Z_W
    op_z_ladder = _OP_Z_LADDER
    op_cx_addr_addr = _OP_CX_ADDR_ADDR
    op_ccx_addr_addr = _OP_CCX_ADDR_ADDR
    op_cx_ladder_target = _OP_CX_LADDER_TARGET

    # Per-arm pruning flags (see the docstring): captured bools fold at
    # kernel-compile time, so a False flag removes its arm entirely.
    use_x_addr = _OP_X_ADDR in present
    use_x_ladder = _OP_X_LADDER in present
    use_cx_addr_ladder = _OP_CX_ADDR_LADDER in present
    use_cx_ladder_ladder = _OP_CX_LADDER_LADDER in present
    use_ccx = _OP_CCX in present
    use_body_x = _OP_BODY_X in present
    use_body_y = _OP_BODY_Y in present
    use_body_z = _OP_BODY_Z in present
    use_cx_ctrl_ladder = _OP_CX_CTRL_LADDER in present
    use_ccx_ctrl = _OP_CCX_CTRL in present
    use_free_x = _OP_FREE_X in present
    use_free_cx = _OP_FREE_CX in present
    use_and_tt = _OP_AND_TT in present
    use_and_wt = _OP_AND_WT in present
    use_copy_tw = _OP_COPY_TW in present
    use_body_x_w = _OP_BODY_X_W in present
    use_body_z_w = _OP_BODY_Z_W in present
    use_z_ladder = _OP_Z_LADDER in present
    use_cx_addr_addr = _OP_CX_ADDR_ADDR in present
    use_ccx_addr_addr = _OP_CCX_ADDR_ADDR in present
    use_cx_ladder_target = _OP_CX_LADDER_TARGET in present

    op_meas_uncompute = _OP_MEAS_UNCOMPUTE
    op_meas_uncompute_ctrl = _OP_MEAS_UNCOMPUTE_CTRL
    use_meas_uncompute = _OP_MEAS_UNCOMPUTE in present
    use_meas_uncompute_ctrl = _OP_MEAS_UNCOMPUTE_CTRL in present

    if controlled and has_work:

        @cudaq.kernel
        def primitives_unary_walk_work_ctrl(control: cudaq.qview,
                                            address: cudaq.qview,
                                            ladder: cudaq.qview,
                                            target: cudaq.qview,
                                            work: cudaq.qview):
            fired = False
            outcomes = [False for _ in range(result_capacity)]
            for i in range(num_ops):
                op = opcodes[i]
                a = ops_a[i]
                b = ops_b[i]
                c = ops_c[i]
                if use_x_addr:
                    if op == op_x_addr:
                        x(address[a])
                if use_x_ladder:
                    if op == op_x_ladder:
                        x(ladder[a])
                if use_cx_addr_ladder:
                    if op == op_cx_addr_ladder:
                        cx(address[a], ladder[b])
                if use_cx_ladder_ladder:
                    if op == op_cx_ladder_ladder:
                        cx(ladder[a], ladder[b])
                if use_ccx:
                    if op == op_ccx:
                        x.ctrl(ladder[a], address[b], ladder[c])
                if use_body_x:
                    if op == op_body_x:
                        x.ctrl(ladder[a], target[b])
                if use_body_y:
                    if op == op_body_y:
                        y.ctrl(ladder[a], target[b])
                if use_body_z:
                    if op == op_body_z:
                        z.ctrl(ladder[a], target[b])
                if use_cx_ctrl_ladder:
                    if op == op_cx_ctrl_ladder:
                        cx(control[0], ladder[b])
                if use_ccx_ctrl:
                    if op == op_ccx_ctrl:
                        x.ctrl(control[0], address[b], ladder[c])
                if use_free_x:
                    if op == op_free_x:
                        x(target[a])
                if use_free_cx:
                    if op == op_free_cx:
                        cx(target[a], target[b])
                if use_and_tt:
                    if op == op_and_tt:
                        x.ctrl(target[a], target[b], work[c])
                if use_and_wt:
                    if op == op_and_wt:
                        x.ctrl(work[a], target[b], work[c])
                if use_copy_tw:
                    if op == op_copy_tw:
                        cx(target[a], work[b])
                if use_body_x_w:
                    if op == op_body_x_w:
                        x.ctrl(ladder[a], work[b], target[c])
                if use_body_z_w:
                    if op == op_body_z_w:
                        z.ctrl(ladder[a], work[b])
                if use_z_ladder:
                    if op == op_z_ladder:
                        z(ladder[a])
                if use_cx_addr_addr:
                    if op == op_cx_addr_addr:
                        cx(address[a], address[b])
                if use_ccx_addr_addr:
                    if op == op_ccx_addr_addr:
                        x.ctrl(address[a], address[b], ladder[c])
                if use_cx_ladder_target:
                    if op == op_cx_ladder_target:
                        cx(ladder[a], target[b])
                if use_measure_reset:
                    if op == op_measure_reset:
                        h(ladder[a])
                        outcomes[b] = mz(ladder[a])
                        if outcomes[b]:
                            x(ladder[a])
                if use_measure_reset_output:
                    if op == op_measure_reset_output:
                        h(target[a])
                        outcomes[b] = mz(target[a])
                        if outcomes[b]:
                            x(target[a])
                if use_parity_z:
                    if op == op_parity_z:
                        parity = False
                        for j in range(mask_offsets[b], mask_offsets[b + 1]):
                            parity = parity != outcomes[mask_slots[j]]
                        if parity:
                            z(ladder[a])
                if use_parity_cz:
                    if op == op_parity_cz:
                        parity = False
                        for j in range(mask_offsets[c], mask_offsets[c + 1]):
                            parity = parity != outcomes[mask_slots[j]]
                        if parity:
                            z.ctrl(ladder[a], ladder[b])
                if use_meas_uncompute:
                    if op == op_meas_uncompute:
                        h(ladder[a])
                        fired = mz(ladder[a])
                        if fired:
                            x(ladder[a])
                            z.ctrl(ladder[b], address[c])
                if use_meas_uncompute_ctrl:
                    if op == op_meas_uncompute_ctrl:
                        h(ladder[a])
                        fired = mz(ladder[a])
                        if fired:
                            x(ladder[a])
                            z.ctrl(control[0], address[c])

        _retain(primitives_unary_walk_work_ctrl)
        return primitives_unary_walk_work_ctrl

    if has_work:

        @cudaq.kernel
        def primitives_unary_walk_work(address: cudaq.qview,
                                       ladder: cudaq.qview,
                                       target: cudaq.qview, work: cudaq.qview):
            fired = False
            outcomes = [False for _ in range(result_capacity)]
            for i in range(num_ops):
                op = opcodes[i]
                a = ops_a[i]
                b = ops_b[i]
                c = ops_c[i]
                if use_x_addr:
                    if op == op_x_addr:
                        x(address[a])
                if use_x_ladder:
                    if op == op_x_ladder:
                        x(ladder[a])
                if use_cx_addr_ladder:
                    if op == op_cx_addr_ladder:
                        cx(address[a], ladder[b])
                if use_cx_ladder_ladder:
                    if op == op_cx_ladder_ladder:
                        cx(ladder[a], ladder[b])
                if use_ccx:
                    if op == op_ccx:
                        x.ctrl(ladder[a], address[b], ladder[c])
                if use_body_x:
                    if op == op_body_x:
                        x.ctrl(ladder[a], target[b])
                if use_body_y:
                    if op == op_body_y:
                        y.ctrl(ladder[a], target[b])
                if use_body_z:
                    if op == op_body_z:
                        z.ctrl(ladder[a], target[b])
                if use_free_x:
                    if op == op_free_x:
                        x(target[a])
                if use_free_cx:
                    if op == op_free_cx:
                        cx(target[a], target[b])
                if use_and_tt:
                    if op == op_and_tt:
                        x.ctrl(target[a], target[b], work[c])
                if use_and_wt:
                    if op == op_and_wt:
                        x.ctrl(work[a], target[b], work[c])
                if use_copy_tw:
                    if op == op_copy_tw:
                        cx(target[a], work[b])
                if use_body_x_w:
                    if op == op_body_x_w:
                        x.ctrl(ladder[a], work[b], target[c])
                if use_body_z_w:
                    if op == op_body_z_w:
                        z.ctrl(ladder[a], work[b])
                if use_z_ladder:
                    if op == op_z_ladder:
                        z(ladder[a])
                if use_cx_addr_addr:
                    if op == op_cx_addr_addr:
                        cx(address[a], address[b])
                if use_ccx_addr_addr:
                    if op == op_ccx_addr_addr:
                        x.ctrl(address[a], address[b], ladder[c])
                if use_cx_ladder_target:
                    if op == op_cx_ladder_target:
                        cx(ladder[a], target[b])
                if use_measure_reset:
                    if op == op_measure_reset:
                        h(ladder[a])
                        outcomes[b] = mz(ladder[a])
                        if outcomes[b]:
                            x(ladder[a])
                if use_measure_reset_output:
                    if op == op_measure_reset_output:
                        h(target[a])
                        outcomes[b] = mz(target[a])
                        if outcomes[b]:
                            x(target[a])
                if use_parity_z:
                    if op == op_parity_z:
                        parity = False
                        for j in range(mask_offsets[b], mask_offsets[b + 1]):
                            parity = parity != outcomes[mask_slots[j]]
                        if parity:
                            z(ladder[a])
                if use_parity_cz:
                    if op == op_parity_cz:
                        parity = False
                        for j in range(mask_offsets[c], mask_offsets[c + 1]):
                            parity = parity != outcomes[mask_slots[j]]
                        if parity:
                            z.ctrl(ladder[a], ladder[b])
                if use_meas_uncompute:
                    if op == op_meas_uncompute:
                        h(ladder[a])
                        fired = mz(ladder[a])
                        if fired:
                            x(ladder[a])
                            z.ctrl(ladder[b], address[c])

        _retain(primitives_unary_walk_work)
        return primitives_unary_walk_work

    if controlled:

        @cudaq.kernel
        def primitives_unary_walk_ctrl(control: cudaq.qview,
                                       address: cudaq.qview,
                                       ladder: cudaq.qview,
                                       target: cudaq.qview):
            fired = False
            outcomes = [False for _ in range(result_capacity)]
            for i in range(num_ops):
                op = opcodes[i]
                a = ops_a[i]
                b = ops_b[i]
                c = ops_c[i]
                if use_x_addr:
                    if op == op_x_addr:
                        x(address[a])
                if use_x_ladder:
                    if op == op_x_ladder:
                        x(ladder[a])
                if use_cx_addr_ladder:
                    if op == op_cx_addr_ladder:
                        cx(address[a], ladder[b])
                if use_cx_ladder_ladder:
                    if op == op_cx_ladder_ladder:
                        cx(ladder[a], ladder[b])
                if use_ccx:
                    if op == op_ccx:
                        x.ctrl(ladder[a], address[b], ladder[c])
                if use_body_x:
                    if op == op_body_x:
                        x.ctrl(ladder[a], target[b])
                if use_body_y:
                    if op == op_body_y:
                        y.ctrl(ladder[a], target[b])
                if use_body_z:
                    if op == op_body_z:
                        z.ctrl(ladder[a], target[b])
                if use_cx_ctrl_ladder:
                    if op == op_cx_ctrl_ladder:
                        cx(control[0], ladder[b])
                if use_ccx_ctrl:
                    if op == op_ccx_ctrl:
                        x.ctrl(control[0], address[b], ladder[c])
                if use_free_x:
                    if op == op_free_x:
                        x(target[a])
                if use_free_cx:
                    if op == op_free_cx:
                        cx(target[a], target[b])
                if use_z_ladder:
                    if op == op_z_ladder:
                        z(ladder[a])
                if use_cx_addr_addr:
                    if op == op_cx_addr_addr:
                        cx(address[a], address[b])
                if use_ccx_addr_addr:
                    if op == op_ccx_addr_addr:
                        x.ctrl(address[a], address[b], ladder[c])
                if use_cx_ladder_target:
                    if op == op_cx_ladder_target:
                        cx(ladder[a], target[b])
                if use_measure_reset:
                    if op == op_measure_reset:
                        h(ladder[a])
                        outcomes[b] = mz(ladder[a])
                        if outcomes[b]:
                            x(ladder[a])
                if use_measure_reset_output:
                    if op == op_measure_reset_output:
                        h(target[a])
                        outcomes[b] = mz(target[a])
                        if outcomes[b]:
                            x(target[a])
                if use_parity_z:
                    if op == op_parity_z:
                        parity = False
                        for j in range(mask_offsets[b], mask_offsets[b + 1]):
                            parity = parity != outcomes[mask_slots[j]]
                        if parity:
                            z(ladder[a])
                if use_parity_cz:
                    if op == op_parity_cz:
                        parity = False
                        for j in range(mask_offsets[c], mask_offsets[c + 1]):
                            parity = parity != outcomes[mask_slots[j]]
                        if parity:
                            z.ctrl(ladder[a], ladder[b])
                if use_meas_uncompute:
                    if op == op_meas_uncompute:
                        h(ladder[a])
                        fired = mz(ladder[a])
                        if fired:
                            x(ladder[a])
                            z.ctrl(ladder[b], address[c])
                if use_meas_uncompute_ctrl:
                    if op == op_meas_uncompute_ctrl:
                        h(ladder[a])
                        fired = mz(ladder[a])
                        if fired:
                            x(ladder[a])
                            z.ctrl(control[0], address[c])

        _retain(primitives_unary_walk_ctrl)
        return primitives_unary_walk_ctrl

    @cudaq.kernel
    def primitives_unary_walk(address: cudaq.qview, ladder: cudaq.qview,
                              target: cudaq.qview):
        fired = False
        outcomes = [False for _ in range(result_capacity)]
        for i in range(num_ops):
            op = opcodes[i]
            a = ops_a[i]
            b = ops_b[i]
            c = ops_c[i]
            if use_x_addr:
                if op == op_x_addr:
                    x(address[a])
            if use_x_ladder:
                if op == op_x_ladder:
                    x(ladder[a])
            if use_cx_addr_ladder:
                if op == op_cx_addr_ladder:
                    cx(address[a], ladder[b])
            if use_cx_ladder_ladder:
                if op == op_cx_ladder_ladder:
                    cx(ladder[a], ladder[b])
            if use_ccx:
                if op == op_ccx:
                    x.ctrl(ladder[a], address[b], ladder[c])
            if use_body_x:
                if op == op_body_x:
                    x.ctrl(ladder[a], target[b])
            if use_body_y:
                if op == op_body_y:
                    y.ctrl(ladder[a], target[b])
            if use_body_z:
                if op == op_body_z:
                    z.ctrl(ladder[a], target[b])
            if use_free_x:
                if op == op_free_x:
                    x(target[a])
            if use_free_cx:
                if op == op_free_cx:
                    cx(target[a], target[b])
            if use_z_ladder:
                if op == op_z_ladder:
                    z(ladder[a])
            if use_cx_addr_addr:
                if op == op_cx_addr_addr:
                    cx(address[a], address[b])
            if use_ccx_addr_addr:
                if op == op_ccx_addr_addr:
                    x.ctrl(address[a], address[b], ladder[c])
            if use_cx_ladder_target:
                if op == op_cx_ladder_target:
                    cx(ladder[a], target[b])
            if use_measure_reset:
                if op == op_measure_reset:
                    h(ladder[a])
                    outcomes[b] = mz(ladder[a])
                    if outcomes[b]:
                        x(ladder[a])
            if use_measure_reset_output:
                if op == op_measure_reset_output:
                    h(target[a])
                    outcomes[b] = mz(target[a])
                    if outcomes[b]:
                        x(target[a])
            if use_parity_z:
                if op == op_parity_z:
                    parity = False
                    for j in range(mask_offsets[b], mask_offsets[b + 1]):
                        parity = parity != outcomes[mask_slots[j]]
                    if parity:
                        z(ladder[a])
            if use_parity_cz:
                if op == op_parity_cz:
                    parity = False
                    for j in range(mask_offsets[c], mask_offsets[c + 1]):
                        parity = parity != outcomes[mask_slots[j]]
                    if parity:
                        z.ctrl(ladder[a], ladder[b])
            if use_meas_uncompute:
                if op == op_meas_uncompute:
                    h(ladder[a])
                    fired = mz(ladder[a])
                    if fired:
                        x(ladder[a])
                        z.ctrl(ladder[b], address[c])

    _retain(primitives_unary_walk)
    return primitives_unary_walk
