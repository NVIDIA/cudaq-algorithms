# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unary iteration with measurement-based ancilla uncomputation.

The clean-input logical SELECT map matches ``unary_iteration_kernels``.
A plain retrace tree computes temporary ANDs, then erases each in the X
basis. Outcome one applies a CZ to its two inputs and resets the measured
line with X. Both outcomes have the same logical action, including on
superposed controls. The uncontrolled root is linear and needs no AND or
measurement; ``controlled=True`` folds an external control into the root.

For ``n`` address bits, ``m`` visited addresses and control flag ``c``,
``C = m + n - 2 + c - popcount(m-1)`` tree ANDs and measurements are emitted.
On a full tree this is ``2**n - 2 + c``. Unlike a restricted-input promise,
this implementation acts as identity for *all* addresses ``k >= m``.

``toffoli_count`` includes tree and body Toffolis. Only the temporary tree
ANDs admit a theoretical 4T compute/0T erasure model with a compatible
synthesis (Gidney, arXiv:1709.06648, Fig. 3). Ordinary Toffolis are emitted
here; no compiled T-count claim is made. Body targets need not be clean.
The tree construction follows Babbush et al., arXiv:1805.03662, Sec. III.A,
Figs. 4–7. Qualtran was consulted for conventions; no code was adopted.

Kernels take ``(address, ladder, target[, work])``, with a leading
one-qubit ``control`` view when requested. Registers must be disjoint;
``ladder`` is clean at entry and exit. Body instructions and work-space
preconditions follow the coherent factory. ``kernel_adj`` reverses the
logical SELECT action on this clean subspace; it is not the inverse of an
arbitrary measurement channel, nor a reversed measurement tape.

Use ``cudaq.get_state`` or ``cudaq.observe`` on a compatible dynamic-circuit
target. ``cudaq.run`` needs an enclosing kernel that returns a classical
value. CUDA-Q 0.16 rejects measurement-feedback kernels under
``cudaq.sample``; nested acceptance on older releases is not support. Automatic
``cudaq.control`` and ``cudaq.adjoint`` are outside the contract; request
``controlled=True`` and use ``kernel_adj`` explicitly. The tests exercise
CUDA-Q 0.16 on qpp-cpu. See the measured-QROM guide for runnable examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import cudaq

from ._unary_iteration import (
    _BODY_OPCODES, _CONTROL_OPS, _EXTENDED_BODY_GATES, _OP_AND_TT, _OP_AND_WT,
    _OP_BODY_X, _OP_BODY_X_W, _OP_BODY_Y, _OP_BODY_Z, _OP_BODY_Z_W, _OP_CCX,
    _OP_CCX_ADDR_ADDR, _OP_CCX_CTRL, _OP_COPY_TW, _OP_CX_ADDR_ADDR,
    _OP_CX_ADDR_LADDER, _OP_CX_CTRL_LADDER, _OP_CX_LADDER_LADDER,
    _OP_CX_LADDER_TARGET, _OP_DESCRIPTIONS, _OP_FREE_CX, _OP_FREE_X,
    _OP_X_ADDR, _OP_X_LADDER, _OP_Z_LADDER, _TOFFOLI_OPCODES, _WORK_OPERANDS,
    _WORK_OPS)

__all__ = ["MeasuredUnaryIterationKernels", "measured_unary_iteration_kernels"]

# Keep-alive registry (see _unary_iteration for the rationale); the
# minted kernel names below (measured_walk*) are unique to this module.
_LIVE_KERNELS: list = []


def _retain(*kernels) -> None:
    _LIVE_KERNELS.extend(kernels)


# Measured-uncompute opcodes, separate from coherent dispatch.
# Both erase ladder[a]: h(ladder[a]); m = mz(ladder[a]); on outcome 1,
# x(ladder[a]) resets the line to |0> and the CZ repairs the phase kick
# against the same two inputs the AND was computed from (any needed
# X-conjugation of the address wire is emitted around the gadget).
# Keep 21 available for coherent SELECT-COPY; these descriptions never
# mutate the shared coherent dictionary.
_OP_MEAS_UNCOMPUTE = 22  # ... fixup z.ctrl(ladder[b], address[c])
_OP_MEAS_UNCOMPUTE_CTRL = 23  # ... fixup z.ctrl(control[0], address[c])

_MEASURED_OPCODES = (_OP_MEAS_UNCOMPUTE, _OP_MEAS_UNCOMPUTE_CTRL)

# List this interpreter's actual dispatch, rather than inheriting new
# coherent-only opcodes when that interpreter gains constructions.
_MEASURED_BASE_OPS = frozenset({
    _OP_X_ADDR, _OP_X_LADDER, _OP_CX_ADDR_LADDER, _OP_CX_LADDER_LADDER,
    _OP_CCX, _OP_BODY_X, _OP_BODY_Y, _OP_BODY_Z, _OP_FREE_X, _OP_FREE_CX,
    _OP_Z_LADDER, _OP_CX_ADDR_ADDR, _OP_CCX_ADDR_ADDR, _OP_CX_LADDER_TARGET,
    _OP_MEAS_UNCOMPUTE
})

# Display order matches kernel execution (reset X, then fixup CZ; the two
# commute). The _CTRL template takes no ladder operand: b is emitted as 0
# and unused — the fixup control is the external control line.
_OP_DESCRIPTIONS_MEASURED = dict(_OP_DESCRIPTIONS)
_OP_DESCRIPTIONS_MEASURED[_OP_MEAS_UNCOMPUTE] = (
    "h(ladder[{a}]); mz -> x(ladder[{a}]), "
    "fixup z.ctrl(ladder[{b}], address[{c}])  # measured uncompute")
_OP_DESCRIPTIONS_MEASURED[_OP_MEAS_UNCOMPUTE_CTRL] = (
    "h(ladder[{a}]); mz -> x(ladder[{a}]), "
    "fixup z.ctrl(control[0], address[{c}])  # measured uncompute")


@dataclass(frozen=True)
class MeasuredUnaryIterationKernels:
    """The minted measured walk (see the module docstring for signatures).

    Attributes
    ----------
    kernel, kernel_adj
        The walk and its inverse walk (``kernel_adj`` is ``None`` when
        minted with ``include_adjoint=False``). Both contain mid-circuit
        measurements: use ``cudaq.get_state`` / ``cudaq.observe``, or a
        returning entry wrapper for ``cudaq.run``. Do not
        pass them to ``cudaq.control`` — the controlled variant must be
        minted with ``controlled=True`` instead.
    num_address, num_ladder, num_items
        Register widths (``num_ladder == num_address`` clean ancillas)
        and the number of iterated addresses.
    controlled
        Whether ``kernel`` takes the leading one-qubit control view.
    num_work
        Width of the trailing clean ``work`` view (0: no work view in the
        signature).
    toffoli_count
        Toffolis in one application — AND computes only (plus any body
        Toffolis); the uncomputes are measured and cost none.
    num_measurements
        Mid-circuit measurements in one application (one per measured
        uncompute, equal to the walk's AND computes on a trivial body).
    """

    kernel: Any
    kernel_adj: Any
    num_address: int
    num_ladder: int
    num_items: int
    controlled: bool
    toffoli_count: int
    num_measurements: int
    num_work: int = 0
    ops: Any = None

    @property
    def tree_and_count(self) -> int:
        """Temporary clean-tree AND computes, excluding body Toffolis."""
        return self.num_measurements

    @property
    def body_toffoli_count(self) -> int:
        """Body Toffolis; their targets need not start clean."""
        return self.toffoli_count - self.tree_and_count

    def describe(self) -> str:
        """Decode the instruction tape into a human-readable gate listing.

        One line per instruction, in execution order. Lines tagged
        ``# Toffoli`` count both tree ANDs and any body Toffolis;
        lines tagged ``# measured uncompute`` are the Gidney gadgets
        ``num_measurements`` counts (each is one h, one mz and, on
        outcome 1, one fix-up CZ and one reset X).
        """
        control = ", controlled" if self.controlled else ""
        header = (f"measured unary-iteration walk: {self.num_items} "
                  f"addresses over {self.num_address} bits{control}; "
                  f"{self.toffoli_count} Toffolis, "
                  f"{self.num_measurements} measurements")
        lines = [
            _OP_DESCRIPTIONS_MEASURED[op].format(a=a, b=b, c=c)
            for op, a, b, c in self.ops
        ]
        return "\n".join([header] + lines)


def _measured_walk_toffoli_count(num_address_bits: int,
                                 num_items: int,
                                 controlled: bool = False) -> int:
    """Exact AND-compute count of the measured walk, without emitting it.

    Descent pays one AND per level below the root (the root is linear
    uncontrolled, one AND controlled), and the transition from leaf ``k``
    recomputes one AND per trailing one of ``k``; sibling crossings are
    CNOTs. On full trees this is ``N - 2`` uncontrolled, ``N - 1``
    controlled. The measurement count equals this on any tree shape
    (every AND-computed line is measured away exactly once).
    """
    # Sum_{k=0}^{m-2} trailing_ones(k) = m - 1 - popcount(m - 1).
    return (num_items + num_address_bits - 2 + int(controlled) -
            (num_items - 1).bit_count())


def _emit_body_gates(ops: list, line: int, k: int,
                     gates: Sequence[tuple]) -> None:
    """Append the validated body instructions for address ``k``."""
    for item in gates:
        gate = item[0]
        operands = item[1:]
        if gate in _BODY_OPCODES:
            if len(operands) != 1:
                raise ValueError(f"body({k}) returned {gate!r} with "
                                 f"{len(operands)} operands (expected 1)")
            target = operands[0]
            if int(target) != target or target < 0:
                raise ValueError(f"body({k}) returned an invalid target "
                                 f"qubit index {target!r}")
            ops.append((_BODY_OPCODES[gate], line, int(target), 0))
        elif gate in _EXTENDED_BODY_GATES:
            opcode, kinds, leaf_slot = _EXTENDED_BODY_GATES[gate]
            if len(operands) != len(kinds):
                raise ValueError(
                    f"body({k}) returned {gate!r} with {len(operands)} "
                    f"operands (expected {len(kinds)})")
            slots = [0, 0, 0]
            cursor = 0
            if leaf_slot is not None:
                slots[leaf_slot] = line
                cursor = leaf_slot + 1
            for kind, operand in zip(kinds, operands):
                if int(operand) != operand or operand < 0:
                    name = "target" if kind == "t" else "work"
                    raise ValueError(f"body({k}) returned an invalid {name} "
                                     f"qubit index {operand!r}")
                slots[cursor] = int(operand)
                cursor += 1
            ops.append((opcode, slots[0], slots[1], slots[2]))
        else:
            raise ValueError(
                f"body({k}) returned unsupported gate {gate!r}: the "
                "unary-iteration body instruction set is 'x', 'y', 'z' "
                "plus the extended gates documented in "
                "cudaq_algorithms.primitives._unary_iteration")


def _emit_measured_walk(num_address_bits: int, num_items: int,
                        controlled: bool, bodies: Sequence[Sequence[tuple]],
                        reverse: bool) -> list:
    """Flatten the measured retrace walk into (opcode, a, b, c) tuples.

    ``reverse=True`` emits the inverse walk: the leaves are visited in
    descending address order with each body's (self-inverse) gates
    reversed. The transition machinery is direction-agnostic — between
    consecutive leaves it measures away the levels below the common
    ancestor against the *outgoing* leaf's bits and recomputes them
    against the incoming leaf's bits — so one emitter serves both.
    """
    n = num_address_bits
    ops: list[tuple[int, int, int, int]] = []

    def emit(opcode: int, a: int = 0, b: int = 0, c: int = 0) -> None:
        ops.append((opcode, a, b, c))

    def wire(level: int) -> int:
        """Address wire examined at 1-based tree level ``level``."""
        return n - level

    def bit_of(leaf: int, level: int) -> bool:
        return ((leaf >> wire(level)) & 1) == 1

    def compute_line(level: int, bit_is_one: bool) -> None:
        """ladder[level-1] <- parent AND (address bit == value): the AND
        compute (a Toffoli onto the clean line; linear at the
        uncontrolled root)."""
        a = wire(level)
        if not bit_is_one:
            emit(_OP_X_ADDR, a)
        if level == 1:
            if controlled:
                emit(_OP_CCX_CTRL, 0, a, 0)
            else:
                emit(_OP_CX_ADDR_LADDER, a, 0)
        else:
            emit(_OP_CCX, level - 2, a, level - 1)
        if not bit_is_one:
            emit(_OP_X_ADDR, a)

    def uncompute_line(level: int, bit_is_one: bool) -> None:
        """Erase ladder[level-1] (currently parent AND (bit == value)):
        the Gidney measured uncompute, X-conjugated as the compute was.
        The uncontrolled root is linear and uncomputed by its own CNOT
        (no measurement)."""
        a = wire(level)
        if level == 1 and not controlled:
            compute_line(level, bit_is_one)  # linear, self-inverse
            return
        if not bit_is_one:
            emit(_OP_X_ADDR, a)
        if level == 1:
            emit(_OP_MEAS_UNCOMPUTE_CTRL, 0, 0, a)
        else:
            emit(_OP_MEAS_UNCOMPUTE, level - 1, level - 2, a)
        if not bit_is_one:
            emit(_OP_X_ADDR, a)

    def cross_to_sibling(level: int) -> None:
        """Flip ladder[level-1] between the two siblings: given the
        parent, exactly one of the two is active, so their XOR is the
        parent line itself."""
        if level == 1:
            if controlled:
                emit(_OP_CX_CTRL_LADDER, 0, 0)
            else:
                emit(_OP_X_LADDER, 0)
        else:
            emit(_OP_CX_LADDER_LADDER, level - 2, level - 1)

    def body_at(k: int) -> None:
        gates = bodies[k]
        if reverse:
            gates = list(reversed(gates))
        _emit_body_gates(ops, n - 1, k, gates)

    leaves = list(range(num_items))
    if reverse:
        leaves.reverse()

    # Descent to the first leaf.
    for level in range(1, n + 1):
        compute_line(level, bit_of(leaves[0], level))
    body_at(leaves[0])

    for previous, leaf in zip(leaves, leaves[1:]):
        # Retrace up to the common ancestor, measuring the path away ...
        d = n - (previous ^ leaf).bit_length() + 1  # topmost changed level
        for level in range(n, d, -1):
            uncompute_line(level, bit_of(previous, level))
        # ... cross to the sibling subtree, and descend into it.
        cross_to_sibling(d)
        for level in range(d + 1, n + 1):
            compute_line(level, bit_of(leaf, level))
        body_at(leaf)

    # Measure the final path away.
    for level in range(n, 0, -1):
        uncompute_line(level, bit_of(leaves[-1], level))
    return ops


def measured_unary_iteration_kernels(
        num_address_bits: int,
        num_items: int,
        body: Callable[[int], Sequence[tuple]],
        *,
        controlled: bool = False,
        include_adjoint: bool = True,
        num_work: int | None = None) -> MeasuredUnaryIterationKernels:
    """Mint the measured unary-iteration walk for a factory-time body.

    The parameters and the body contract are identical to
    ``unary_iteration_kernels`` (see ``_unary_iteration``); ``body`` is
    called exactly once per address, in address order, and the returned
    gate lists are reused (reversed) for the inverse walk. The minted
    kernels contain mid-circuit measurements: they compose as sub-kernels
    of plain kernels, but must not be passed to ``cudaq.control`` (mint
    with ``controlled=True`` instead) nor run under ``cudaq.sample``.

    Parameters
    ----------
    num_address_bits
        Address register width (>= 1).
    num_items
        Number of iterated addresses (``1 <= num_items <=
        2^num_address_bits``); addresses ``k >= num_items`` are never
        entered and the walk acts as the identity on them.
    body
        Factory-time callback ``k -> sequence of (gate, operands...)``.
    controlled
        Mint the externally controlled walk (leading one-qubit view);
        the control folds into the tree root as one extra AND.
    include_adjoint
        Mint ``kernel_adj`` (the reverse-order walk with reversed
        bodies); pass ``False`` when the walk is known to be an
        involution (commuting self-inverse body, e.g. X-only QROM).
    num_work
        Width of the trailing clean ``work`` view. ``None`` (default)
        infers the width from the body's work usage.
    """
    if int(num_address_bits) != num_address_bits or num_address_bits < 1:
        raise ValueError("num_address_bits must be a positive integer")
    num_address_bits = int(num_address_bits)
    capacity = 1 << num_address_bits
    if int(num_items) != num_items or not 1 <= num_items <= capacity:
        raise ValueError(
            f"num_items must be an integer in [1, 2^num_address_bits = "
            f"{capacity}], got {num_items}")
    num_items = int(num_items)

    bodies = [list(body(k)) for k in range(num_items)]
    ops = _emit_measured_walk(num_address_bits,
                              num_items,
                              bool(controlled),
                              bodies,
                              reverse=False)
    required_work = 0
    for op in ops:
        for position in _WORK_OPERANDS.get(op[0], ()):
            required_work = max(required_work, op[1 + position] + 1)
    if num_work is None:
        num_work = required_work
    if int(num_work) != num_work or num_work < required_work:
        raise ValueError(
            f"num_work must be an integer >= the body's work usage "
            f"({required_work}), got {num_work}")
    num_work = int(num_work)

    toffolis = sum(1 for op in ops if op[0] in _TOFFOLI_OPCODES)
    measurements = sum(1 for op in ops if op[0] in _MEASURED_OPCODES)
    kernel = _mint_measured_interpreter(ops, bool(controlled), num_work > 0)
    kernel_adj = None
    if include_adjoint:
        adj_ops = _emit_measured_walk(num_address_bits,
                                      num_items,
                                      bool(controlled),
                                      bodies,
                                      reverse=True)
        kernel_adj = _mint_measured_interpreter(adj_ops, bool(controlled),
                                                num_work > 0)
    return MeasuredUnaryIterationKernels(kernel=kernel,
                                         kernel_adj=kernel_adj,
                                         num_address=num_address_bits,
                                         num_ladder=num_address_bits,
                                         num_items=num_items,
                                         controlled=bool(controlled),
                                         toffoli_count=toffolis,
                                         num_measurements=measurements,
                                         num_work=num_work,
                                         ops=tuple(ops))


def _mint_measured_interpreter(ops: list, controlled: bool, has_work: bool):
    """Mint a measured interpreter with the coherent dispatch safeguards.

    Unsupported opcodes are rejected before compilation. Captured per-opcode
    booleans prune absent arms, avoiding invalid dead multi-control IR.
    """
    supported = _MEASURED_BASE_OPS
    if controlled:
        supported = supported | _CONTROL_OPS | {_OP_MEAS_UNCOMPUTE_CTRL}
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
        def primitives_measured_walk_work_ctrl(control: cudaq.qview,
                                               address: cudaq.qview,
                                               ladder: cudaq.qview,
                                               target: cudaq.qview,
                                               work: cudaq.qview):
            fired = False
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

        _retain(primitives_measured_walk_work_ctrl)
        return primitives_measured_walk_work_ctrl

    if has_work:

        @cudaq.kernel
        def primitives_measured_walk_work(address: cudaq.qview,
                                          ladder: cudaq.qview,
                                          target: cudaq.qview,
                                          work: cudaq.qview):
            fired = False
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
                if use_meas_uncompute:
                    if op == op_meas_uncompute:
                        h(ladder[a])
                        fired = mz(ladder[a])
                        if fired:
                            x(ladder[a])
                            z.ctrl(ladder[b], address[c])

        _retain(primitives_measured_walk_work)
        return primitives_measured_walk_work

    if controlled:

        @cudaq.kernel
        def primitives_measured_walk_ctrl(control: cudaq.qview,
                                          address: cudaq.qview,
                                          ladder: cudaq.qview,
                                          target: cudaq.qview):
            fired = False
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

        _retain(primitives_measured_walk_ctrl)
        return primitives_measured_walk_ctrl

    @cudaq.kernel
    def primitives_measured_walk(address: cudaq.qview, ladder: cudaq.qview,
                                 target: cudaq.qview):
        fired = False
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
            if use_meas_uncompute:
                if op == op_meas_uncompute:
                    h(ladder[a])
                    fired = mz(ladder[a])
                    if fired:
                        x(ladder[a])
                        z.ctrl(ladder[b], address[c])

    _retain(primitives_measured_walk)
    return primitives_measured_walk
