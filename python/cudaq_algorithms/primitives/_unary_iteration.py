# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unary iteration: apply a per-address body once for each address value.

``unary_iteration_kernels`` mints a flat device kernel that walks a binary
tree over the address register (Babbush et al., `arXiv:1805.03662`,
Fig. 7): one clean ladder ancilla per address bit carries the "this
subtree is active" line, and the caller's body gates for address ``k``
fire controlled on the leaf line — i.e. exactly when the address register
equals ``k``. The walk descends once, iterates the leaves in address
order, and retraces once; sibling transitions reuse the parent line with
a CNOT, and deeper transitions fuse the uncompute of the old path with
the compute of the new one through a single guard Toffoli on
CNOT-conjugated address wires.

Toffoli cost (unitary-uncompute accounting — see below): a full tree of
``N = 2^num_address_bits`` addresses costs exactly ``3 N / 2 - 5``
Toffolis uncontrolled (``N >= 8``; 2 at ``N = 4``, 0 at ``N = 2``) and
``3 N / 2 - 1`` controlled. Partial trees (``num_items < N``) cost less;
the emitter reports the exact number as ``toffoli_count``. The paper's
headline count for unary iteration is ``N - 1`` Toffolis, but that
figure prices the AND-ancilla *uncomputation* at zero via
measurement-and-fixup. This library keeps every primitive strictly
unitary (statevector-testable, inverse-composable, no mid-circuit
measurement), and with unitary uncomputation the fused walk's
``3 N / 2 + O(1)`` is the best we found: an exhaustive search over
CNOT/Toffoli circuits (arbitrary CNOT-conjugated controls, dirty address
wires, extra ancillas) proves 2 Toffolis minimal for the controlled
two-address walk, 5 for the controlled four-address walk and 2 for the
uncontrolled four-address walk — all matched by this construction, and
all above the measurement-assisted ``N - 1``. The naive
compute/uncompute walk costs ``2 N - 4``.

T-count caveat (so the comparison stays honest under either metric):
roughly a third of this walk's Toffolis are the fused ones, whose
target is a *dirty* ladder line and therefore cost the generic 7 T,
while a compute-from-|0> AND costs 4 T (Gidney, arXiv:1709.06648).
In T gates the fused walk is therefore ~7.5 N against the naive
unitary walk's ~8 N — a ~6% advantage, not the 25% the Toffoli count
suggests. Both numbers beat nothing measured: the literature's
measured walk is 4 N T.

The callback pattern (READ THIS — it is the template for QROM and for
factory-composed SELECTs)
------------------------------------------------------------------------

CUDA-Q kernels are compiled from fixed Python source, so a callback can
never execute *inside* a kernel. The callback therefore runs at **factory
time**: ``body(k)`` is called once per address on the host and must return
the gate list for that address as data. The emitter flattens the whole
tree walk (ladder gadgets + body gates) into parallel opcode/operand
integer lists, and the minted kernel is a single flat interpreter loop
over those captured lists. Flatness is load-bearing: CUDA-Q's
control-variant generation rejects kernels that call other kernels, so
anything built this way stays ``cudaq.control``-compatible and can sit
inside a controlled SELECT.

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
  acts as the identity when the control is |0>. (A separate one-qubit
  view, not a leading qubit of a combined register, so callers never mix
  a qview with a bare qubit in one control set. Free body gates still
  execute at control |0>, but their conjugation pairing cancels them.)

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

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import cudaq

__all__ = ["UnaryIterationKernels", "unary_iteration_kernels"]

# Keep-alive registry for factory-minted kernels. CUDA-Q identifies a
# kernel by ``<function name>..<hex(id(decorator))>`` and retains compiled
# modules under that key without unretaining on deallocation; when a
# same-named factory kernel is created at a recycled ``id()`` of a dead
# one, the stale module collides with the new kernel and compilation fails
# with arbitrary errors. Pinning every minted kernel for the process
# lifetime keeps the ids from recycling; the cost is a few KB per factory
# call. The kernel names below are unique to this subpackage so they can
# never collide with other subpackages' registries.
_LIVE_KERNELS: list = []


def _retain(*kernels) -> None:
    _LIVE_KERNELS.extend(kernels)


# Interpreter opcodes (operands a, b, c index the signature's registers).
_OP_X_ADDR = 0  # x(address[a])
_OP_X_LADDER = 1  # x(ladder[a])
_OP_CX_ADDR_LADDER = 2  # cx(address[a], ladder[b])
_OP_CX_LADDER_LADDER = 3  # cx(ladder[a], ladder[b])
_OP_CCX = 4  # x.ctrl(ladder[a], address[b], ladder[c])
_OP_BODY_X = 5  # x.ctrl(ladder[a], target[b])
_OP_BODY_Y = 6  # y.ctrl(ladder[a], target[b])
_OP_BODY_Z = 7  # z.ctrl(ladder[a], target[b])
_OP_CX_CTRL_LADDER = 8  # cx(control[0], ladder[b])
_OP_CCX_CTRL = 9  # x.ctrl(control[0], address[b], ladder[c])
_OP_FREE_X = 10  # x(target[a])
_OP_FREE_CX = 11  # cx(target[a], target[b])
_OP_AND_TT = 12  # x.ctrl(target[a], target[b], work[c])
_OP_AND_WT = 13  # x.ctrl(work[a], target[b], work[c])
_OP_COPY_TW = 14  # cx(target[a], work[b])
_OP_BODY_X_W = 15  # x.ctrl(ladder[a], work[b], target[c])
_OP_BODY_Z_W = 16  # z.ctrl(ladder[a], work[b])
_OP_Z_LADDER = 17  # z(ladder[a])
_OP_CX_ADDR_ADDR = 18  # cx(address[a], address[b])
_OP_CCX_ADDR_ADDR = 19  # x.ctrl(address[a], address[b], ladder[c])
_OP_CX_LADDER_TARGET = 20  # cx(ladder[a], target[b])

_BODY_OPCODES = {"x": _OP_BODY_X, "y": _OP_BODY_Y, "z": _OP_BODY_Z}

# Extended body gates: name -> (opcode, operand kinds, leaf line slot).
# Operand kinds: "t" = target index, "w" = work index; the leaf slot names
# which operand position (0-based, in the emitted (a, b, c)) receives the
# active ladder line, or None for free (uncontrolled) gates.
_EXTENDED_BODY_GATES = {
    "free_x": (_OP_FREE_X, ("t", ), None),
    "free_cx": (_OP_FREE_CX, ("t", "t"), None),
    "and_tt": (_OP_AND_TT, ("t", "t", "w"), None),
    "and_wt": (_OP_AND_WT, ("w", "t", "w"), None),
    "copy_tw": (_OP_COPY_TW, ("t", "w"), None),
    "x_w": (_OP_BODY_X_W, ("w", "t"), 0),
    "z_w": (_OP_BODY_Z_W, ("w", ), 0),
    "sign": (_OP_Z_LADDER, (), 0),
}

_TOFFOLI_OPCODES = (_OP_CCX, _OP_CCX_CTRL, _OP_AND_TT, _OP_AND_WT,
                    _OP_BODY_X_W, _OP_CCX_ADDR_ADDR)

# Opcode -> operand positions (within (a, b, c)) that index the work
# register, used to infer the required work width from the emitted ops.
_WORK_OPERANDS = {
    _OP_AND_TT: (2, ),
    _OP_AND_WT: (0, 2),
    _OP_COPY_TW: (1, ),
    _OP_BODY_X_W: (1, ),
    _OP_BODY_Z_W: (1, ),
}

# Human-readable templates for describe(); {a}/{b}/{c} are the operands.
_OP_DESCRIPTIONS = {
    _OP_X_ADDR: "x(address[{a}])",
    _OP_X_LADDER: "x(ladder[{a}])",
    _OP_CX_ADDR_LADDER: "cx(address[{a}] -> ladder[{b}])",
    _OP_CX_LADDER_LADDER: "cx(ladder[{a}] -> ladder[{b}])",
    _OP_CCX: "ccx(ladder[{a}], address[{b}] -> ladder[{c}])  # Toffoli",
    _OP_BODY_X: "x(target[{b}]) ctrl ladder[{a}]",
    _OP_BODY_Y: "y(target[{b}]) ctrl ladder[{a}]",
    _OP_BODY_Z: "z(target[{b}]) ctrl ladder[{a}]",
    _OP_CX_CTRL_LADDER: "cx(control[0] -> ladder[{b}])",
    _OP_CCX_CTRL: "ccx(control[0], address[{b}] -> ladder[{c}])  # Toffoli",
    _OP_FREE_X: "x(target[{a}])",
    _OP_FREE_CX: "cx(target[{a}] -> target[{b}])",
    _OP_AND_TT: "ccx(target[{a}], target[{b}] -> work[{c}])  # Toffoli",
    _OP_AND_WT: "ccx(work[{a}], target[{b}] -> work[{c}])  # Toffoli",
    _OP_COPY_TW: "cx(target[{a}] -> work[{b}])",
    _OP_BODY_X_W: "x(target[{c}]) ctrl ladder[{a}], work[{b}]  # Toffoli",
    _OP_BODY_Z_W: "z(work[{b}]) ctrl ladder[{a}]",
    _OP_Z_LADDER: "z(ladder[{a}])",
    _OP_CX_ADDR_ADDR: "cx(address[{a}] -> address[{b}])",
    _OP_CCX_ADDR_ADDR:
    "ccx(address[{a}], address[{b}] -> ladder[{c}])  # Toffoli",
    _OP_CX_LADDER_TARGET: "cx(ladder[{a}] -> target[{b}])",
}


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
    ops: Any = None

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


def _trailing_ones(k: int) -> int:
    t = 0
    while k & 1:
        t += 1
        k >>= 1
    return t


def _walk_toffoli_count(num_address_bits: int,
                        num_items: int,
                        controlled: bool = False) -> int:
    """Exact Toffoli count of the fused walk, without emitting it.

    Cross-checked against the emitted instruction stream by the resource
    tests; used by ``QROM`` to price variants before minting anything.
    """
    n = num_address_bits
    extra = 1 if controlled else 0
    total = 2 * (n - 1 + extra)  # descent + unwind
    for k in range(num_items - 1):
        t = _trailing_ones(k)
        if t == 0:
            continue
        if n - t == 1 and not controlled:  # unguarded root crossing
            total += max(2 * t - 3, 0)
        else:
            total += 2 * t - 1
    return total


def _emit_walk(num_address_bits: int, num_items: int, controlled: bool,
               body: Callable[[int], Sequence[tuple[str, int]]]) -> list:
    """Flatten the fused tree walk into (opcode, a, b, c) instructions."""
    n = num_address_bits
    ops: list[tuple[int, int, int, int]] = []

    def emit(opcode: int, a: int = 0, b: int = 0, c: int = 0) -> None:
        ops.append((opcode, a, b, c))

    def emit_body(line: int, k: int) -> None:
        for item in body(k):
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
                emit(_BODY_OPCODES[gate], line, int(target))
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
                        raise ValueError(
                            f"body({k}) returned an invalid {name} "
                            f"qubit index {operand!r}")
                    slots[cursor] = int(operand)
                    cursor += 1
                emit(opcode, slots[0], slots[1], slots[2])
            else:
                raise ValueError(
                    f"body({k}) returned unsupported gate {gate!r}: the "
                    "unary-iteration body instruction set is 'x', 'y', 'z' "
                    "plus the extended gates documented in "
                    "cudaq_algorithms.primitives._unary_iteration")

    def wire(level: int) -> int:
        """Address wire examined at 1-based tree level ``level``."""
        return n - level

    def set_line(level: int, bit_is_one: bool) -> None:
        """ladder[level-1] ^= parent AND (address bit == value).

        Self-inverse: emitted both to compute a line (from |0>) and to
        clear it (against the same parent value); the descent, the
        unclamp/reclamp halves of deep transitions and the final unwind
        are all built from this one gadget.
        """
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

    def root_flip() -> None:
        """Flip the level-1 line between the two root siblings."""
        if controlled:
            emit(_OP_CX_CTRL_LADDER, 0, 0)
        else:
            emit(_OP_X_LADDER, 0)

    # Descent to leaf 0 (all address bits 0).
    for level in range(1, n + 1):
        set_line(level, False)
    emit_body(n - 1, 0)

    for k in range(num_items - 1):
        t = _trailing_ones(k)
        if t == 0:
            # Sibling flip at the deepest level: left XOR right = parent.
            if n == 1:
                root_flip()
            else:
                emit(_OP_CX_LADDER_LADDER, n - 2, n - 1)
        elif n - t == 1 and not controlled and t >= 2:
            # Unguarded root crossing: the fused differences at levels 2
            # and 3 are (b1^b2) and (b1^b2)(b1^b3) of the top address
            # bits — free CNOTs and one two-conjugated-wire Toffoli.
            for level in range(n, 3, -1):
                set_line(level, True)  # unclamp against old parents
            w1, w2, w3 = wire(1), wire(2), wire(3)
            emit(_OP_CX_ADDR_ADDR, w1, w2)
            emit(_OP_CX_ADDR_ADDR, w1, w3)
            emit(_OP_CCX_ADDR_ADDR, w2, w3, 2)
            emit(_OP_CX_ADDR_ADDR, w1, w3)
            emit(_OP_CX_ADDR_ADDR, w1, w2)
            emit(_OP_CX_ADDR_LADDER, w1, 1)
            emit(_OP_CX_ADDR_LADDER, w2, 1)
            root_flip()
            for level in range(4, n + 1):
                set_line(level, False)  # reclamp against new parents
        else:
            d = n - t  # flip level
            for level in range(n, d + 1, -1):
                set_line(level, True)  # unclamp against old parents
            # Fused level d+1: XOR in guard AND (bit_d ^ bit_{d+1}).
            wd, wd1 = wire(d), wire(d + 1)
            if d == 1 and not controlled:
                emit(_OP_CX_ADDR_LADDER, wd, d)
                emit(_OP_CX_ADDR_LADDER, wd1, d)
            else:
                emit(_OP_CX_ADDR_ADDR, wd, wd1)
                if d == 1:
                    emit(_OP_CCX_CTRL, 0, wd1, d)
                else:
                    emit(_OP_CCX, d - 2, wd1, d)
                emit(_OP_CX_ADDR_ADDR, wd, wd1)
            # Flip level d to the sibling subtree.
            if d == 1:
                root_flip()
            else:
                emit(_OP_CX_LADDER_LADDER, d - 2, d - 1)
            for level in range(d + 2, n + 1):
                set_line(level, False)  # reclamp against new parents
        emit_body(n - 1, k + 1)

    # Unwind from the last visited leaf.
    last = num_items - 1
    for level in range(n, 0, -1):
        set_line(level, ((last >> wire(level)) & 1) == 1)
    return ops


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
        Factory-time callback ``k -> sequence of (gate, target)`` (see
        the module docstring). Called exactly once per address, in
        address order, during this factory call.
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
    if int(num_address_bits) != num_address_bits or num_address_bits < 1:
        raise ValueError("num_address_bits must be a positive integer")
    num_address_bits = int(num_address_bits)
    capacity = 1 << num_address_bits
    if int(num_items) != num_items or not 1 <= num_items <= capacity:
        raise ValueError(
            f"num_items must be an integer in [1, 2^num_address_bits = "
            f"{capacity}], got {num_items}")
    num_items = int(num_items)

    ops = _emit_walk(num_address_bits, num_items, bool(controlled), body)
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


def _mint_interpreter(ops: list, controlled: bool, has_work: bool):
    """Mint the flat interpreter kernel over a flattened instruction list.

    Every emitted gate is self-inverse, so the inverse walk is this same
    interpreter over the reversed list — how ``kernel_adj`` is minted.
    """
    num_ops = len(ops)
    opcodes = [op[0] for op in ops]
    ops_a = [op[1] for op in ops]
    ops_b = [op[2] for op in ops]
    ops_c = [op[3] for op in ops]
    if num_ops == 0:
        # An empty list must never cross the kernel boundary
        # (cuda-quantum#4847): pad with one never-dispatched entry.
        opcodes, ops_a, ops_b, ops_c = [-1], [0], [0], [0]

    if controlled and has_work:

        @cudaq.kernel
        def primitives_unary_walk_work_ctrl(control: cudaq.qview,
                                            address: cudaq.qview,
                                            ladder: cudaq.qview,
                                            target: cudaq.qview,
                                            work: cudaq.qview):
            for i in range(num_ops):
                op = opcodes[i]
                a = ops_a[i]
                b = ops_b[i]
                c = ops_c[i]
                if op == 0:
                    x(address[a])
                if op == 1:
                    x(ladder[a])
                if op == 2:
                    cx(address[a], ladder[b])
                if op == 3:
                    cx(ladder[a], ladder[b])
                if op == 4:
                    x.ctrl(ladder[a], address[b], ladder[c])
                if op == 5:
                    x.ctrl(ladder[a], target[b])
                if op == 6:
                    y.ctrl(ladder[a], target[b])
                if op == 7:
                    z.ctrl(ladder[a], target[b])
                if op == 8:
                    cx(control[0], ladder[b])
                if op == 9:
                    x.ctrl(control[0], address[b], ladder[c])
                if op == 10:
                    x(target[a])
                if op == 11:
                    cx(target[a], target[b])
                if op == 12:
                    x.ctrl(target[a], target[b], work[c])
                if op == 13:
                    x.ctrl(work[a], target[b], work[c])
                if op == 14:
                    cx(target[a], work[b])
                if op == 15:
                    x.ctrl(ladder[a], work[b], target[c])
                if op == 16:
                    z.ctrl(ladder[a], work[b])
                if op == 17:
                    z(ladder[a])
                if op == 18:
                    cx(address[a], address[b])
                if op == 19:
                    x.ctrl(address[a], address[b], ladder[c])
                if op == 20:
                    cx(ladder[a], target[b])

        _retain(primitives_unary_walk_work_ctrl)
        return primitives_unary_walk_work_ctrl

    if has_work:

        @cudaq.kernel
        def primitives_unary_walk_work(address: cudaq.qview,
                                       ladder: cudaq.qview,
                                       target: cudaq.qview, work: cudaq.qview):
            for i in range(num_ops):
                op = opcodes[i]
                a = ops_a[i]
                b = ops_b[i]
                c = ops_c[i]
                if op == 0:
                    x(address[a])
                if op == 1:
                    x(ladder[a])
                if op == 2:
                    cx(address[a], ladder[b])
                if op == 3:
                    cx(ladder[a], ladder[b])
                if op == 4:
                    x.ctrl(ladder[a], address[b], ladder[c])
                if op == 5:
                    x.ctrl(ladder[a], target[b])
                if op == 6:
                    y.ctrl(ladder[a], target[b])
                if op == 7:
                    z.ctrl(ladder[a], target[b])
                if op == 10:
                    x(target[a])
                if op == 11:
                    cx(target[a], target[b])
                if op == 12:
                    x.ctrl(target[a], target[b], work[c])
                if op == 13:
                    x.ctrl(work[a], target[b], work[c])
                if op == 14:
                    cx(target[a], work[b])
                if op == 15:
                    x.ctrl(ladder[a], work[b], target[c])
                if op == 16:
                    z.ctrl(ladder[a], work[b])
                if op == 17:
                    z(ladder[a])
                if op == 18:
                    cx(address[a], address[b])
                if op == 19:
                    x.ctrl(address[a], address[b], ladder[c])
                if op == 20:
                    cx(ladder[a], target[b])

        _retain(primitives_unary_walk_work)
        return primitives_unary_walk_work

    if controlled:

        @cudaq.kernel
        def primitives_unary_walk_ctrl(control: cudaq.qview,
                                       address: cudaq.qview,
                                       ladder: cudaq.qview,
                                       target: cudaq.qview):
            for i in range(num_ops):
                op = opcodes[i]
                a = ops_a[i]
                b = ops_b[i]
                c = ops_c[i]
                if op == 0:
                    x(address[a])
                if op == 1:
                    x(ladder[a])
                if op == 2:
                    cx(address[a], ladder[b])
                if op == 3:
                    cx(ladder[a], ladder[b])
                if op == 4:
                    x.ctrl(ladder[a], address[b], ladder[c])
                if op == 5:
                    x.ctrl(ladder[a], target[b])
                if op == 6:
                    y.ctrl(ladder[a], target[b])
                if op == 7:
                    z.ctrl(ladder[a], target[b])
                if op == 8:
                    cx(control[0], ladder[b])
                if op == 9:
                    x.ctrl(control[0], address[b], ladder[c])
                if op == 10:
                    x(target[a])
                if op == 11:
                    cx(target[a], target[b])
                if op == 17:
                    z(ladder[a])
                if op == 18:
                    cx(address[a], address[b])
                if op == 19:
                    x.ctrl(address[a], address[b], ladder[c])
                if op == 20:
                    cx(ladder[a], target[b])

        _retain(primitives_unary_walk_ctrl)
        return primitives_unary_walk_ctrl

    @cudaq.kernel
    def primitives_unary_walk(address: cudaq.qview, ladder: cudaq.qview,
                              target: cudaq.qview):
        for i in range(num_ops):
            op = opcodes[i]
            a = ops_a[i]
            b = ops_b[i]
            c = ops_c[i]
            if op == 0:
                x(address[a])
            if op == 1:
                x(ladder[a])
            if op == 2:
                cx(address[a], ladder[b])
            if op == 3:
                cx(ladder[a], ladder[b])
            if op == 4:
                x.ctrl(ladder[a], address[b], ladder[c])
            if op == 5:
                x.ctrl(ladder[a], target[b])
            if op == 6:
                y.ctrl(ladder[a], target[b])
            if op == 7:
                z.ctrl(ladder[a], target[b])
            if op == 10:
                x(target[a])
            if op == 11:
                cx(target[a], target[b])
            if op == 17:
                z(ladder[a])
            if op == 18:
                cx(address[a], address[b])
            if op == 19:
                x.ctrl(address[a], address[b], ladder[c])
            if op == 20:
                cx(ladder[a], target[b])

    _retain(primitives_unary_walk)
    return primitives_unary_walk
