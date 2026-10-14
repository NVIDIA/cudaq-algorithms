# SPDX-License-Identifier: Apache-2.0
"""Backend-independent unary iteration tapes and validation."""
from typing import Callable, Sequence

Op = tuple[int, int, int, int]
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
_OP_CCX_LADDER_LADDER_TARGET = 21  # ccx(ladder[a], ladder[b], target[c])
# Measured-uncompute gadgets (measured_unary_iteration_kernels). Both erase
# ladder[a] in the X basis: h(ladder[a]); fired = mz(ladder[a]); on outcome
# one, x(ladder[a]) resets the line to |0> and a CZ against the two inputs
# the AND was computed from repairs the phase kick (any X-conjugation of the
# address wire is emitted around the gadget by the measured emitter).
_OP_MEAS_UNCOMPUTE = 22  # fixup z.ctrl(ladder[b], address[c])
_OP_MEAS_UNCOMPUTE_CTRL = 23  # fixup z.ctrl(control[0], address[c])

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

# Dispatch coverage of the four interpreter variants (_mint_interpreter):
# every variant handles the base set; the control/work bundles are
# handled only by kernels whose signature carries that register view.
# An opcode outside the minted variant's set would be SILENTLY SKIPPED
# by the dispatch loop, so _mint_interpreter rejects such tapes at mint
# time; test_primitives_interpreter.py pins each opcode's action under
# every variant that supports it.
_BASE_OPS = frozenset({
    _OP_X_ADDR, _OP_X_LADDER, _OP_CX_ADDR_LADDER, _OP_CX_LADDER_LADDER,
    _OP_CCX, _OP_BODY_X, _OP_BODY_Y, _OP_BODY_Z, _OP_FREE_X, _OP_FREE_CX,
    _OP_Z_LADDER, _OP_CX_ADDR_ADDR, _OP_CCX_ADDR_ADDR, _OP_CX_LADDER_TARGET,
    _OP_CCX_LADDER_LADDER_TARGET
})
_CONTROL_OPS = frozenset({_OP_CX_CTRL_LADDER, _OP_CCX_CTRL})
_WORK_OPS = frozenset(
    {_OP_AND_TT, _OP_AND_WT, _OP_COPY_TW, _OP_BODY_X_W, _OP_BODY_Z_W})
# Measured gadgets: the plain gadget is dispatched by every variant, the
# control-rooted gadget by the controlled variants. They stay out of
# _BASE_OPS so test_primitives_interpreter.py keeps pinning the coherent
# set exactly; the measured tests pin these two.
_MEASURED_OPS = frozenset({_OP_MEAS_UNCOMPUTE})
_MEASURED_CONTROL_OPS = frozenset({_OP_MEAS_UNCOMPUTE_CTRL})

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
# The measured gadgets' templates are added after the literal: a
# multi-line value inside the literal makes yapf split every entry of
# the table.
_OP_DESCRIPTIONS[_OP_MEAS_UNCOMPUTE] = (
    "h(ladder[{a}]); mz -> x(ladder[{a}]), "
    "fixup z.ctrl(ladder[{b}], address[{c}])  # measured uncompute")
_OP_DESCRIPTIONS[_OP_MEAS_UNCOMPUTE_CTRL] = (
    "h(ladder[{a}]); mz -> x(ladder[{a}]), "
    "fixup z.ctrl(control[0], address[{c}])  # measured uncompute")


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

    Closed form of the per-address recurrence, cross-checked against the
    emitted instruction stream and against the explicit loop by the
    resource tests: among the transitions ``k = 0 .. m-2`` exactly
    ``floor((m - 1 + 2^t) / 2^(t+1))`` have ``t`` trailing ones, and a
    transition with ``t >= 1`` trailing ones costs ``2t - 1`` Toffolis,
    except the uncontrolled root crossing (``t == n - 1``) at
    ``max(2t - 3, 0)``. Valid for ``1 <= num_items <= 2^num_address_bits``,
    which the factory enforces. ``QROM`` prices every candidate
    construction with this before minting anything, so it must stay
    O(num_address_bits): a per-address loop made wide, sparse tables take
    seconds per address bit.
    """
    n = num_address_bits
    m = num_items
    if not 1 <= m <= (1 << n):
        raise ValueError(
            f"num_items must be in [1, 2^{n}] for the walk count, got {m}")
    total = 2 * (n - 1 + (1 if controlled else 0))  # descent + unwind
    for t in range(1, n):
        transitions = (m - 1 + (1 << t)) >> (t + 1)
        if transitions == 0:
            break
        if n - t == 1 and not controlled:  # unguarded root crossing
            total += transitions * max(2 * t - 3, 0)
        else:
            total += transitions * (2 * t - 1)
    return total


def _emit_walk(num_address_bits: int, num_items: int, controlled: bool,
               body: Callable[[int], Sequence[tuple[str, int]]]) -> list:
    """Flatten the fused tree walk into (opcode, a, b, c) instructions."""
    n = num_address_bits
    ops: list[tuple[int, int, int, int]] = []

    def emit(opcode: int, a: int = 0, b: int = 0, c: int = 0) -> None:
        ops.append((opcode, a, b, c))

    def emit_body(line: int, k: int) -> None:
        emit_body_gates(ops, line, k, body(k))

    def wire(level: int) -> int:
        """Address wire examined at 1-based tree level ``level``."""
        return n - level

    def set_line(level: int, bit_is_one: bool) -> None:
        """ladder[level-1] ^= parent AND (address bit == value).

        Self-inverse: emitted both to compute a line (from ``|0>``) and to
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

    def sibling_flip(level: int) -> None:
        """Move the level line to its sibling: left XOR right = parent.

        Exactly one of the two sibling indicators is active given the
        parent, so one CNOT from the parent line retargets the child
        line in place. ``level == n`` is the leaf/word-line transition
        between even and odd addresses; ``level == 1`` has no parent
        line and degenerates to ``root_flip``.
        """
        if level == 1:
            root_flip()
        else:
            emit(_OP_CX_LADDER_LADDER, level - 2, level - 1)

    def unguarded_root_crossing() -> None:
        """Rewrite levels 1-3 across an uncontrolled root, Toffoli-free
        at levels 1-2.

        With no guard line above the root, the fused differences at
        levels 2 and 3 are (b1^b2) and (b1^b2)(b1^b3) of the top address
        bits: level 2 moves by two free CNOTs, level 3 by one Toffoli on
        two CNOT-conjugated address wires, and level 1 by ``root_flip``.
        """
        w1, w2, w3 = wire(1), wire(2), wire(3)
        emit(_OP_CX_ADDR_ADDR, w1, w2)
        emit(_OP_CX_ADDR_ADDR, w1, w3)
        emit(_OP_CCX_ADDR_ADDR, w2, w3, 2)
        emit(_OP_CX_ADDR_ADDR, w1, w3)
        emit(_OP_CX_ADDR_ADDR, w1, w2)
        emit(_OP_CX_ADDR_LADDER, w1, 1)
        emit(_OP_CX_ADDR_LADDER, w2, 1)
        root_flip()

    # Descent to leaf 0 (all address bits 0).
    for level in range(1, n + 1):
        set_line(level, False)
    emit_body(n - 1, 0)

    for k in range(num_items - 1):
        t = _trailing_ones(k)
        if t == 0:
            sibling_flip(n)
        elif n - t == 1 and not controlled and t >= 2:
            for level in range(n, 3, -1):
                set_line(level, True)  # unclamp against old parents
            unguarded_root_crossing()
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
            sibling_flip(d)
            for level in range(d + 2, n + 1):
                set_line(level, False)  # reclamp against new parents
        emit_body(n - 1, k + 1)

    # Unwind from the last visited leaf.
    last = num_items - 1
    for level in range(n, 0, -1):
        set_line(level, ((last >> wire(level)) & 1) == 1)
    return ops


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


emit_body_gates = _emit_body_gates
_MEASURED_OPCODES = (_OP_MEAS_UNCOMPUTE, _OP_MEAS_UNCOMPUTE_CTRL)

_ADDRESS_OPERANDS = {
    _OP_X_ADDR: (0, ),
    _OP_CX_ADDR_LADDER: (0, ),
    _OP_CCX: (1, ),
    _OP_CCX_CTRL: (1, ),
    _OP_CX_ADDR_ADDR: (0, 1),
    _OP_CCX_ADDR_ADDR: (0, 1),
    _OP_MEAS_UNCOMPUTE: (2, ),
    _OP_MEAS_UNCOMPUTE_CTRL: (2, ),
}


def materialize_bodies(body: Callable[[int], Sequence[tuple]],
                       num_items: int) -> tuple[tuple[tuple, ...], ...]:
    """Consume each callback and its gate iterable once, in address order."""
    return tuple(
        tuple(tuple(gate) for gate in body(k)) for k in range(num_items))


def validate_unary_inputs(num_address_bits: int,
                          num_items: int) -> tuple[int, int]:
    if int(num_address_bits) != num_address_bits or num_address_bits < 1:
        raise ValueError("num_address_bits must be a positive integer")
    num_address_bits = int(num_address_bits)
    capacity = 1 << num_address_bits
    if int(num_items) != num_items or not 1 <= num_items <= capacity:
        raise ValueError(
            f"num_items must be an integer in [1, 2^num_address_bits = "
            f"{capacity}], got {num_items}")
    num_items = int(num_items)

    return num_address_bits, num_items


def infer_num_work(ops: Sequence[Op], num_work: int | None) -> int:
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

    return num_work


# Persistent cache outcomes; SELECT-COPY extensions are registered below.
_OP_MEASURE_RESET = 24
_OP_MEASURE_RESET_OUTPUT = 25
_OP_PARITY_Z_LADDER = 26
_OP_PARITY_CZ_LADDER = 27
_FEEDBACK_OPS = frozenset({24, 25, 26, 27})
_LADDER_OPERANDS = {
    1: (0, ),
    2: (1, ),
    3: (0, 1),
    4: (0, 2),
    5: (0, ),
    6: (0, ),
    7: (0, ),
    8: (1, ),
    9: (2, ),
    15: (0, ),
    16: (0, ),
    17: (0, ),
    19: (2, ),
    20: (0, ),
    21: (0, 1),
    22: (0, 1),
    23: (0, ),
    24: (0, ),
    26: (0, ),
    27: (0, 1),
}
_TARGET_OPERANDS = {
    5: (1, ),
    6: (1, ),
    7: (1, ),
    10: (0, ),
    11: (0, 1),
    12: (0, 1),
    13: (1, ),
    14: (0, ),
    15: (2, ),
    20: (1, ),
    21: (2, ),
    25: (0, ),
}
_OP_DESCRIPTIONS.update({
    24:
    'h(ladder[{a}]); mz -> result[{b}]; reset(ladder[{a}])',
    25:
    'h(output[{a}]); mz -> result[{b}]; reset(output[{a}])',
    26:
    'z(ladder[{a}]) if parity(mask[{b}])',
    27:
    'cz(ladder[{a}], ladder[{b}]) if parity(mask[{c}])',
})


def validate_feedback_ops(ops: Sequence[Op],
                          *,
                          parity_masks,
                          num_results: int,
                          num_address: int,
                          num_ladder: int,
                          num_output: int,
                          controlled: bool,
                          num_work: int = 0,
                          initialized_results=()) -> None:
    """Validate a complete tape, or a phase tape with explicit incoming slots.

    Persistent outcomes are initialized once per invocation. The runtime must
    additionally opt in to feedback dispatch; registering these schemas does
    not silently enable opcodes in the existing interpreter.
    """
    widths = (num_results, num_address, num_ladder, num_output, num_work)
    if any(not isinstance(n, int) or n < 0 for n in widths):
        raise ValueError('register widths must be nonnegative integers')
    for mask in parity_masks:
        if any(not isinstance(t, int) or not 0 <= t < num_results
               for t in mask):
            raise ValueError('parity mask result slot out of bounds')
        if len(set(mask)) != len(mask):
            raise ValueError('parity mask slots must be unique')
    initialized = set(initialized_results)
    if any(not isinstance(t, int) or not 0 <= t < num_results
           for t in initialized):
        raise ValueError('initialized result slot out of bounds')
    allowed = _BASE_OPS | _MEASURED_OPS | _FEEDBACK_OPS | _SELECT_COPY_OPS
    if controlled:
        allowed |= _CONTROL_OPS | _MEASURED_CONTROL_OPS
    if num_work:
        allowed |= _WORK_OPS
    for op in ops:
        if len(op) != 4 or any(not isinstance(x, int) or x < 0 for x in op):
            raise ValueError('operations require four nonnegative integers')
        code, a, b, c = op
        if code not in allowed:
            raise ValueError('opcode unsupported by register signature')
        for roles, width in ((_ADDRESS_OPERANDS,
                              num_address), (_LADDER_OPERANDS, num_ladder),
                             (_TARGET_OPERANDS, num_output), (_WORK_OPERANDS,
                                                              num_work)):
            if any(op[1 + pos] >= width for pos in roles.get(code, ())):
                raise ValueError('operation wire out of bounds')
        if code in (21, 30, 31) and a == b:
            raise ValueError('control and target must be distinct')
        if code in (28, 29) and len({a, b, c}) != 3:
            raise ValueError('three ladder operands must be distinct')
        if code == 31 and c != 0:
            raise ValueError('reserved operand must be zero')
        if code in (24, 25):
            if c != 0 or b >= num_results:
                raise ValueError(
                    'invalid measurement result slot or reserved operand')
            if b in initialized:
                raise ValueError('persistent result slot already initialized')
            initialized.add(b)
        elif code in (26, 27, 30):
            mask_id = b if code == 26 else c
            if mask_id >= len(parity_masks) or (code == 26 and c != 0):
                raise ValueError('invalid parity mask or reserved operand')
            if not set(parity_masks[mask_id]) <= initialized:
                raise ValueError(
                    'parity result slots must be initialized before use')
            if code in (27, 30) and a == b:
                raise ValueError('control and target must be distinct')


# SELECT-COPY Restore: both controls are ladder lines, including a borrowed
# bank line. Only the fresh scratch target is measured/reset.
_OP_CCX_LADDER_LADDER = 28  # ccx(ladder[a], ladder[b], ladder[c])
_OP_MEAS_UNCOMPUTE_LADDER = 29  # measure/reset ladder[a]; CZ(ladder[b], ladder[c])
_OP_DESCRIPTIONS[
    _OP_CCX_LADDER_LADDER] = 'ccx(ladder[{a}], ladder[{b}] -> ladder[{c}])  # Toffoli'
_OP_DESCRIPTIONS[_OP_MEAS_UNCOMPUTE_LADDER] = (
    'h/mz/reset(ladder[{a}]); outcome CZ(ladder[{b}], ladder[{c}])')

# SELECT-COPY uses the same pure schema and later the same device interpreter.
_OP_PARITY_CX_LADDER_LADDER = 30  # cx(ladder[a], ladder[b]) if parity(mask[c])
_OP_CZ_LADDER_LADDER = 31  # cz(ladder[a], ladder[b]); c is reserved zero
_SELECT_COPY_OPS = frozenset({21, 28, 29, 30, 31})
_FEEDBACK_OPS = _FEEDBACK_OPS | {30}
_TOFFOLI_OPCODES = _TOFFOLI_OPCODES + (21, 28)
_MEASURED_OPCODES = _MEASURED_OPCODES + (29, )
_LADDER_OPERANDS.update({28: (0, 1, 2), 29: (0, 1, 2), 30: (0, 1), 31: (0, 1)})
_OP_DESCRIPTIONS.update({
    21: 'ccx(ladder[{a}], ladder[{b}] -> target[{c}])  # Toffoli',
    30: 'cx(ladder[{a}] -> ladder[{b}]) if parity(mask[{c}])',
    31: 'cz(ladder[{a}], ladder[{b}])',
})
