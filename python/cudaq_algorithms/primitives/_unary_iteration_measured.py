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
value. CUDA-Q 0.16's compiler rejects measurement-feedback kernels, direct or
nested, under ``cudaq.sample``; this package adds no check of its own, and
older releases accepted nested feedback while returning samples that are not
the logical lookup. Automatic ``cudaq.control`` and ``cudaq.adjoint`` are
outside the contract; request ``controlled=True`` and use ``kernel_adj``
explicitly. The tests exercise CUDA-Q 0.16 on qpp-cpu. See the measured-QROM
guide for runnable examples.
"""

from __future__ import annotations

from ._unary_iteration_ir import (_emit_body_gates, _emit_measured_walk,
                                  _measured_walk_toffoli_count, infer_num_work,
                                  materialize_bodies, validate_unary_inputs)

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ._unary_iteration import (
    _BODY_OPCODES, _EXTENDED_BODY_GATES, _OP_CCX, _OP_CCX_CTRL,
    _OP_CX_ADDR_LADDER, _OP_CX_CTRL_LADDER, _OP_CX_LADDER_LADDER,
    _OP_DESCRIPTIONS, _OP_MEAS_UNCOMPUTE, _OP_MEAS_UNCOMPUTE_CTRL, _OP_X_ADDR,
    _OP_X_LADDER, _TOFFOLI_OPCODES, _WORK_OPERANDS, _mint_interpreter)

__all__ = ["MeasuredUnaryIterationKernels", "measured_unary_iteration_kernels"]

# Measured-uncompute opcodes; defined next to the coherent table so the two
# dispatch sets can never collide.
_MEASURED_OPCODES = (_OP_MEAS_UNCOMPUTE, _OP_MEAS_UNCOMPUTE_CTRL)


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
    ops: Any = ()

    @property
    def tree_and_count(self) -> int:
        """Temporary clean-tree AND computes, counted from the tape (opcodes
        ``_OP_CCX`` and ``_OP_CCX_CTRL``) independently of
        ``num_measurements``; body Toffolis are excluded."""
        return sum(1 for op in self.ops if op[0] in (_OP_CCX, _OP_CCX_CTRL))

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
            _OP_DESCRIPTIONS[op].format(a=a, b=b, c=c)
            for op, a, b, c in self.ops
        ]
        return "\n".join([header] + lines)


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
        infers the width from the body's work usage. An explicit value
        must be at least that usage, and any positive value adds the
        ``work`` view to the kernel signature even when the body never
        touches it.
    """
    num_address_bits, num_items = validate_unary_inputs(
        num_address_bits, num_items)
    bodies = materialize_bodies(body, num_items)
    ops = _emit_measured_walk(num_address_bits,
                              num_items,
                              bool(controlled),
                              bodies,
                              reverse=False)
    num_work = infer_num_work(ops, num_work)

    toffolis = sum(1 for op in ops if op[0] in _TOFFOLI_OPCODES)
    measurements = sum(1 for op in ops if op[0] in _MEASURED_OPCODES)
    kernel = _mint_interpreter(ops, bool(controlled), num_work > 0)
    kernel_adj = None
    if include_adjoint:
        adj_ops = _emit_measured_walk(num_address_bits,
                                      num_items,
                                      bool(controlled),
                                      bodies,
                                      reverse=True)
        kernel_adj = _mint_interpreter(adj_ops, bool(controlled), num_work > 0)
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
