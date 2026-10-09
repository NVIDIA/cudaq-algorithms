# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sequential coherent lookup with adjacent unload/load fusion.

``QROMChain(tables, address_bits, output_bits)`` returns ``m + 1`` device
kernels via ``kernels()`` for ``m`` tables. All use the QROM signature
``(address, ladder, output)`` and share ``num_ladder`` clean ancillas.

With ``fused=True``, the first kernel loads table zero, each interior
kernel XOR-loads the difference of consecutive tables, and the last
unloads the final table. Shorter tables are zero-padded before XOR.
With ``fused=False``, each interior kernel performs the two lookups
separately. Both modes therefore implement the same circuit when
arbitrary operations on output and other data registers are interleaved,
provided the caller leaves address and ladder unchanged.

The stronger table-value invariant requires that callers preserve the
encoded output word, for example by using its bits as controls for
rotations on a separate register. Starting with zero output, step j
then holds ``tables[j][address]`` and the final step restores zero.
If a caller changes the output (for example with a Hadamard), this
invariant and final-zero guarantee do not hold; equivalence to separate
load/operate/unload circuits still does. Phases or entanglement created
by controlled operations on other registers are preserved.

This is whole-lookup XOR fusion, inspired by Section II.C of
`arXiv:2605.20334 <https://arxiv.org/abs/2605.20334>`_. Each step restores
its own clean bank. The paper's retained dirty-bank and cache schedule
is a different construction. For equal-length tables with per-lookup
cost C, fused cost is ``(m + 1) C`` and separate cost is ``2 m C``;
the saving is positive for ``m > 1`` and ``C > 0``. Unequal lengths or
different choices are priced per step, with no fixed reduction promised.
"""

from __future__ import annotations

from collections.abc import Sequence

from ._qrom import QROM, _normalize_table
from ._unary_iteration import _mint_interpreter

__all__ = ["QROMChain"]


class QROMChain:
    """Sequential lookups of ``m`` tables over one address register.

    Parameters
    ----------
    tables
        The lookup tables, one per step: each a non-empty sequence of
        non-negative integers fitting in ``output_bits`` bits. Lengths
        may differ (each at most ``2^address_bits``); addresses beyond a
        table's length read as zero.
    address_bits, output_bits
        Shared address register width and the single shared output
        register width (every table is read into the same register, so
        one width serves the whole chain).
    fused
        ``True`` (default): ``m + 1`` lookups via difference tables
        (`arXiv:2605.20334`, Sec. II.C). ``False``: the naive
        ``2 m``-lookup reference with identical behavior.
    variant, block_size, alpha, max_ancillas
        Forwarded to every step's :class:`QROM` (default: each step
        auto-priced independently).

    See the module docstring for the step-kernel contract and the use
    pattern; ``kernels()`` returns the ``m + 1`` step kernels.
    """

    def __init__(self,
                 tables: Sequence[Sequence[int]],
                 address_bits: int,
                 output_bits: int,
                 *,
                 fused: bool = True,
                 variant: str = "auto",
                 block_size: int | None = None,
                 alpha: int | None = None,
                 max_ancillas: int | None = None) -> None:
        if isinstance(tables,
                      (str, bytes)) or not isinstance(tables, Sequence):
            raise ValueError(
                "tables must be a sequence of lookup tables (one table "
                f"per chain step), got {type(tables).__name__}")
        tables = list(tables)
        if len(tables) == 0:
            raise ValueError("tables must contain at least one table")
        cleaned = []
        for j, table in enumerate(tables):
            if isinstance(table,
                          (str, bytes)) or not isinstance(table, Sequence):
                raise ValueError(
                    f"tables[{j}] must be a sequence of integers, got "
                    f"{type(table).__name__}")
            try:
                entries, address_bits, output_bits = _normalize_table(
                    table, address_bits, output_bits)
            except (ValueError, TypeError, OverflowError) as error:
                raise ValueError(f"QROMChain table {j}: {error}") from error
            cleaned.append(entries)
        # Original tables were validated before XOR; strategy errors below
        # are annotated with the step where they arise.
        qrom_options = dict(variant=variant,
                            block_size=block_size,
                            alpha=alpha,
                            max_ancillas=max_ancillas)

        def mint(table, label):
            try:
                return QROM(table, address_bits, output_bits, **qrom_options)
            except ValueError as error:
                raise ValueError(f"QROMChain {label}: {error}") from error

        self._fused = bool(fused)
        if self._fused:
            # Load tables[0]; transition j XOR-loads the difference
            # tables[j - 1] ^ tables[j] (shorter tables zero-padded, as
            # out-of-range addresses read zero); unload tables[m - 1].
            step_tables = [cleaned[0]]
            for j in range(1, len(cleaned)):
                prev, this = cleaned[j - 1], cleaned[j]
                length = max(len(prev), len(this))
                step_tables.append([(prev[k] if k < len(prev) else 0) ^
                                    (this[k] if k < len(this) else 0)
                                    for k in range(length)])
            step_tables.append(cleaned[-1])
            self._steps = tuple(
                mint(table, f"fused step {j} of {len(step_tables)}")
                for j, table in enumerate(step_tables))
            self._kernels = tuple(step.kernel for step in self._steps)
            step_toffolis = tuple(step.toffoli_count for step in self._steps)
        else:
            self._steps = tuple(
                mint(table, f"table {j} of {len(cleaned)}")
                for j, table in enumerate(cleaned))
            # Transition j replays the previous lookup (the unload) then
            # the next one (the load) inside one flat interpreter kernel.
            kernels = [self._steps[0].kernel]
            step_toffolis = [self._steps[0].toffoli_count]
            for j in range(1, len(self._steps)):
                ops = list(self._steps[j - 1]._ops) + list(self._steps[j]._ops)
                kernels.append(
                    _mint_interpreter(ops, controlled=False, has_work=False))
                step_toffolis.append(self._steps[j - 1].toffoli_count +
                                     self._steps[j].toffoli_count)
            kernels.append(self._steps[-1].kernel)
            step_toffolis.append(self._steps[-1].toffoli_count)
            self._kernels = tuple(kernels)
            step_toffolis = tuple(step_toffolis)

        self._num_tables = len(cleaned)
        self._num_address = self._steps[0].num_address
        self._output_bits = output_bits
        self._num_ladder = max(step.num_ladder for step in self._steps)
        self._step_toffolis = step_toffolis
        self._toffoli_count = sum(step_toffolis)

    @property
    def num_tables(self) -> int:
        """Number of chained lookups ``m`` (``kernels()`` has ``m + 1``)."""
        return self._num_tables

    @property
    def num_address(self) -> int:
        return self._num_address

    @property
    def num_ladder(self) -> int:
        """Clean ancillas one shared ladder register needs (``|0>`` in and
        out around every step kernel): the maximum over the steps'
        ``QROM.num_ladder`` — a wider ladder view is safe for every step
        (unused lines are never touched)."""
        return self._num_ladder

    @property
    def num_output(self) -> int:
        return self._output_bits

    @property
    def fused(self) -> bool:
        return self._fused

    @property
    def steps(self) -> tuple:
        """The underlying ``QROM`` lookups: the ``m + 1`` difference-table
        lookups when fused, the ``m`` per-table lookups when naive."""
        return self._steps

    @property
    def step_toffoli_counts(self) -> tuple:
        """Toffolis per step kernel (length ``m + 1``, sums to
        ``toffoli_count``)."""
        return self._step_toffolis

    @property
    def toffoli_count(self) -> int:
        return self._toffoli_count

    def kernels(self) -> tuple:
        """The ``m + 1`` step kernels ``(address, ladder, output)``, in
        application order (see the module docstring's use pattern)."""
        return self._kernels

    def describe(self) -> str:
        """One block per step kernel: the step's role in the chain and
        its underlying lookup's gate listing (see ``QROM.describe``)."""
        blocks = [f"QROM chain: {self!r}"]
        if self._fused:
            roles = (["step 0: load tables[0]"] + [
                f"step {j}: fused unload tables[{j - 1}] + load tables[{j}] "
                f"(difference-table lookup)"
                for j in range(1, self._num_tables)
            ] + [
                f"step {self._num_tables}: unload tables[{self._num_tables - 1}]"
            ])
            for role, step in zip(roles, self._steps):
                blocks.append(f"-- {role}\n{step.describe()}")
        else:
            blocks.append(f"-- step 0: load tables[0]\n"
                          f"{self._steps[0].describe()}")
            for j in range(1, self._num_tables):
                blocks.append(
                    f"-- step {j}: unload tables[{j - 1}] then load "
                    f"tables[{j}] (two lookups replayed back to back)\n"
                    f"{self._steps[j - 1].describe()}\n"
                    f"{self._steps[j].describe()}")
            blocks.append(f"-- step {self._num_tables}: unload "
                          f"tables[{self._num_tables - 1}]\n"
                          f"{self._steps[-1].describe()}")
        return "\n".join(blocks)

    def __repr__(self) -> str:
        return (f"QROMChain(tables={self._num_tables}, "
                f"address_bits={self._num_address}, "
                f"output_bits={self._output_bits}, "
                f"fused={self._fused}, "
                f"ancillas={self._num_ladder}, "
                f"toffolis={self._toffoli_count})")
