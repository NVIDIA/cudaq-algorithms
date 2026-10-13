# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sequential coherent lookup with adjacent unload/load fusion.

``QROMChain(tables, address_bits, output_bits)`` returns ``m + 1`` device
kernels via ``kernels()`` for ``m`` tables. All use the QROM signature
``(address, ladder, output)`` and share ``num_ladder`` clean ancillas.

With ``fused=True``, the first kernel loads table zero, each interior
kernel XOR-loads the difference of consecutive tables, and the last
unloads the final table. Shorter tables are zero-padded before XOR.
With ``fused=False``, each interior kernel calls the two step lookups
back to back. Both modes implement the same unitary on the clean-ladder
sector whatever the caller interleaves on the output and other data
registers between steps; the ladder must be clean at each step kernel.

The stronger table-value invariant, in which the output holds
``tables[j][address]`` after step ``j`` and returns to zero after the
last step, additionally needs the address unchanged and the encoded
word preserved between steps (for example used only as controls on a
separate register). An operation that changes the word, such as a
Hadamard on an output bit, or a change of address between steps removes
that invariant, while the equivalence to separate load/operate/unload
circuits still holds. Phases or entanglement created by controlled
operations on other registers are preserved.

This is whole-lookup XOR fusion, inspired by Section II.C of
`arXiv:2605.20334 <https://arxiv.org/abs/2605.20334>`_. Each step restores
its own clean bank. The paper's retained dirty-bank and cache schedule
is a different construction. For equal-length tables with per-lookup
cost C, fused cost is ``(m + 1) C`` and separate cost is ``2 m C``;
the saving is positive for ``m > 1`` and ``C > 0``. Unequal lengths or
different choices are priced per step, with no fixed reduction promised.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence

import cudaq

from ._qrom import QROM, _normalize_table
from ._unary_iteration import _retain

__all__ = ["QROMChain"]

_NOT_A_TABLE = (str, bytes, bytearray, memoryview)


def _pair_kernel(first, second):
    """Mint ``(address, ladder, output) -> first(...); second(...)``.

    The naive chain replays the previous lookup (the unload) and the next
    one (the load) back to back. Calling the two already-minted step
    kernels, instead of minting a third interpreter over their
    concatenated tapes, lets CUDA-Q share each step's tape between its two
    call sites in the caller's circuit; the concatenated form roughly
    doubled first-execution compile time and QIR size for 8k-op steps.
    """

    @cudaq.kernel
    def primitives_qrom_chain_pair(address: cudaq.qview, ladder: cudaq.qview,
                                   output: cudaq.qview):
        first(address, ladder, output)
        second(address, ladder, output)

    _retain(primitives_qrom_chain_pair)
    return primitives_qrom_chain_pair


class QROMChain:
    """Sequential lookups of ``m`` tables over one address register.

    Parameters
    ----------
    tables
        The lookup tables, one per step: each a non-empty iterable (list,
        tuple or numpy array) of non-negative integers fitting in
        ``output_bits`` bits. Lengths may differ (each at most
        ``2^address_bits``); addresses beyond a table's length read as
        zero.
    address_bits, output_bits
        Shared address register width and the single shared output
        register width (every table is read into the same register, so
        one width serves the whole chain).
    fused
        ``True`` (default): ``m + 1`` lookups via difference tables
        (`arXiv:2605.20334`, Sec. II.C). ``False``: the naive
        ``2 m``-lookup reference (pair kernels over the per-table
        lookups) with identical behavior.
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
                      _NOT_A_TABLE) or not isinstance(tables, Iterable):
            raise ValueError(
                "tables must be a sequence of lookup tables (one table "
                f"per chain step), got {type(tables).__name__}")
        tables = list(tables)
        if len(tables) == 0:
            raise ValueError("tables must contain at least one table")
        if not isinstance(fused, bool):
            raise ValueError("fused must be a bool (pass bool(flag) for "
                             "numpy booleans)")
        # The register widths are chain-wide, not table properties:
        # validate them once, before any table is read.
        if (isinstance(address_bits, bool) or int(address_bits) != address_bits
                or address_bits < 1):
            raise ValueError("address_bits must be a positive integer")
        if (isinstance(output_bits, bool) or int(output_bits) != output_bits
                or output_bits < 1):
            raise ValueError("output_bits must be a positive integer")
        address_bits = int(address_bits)
        output_bits = int(output_bits)
        cleaned = []
        for j, table in enumerate(tables):
            if isinstance(table,
                          _NOT_A_TABLE) or not isinstance(table, Iterable):
                raise ValueError(
                    f"tables[{j}] must be a sequence of integers, got "
                    f"{type(table).__name__}")
            try:
                entries, _, _ = _normalize_table(table, address_bits,
                                                 output_bits)
            except (ValueError, TypeError, OverflowError) as error:
                raise ValueError(f"QROMChain table {j}: {error}") from error
            cleaned.append(entries)
        qrom_options = dict(variant=variant,
                            block_size=block_size,
                            alpha=alpha,
                            max_ancillas=max_ancillas)

        def mint(table):
            # Strategy errors do not depend on the table (the options are
            # shared and the budget filter ignores entries), so they carry
            # a chain-level prefix rather than a step index.
            try:
                return QROM(table, address_bits, output_bits, **qrom_options)
            except (ValueError, TypeError, OverflowError) as error:
                raise ValueError(f"QROMChain: {error}") from error

        self._fused = fused
        if fused:
            # Load tables[0]; transition j XOR-loads the difference
            # tables[j - 1] ^ tables[j] (shorter tables zero-padded, as
            # out-of-range addresses read zero); unload tables[m - 1].
            step_tables = [cleaned[0]]
            for j in range(1, len(cleaned)):
                step_tables.append([
                    p ^ t for p, t in itertools.zip_longest(
                        cleaned[j - 1], cleaned[j], fillvalue=0)
                ])
            step_tables.append(cleaned[-1])
            first = mint(step_tables[0])
            if len(cleaned) == 1:
                # One table loads and unloads with the same lookup.
                self._steps = (first, first)
            else:
                self._steps = (first, ) + tuple(
                    mint(table) for table in step_tables[1:])
            self._kernels = tuple(step.kernel for step in self._steps)
            step_toffolis = tuple(step.toffoli_count for step in self._steps)
        else:
            self._steps = tuple(mint(table) for table in cleaned)
            # Transition j replays the previous lookup (the unload) then
            # the next one (the load) through a pair kernel.
            kernels = [self._steps[0].kernel]
            step_toffolis = [self._steps[0].toffoli_count]
            for j in range(1, len(self._steps)):
                kernels.append(
                    _pair_kernel(self._steps[j - 1].kernel,
                                 self._steps[j].kernel))
                step_toffolis.append(self._steps[j - 1].toffoli_count +
                                     self._steps[j].toffoli_count)
            kernels.append(self._steps[-1].kernel)
            step_toffolis.append(self._steps[-1].toffoli_count)
            self._kernels = tuple(kernels)
            step_toffolis = tuple(step_toffolis)

        self._num_tables = len(cleaned)
        self._num_address = address_bits
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
