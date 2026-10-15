# Copyright (c) 2026 NVIDIA Corporation & Affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public factories for explicitly delimited retained-bank QROM epochs."""
from collections.abc import Iterable

from ._qrom_retained_plan import difference_tables, select_retained_qrom
from ._unary_iteration import _mint_interpreter

__all__ = ["RetainedQROMChain"]


class RetainedQROMChain:
    """Share a bank and offset cache across ``m+1`` difference lookups.

    Invoke ``begin_kernel``, every member of ``kernels()`` in order, then
    ``end_kernel``. All take ``(address, ladder, output)``, with a leading
    one-qubit control register when ``controlled=True``. The ladder consists
    of ``num_clean`` initially zero qubits followed by ``num_dirty`` borrowed
    qubits; both are restored at the end of the complete epoch.

    While an epoch is live, address and control must remain unchanged and
    ladder must remain untouched. Operations on output and other data may be
    conditioned on address or control. End the epoch before changing either.
    These are quantum register preconditions, not host lifecycle checks.
    Arbitrary output operations are legal; returning output to zero additionally
    requires initially zero output and interleavings preserving the table word.
    """

    def __init__(self,
                 tables: Iterable[Iterable[int]],
                 address_bits: int,
                 output_bits: int,
                 *,
                 block_size: int | None = None,
                 controlled: bool = False,
                 dirty_banks: bool = True,
                 max_ancillas: int | None = None):
        differences = difference_tables(tables, address_bits, output_bits)
        selection = select_retained_qrom(differences,
                                         address_bits,
                                         output_bits,
                                         block_size=block_size,
                                         controlled=controlled,
                                         dirty_banks=dirty_banks,
                                         max_ancillas=max_ancillas)
        self._plan = selection.plan
        self._block_size = selection.block_size
        self._controlled = controlled
        self._dirty_banks = dirty_banks
        self._num_tables = len(differences) - 1

        def mint(plan):
            return _mint_interpreter(plan.ops,
                                     controlled,
                                     False,
                                     num_results=plan.num_results,
                                     parity_masks=plan.parity_masks)

        self._begin_kernel = mint(self._plan.begin)
        self._kernels = tuple(mint(step) for step in self._plan.steps)
        self._end_kernel = mint(self._plan.end)

    @property
    def plan(self):
        return self._plan

    @property
    def resources(self):
        return self._plan.resources

    @property
    def num_address(self):
        return self._plan.num_address

    @property
    def num_output(self):
        return self._plan.num_output

    @property
    def num_cache(self):
        return self._plan.num_cache

    @property
    def num_clean(self):
        return self.resources.num_clean

    @property
    def num_dirty(self):
        return self.resources.num_dirty

    @property
    def num_ladder(self):
        return self.resources.num_ladder

    @property
    def num_tables(self):
        return self._num_tables

    @property
    def block_size(self):
        return self._block_size

    @property
    def controlled(self):
        return self._controlled

    @property
    def dirty_banks(self):
        return self._dirty_banks

    @property
    def toffoli_count(self):
        return self.resources.toffoli_count

    @property
    def begin_kernel(self):
        return self._begin_kernel

    def kernels(self) -> tuple:
        return self._kernels

    @property
    def end_kernel(self):
        return self._end_kernel
