# SPDX-License-Identifier: Apache-2.0
"""Shared pure table validation and register geometry."""
from dataclasses import dataclass
from collections.abc import Iterable, Sequence
from ._unary_iteration_ir import Op, _ADDRESS_OPERANDS, _OP_BODY_X, _OP_CX_LADDER_LADDER


def validate_qrom_data(data: Iterable[int], address_bits: int,
                       output_bits: int) -> tuple[tuple[int, ...], int, int]:
    if int(address_bits) != address_bits or address_bits < 1:
        raise ValueError("address_bits must be a positive integer")
    if int(output_bits) != output_bits or output_bits < 1:
        raise ValueError("output_bits must be a positive integer")
    address_bits = int(address_bits)
    output_bits = int(output_bits)
    # Materialize first: a single-pass iterable must not be consumed
    # by one check and leave the others looking at an empty iterator.
    raw = list(data)
    if len(raw) == 0:
        raise ValueError("data must be non-empty")
    if any(int(v) != v for v in raw):
        raise ValueError("data entries must be integers")
    entries = [int(v) for v in raw]
    if len(entries) > (1 << address_bits):
        raise ValueError(f"data has {len(entries)} entries but address_bits="
                         f"{address_bits} addresses only {1 << address_bits}")
    if any(v < 0 for v in entries):
        raise ValueError("data entries must be non-negative")
    if max(entries) >= (1 << output_bits):
        raise ValueError(
            f"data entry {max(entries)} does not fit in output_bits="
            f"{output_bits} bits (max representable "
            f"{(1 << output_bits) - 1})")
    return tuple(entries), address_bits, output_bits


@dataclass(frozen=True)
class BlockGeometry:
    address_bits: int
    output_bits: int
    num_entries: int
    block_size: int

    @property
    def low_bits(self) -> int:
        return self.block_size.bit_length() - 1

    @property
    def high_bits(self) -> int:
        return self.address_bits - self.low_bits

    @property
    def num_blocks(self) -> int:
        return (self.num_entries + self.block_size - 1) // self.block_size


def rebase_block_walk(ops: Sequence[Op], *, address_offset: int,
                      output_bits: int, bank_offset: int) -> tuple[Op, ...]:
    rebased = []
    for opcode, a, b, c in ops:
        if opcode == _OP_BODY_X and b >= output_bits:
            rebased.append(
                (_OP_CX_LADDER_LADDER, a, bank_offset + b - output_bits, 0))
            continue
        operands = [a, b, c]
        for position in _ADDRESS_OPERANDS.get(opcode, ()):
            operands[position] += address_offset
        rebased.append((opcode, *operands))
    return tuple(rebased)
