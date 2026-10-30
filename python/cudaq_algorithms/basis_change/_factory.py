# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public plans for fermionic basis changes on existing quantum registers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from ._encoding import AngleEncoding
from ._kernels import classical_kernel, programmable_kernel
from ._metrics import estimate_fermion_basis_change_error
from ._plan import (_MAX_QUBIT_INDEX, GivensBasisChangeSchedule, _index,
                    _sequence, _validate_tolerance,
                    compile_givens_basis_change)


def _register_layout(structure, encoding, parameter_registers):
    width, count = encoding.width, structure.num_parameters
    system_width = structure.num_system_qubits
    if parameter_registers is None:
        # Default words obey the same index bound as explicit words.
        last = system_width + count * width - 1
        if last > _MAX_QUBIT_INDEX:
            raise ValueError(
                f"default parameter words would end at qubit {last}, beyond "
                f"the largest qubit index {_MAX_QUBIT_INDEX}")
        return tuple(
            tuple(
                range(system_width + slot * width, system_width +
                      (slot + 1) * width)) for slot in range(count))
    # Slot and bit order carry meaning, so unordered containers are rejected.
    layout = tuple(
        _sequence(word, "each parameter word")
        for word in _sequence(parameter_registers, "parameter_registers"))
    if len(layout) != count:
        raise ValueError(
            "parameter_registers must match the parameter slot count")
    # Indices below system_width form the system span; it is not
    # materialized, since it can reach 2**31 qubits.
    occupied = set()
    result = []
    for word in layout:
        if len(word) != width:
            raise ValueError(
                "each parameter word must match the encoding width")
        word = tuple(
            _index(index, "parameter register index") for index in word)
        if len(set(word)) != width:
            raise ValueError("each parameter word must use distinct indices")
        if any(index < system_width or index in occupied for index in word):
            raise ValueError(
                "parameter words must not overlap each other or the system register"
            )
        occupied.update(word)
        result.append(word)
    return tuple(result)


@dataclass(frozen=True)
class FermionBasisChange:
    """Immutable plan acting on an existing system and optional angle words.

    Construct plans with :func:`fermion_basis_change`. Kernels accept one
    qview spanning at least ``num_qubits`` qubits (``num_qubits + 1`` for
    :meth:`controlled_kernel`). CUDA-Q does not bounds-check views, so a
    narrower view silently addresses the following qubits of the parent
    allocation. System modes follow ``structure.mode_map``; optional words
    follow ``parameter_registers``. The caller owns their preparation and
    uncomputation. No QROM or workspace is allocated here.

    :meth:`controlled_kernel` puts the enable only on the Rz and R1
    rotations; ``cudaq.control`` of :meth:`kernel` puts it on every emitted
    gate, including the Clifford and parity scaffold. The ``adjoint`` flag of
    :meth:`controlled_kernel` must be a Python ``bool``.
    """

    #: Compiled host schedule with its structure and optional angle values.
    schedule: GivensBasisChangeSchedule
    #: Word encoding of a programmable plan, or None for a classical plan.
    angle_encoding: AngleEncoding | None = None
    #: Qubit indices of each parameter word in slot order; empty if classical.
    parameter_registers: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self):
        if not isinstance(self.schedule, GivensBasisChangeSchedule):
            raise ValueError("schedule must be a GivensBasisChangeSchedule")
        if self.angle_encoding is None:
            if self.schedule.parameters is None:
                raise ValueError(
                    "classical basis changes require numerical parameters")
            if self.parameter_registers is not None and self.parameter_registers != (
            ):
                raise ValueError(
                    "parameter_registers require an angle encoding")
            object.__setattr__(self, "parameter_registers", ())
        else:
            if not isinstance(self.angle_encoding, AngleEncoding):
                raise ValueError("angle_encoding must be an AngleEncoding")
            object.__setattr__(
                self, "parameter_registers",
                _register_layout(self.structure, self.angle_encoding,
                                 self.parameter_registers))

    @property
    def structure(self):
        """Static Givens structure of the schedule, including ``mode_map``."""
        return self.schedule.structure

    @property
    def is_programmable(self):
        """True when caller-owned quantum words supply the executed angles."""
        return self.angle_encoding is not None

    @property
    def angle_bits(self):
        """Bits per parameter word, or None for a classical plan."""
        return None if self.angle_encoding is None else self.angle_encoding.width

    @property
    def num_qubits(self):
        """Combined register span, excluding lookup and external controls."""
        return max(
            (index + 1 for word in self.parameter_registers for index in word),
            default=self.structure.num_system_qubits)

    @cached_property
    def encoded_words(self):
        """Host integer codes in parameter order, or None without a reference."""
        if not self.is_programmable or self.schedule.parameters is None:
            return None
        return tuple(
            self.angle_encoding.encode(value)
            for value in self.schedule.parameters)

    @cached_property
    def represented_parameters(self):
        """Decoded radian values of encoded_words, or None when unavailable."""
        if self.encoded_words is None:
            return None
        return tuple(
            self.angle_encoding.decode(code) for code in self.encoded_words)

    @cached_property
    def quantization_error(self):
        """Circular parameter-error report; excludes host and synthesis error.

        Bounds hold in exact arithmetic for the supplied values; they are
        evaluated in double precision from rounded decodes, so they are not
        certified below about 1e-15 rad per tape occurrence.
        """
        if self.represented_parameters is None:
            return None
        return estimate_fermion_basis_change_error(self.schedule,
                                                   self.represented_parameters)

    @property
    def quantization_error_bound(self):
        """Conservative Fock-space bound from ``quantization_error``, or None."""
        if self.quantization_error is None:
            return None
        return self.quantization_error.basis_change_error_bound

    def _build(self, adjoint, controlled):
        if self.is_programmable:
            return programmable_kernel(self.schedule,
                                       self.parameter_registers,
                                       self.angle_encoding.weights,
                                       adjoint=adjoint,
                                       controlled=controlled)
        return classical_kernel(self.schedule,
                                adjoint=adjoint,
                                controlled=controlled)

    @cached_property
    def _forward(self):
        return self._build(False, False)

    @cached_property
    def _inverse(self):
        return self._build(True, False)

    @cached_property
    def _controlled_forward(self):
        return self._build(False, True)

    @cached_property
    def _controlled_inverse(self):
        return self._build(True, True)

    def kernel(self):
        """Return a kernel applying the transformation to a combined qview."""
        return self._forward

    def adjoint_kernel(self):
        """Return the inverse, including all exact mode phases."""
        return self._inverse

    def controlled_kernel(self, *, adjoint=False):
        """Return a kernel with enable at index zero and the base register after it.

        Allocate ``num_qubits + 1`` qubits. Every base index shifts by one;
        the inactive enable branch is exact identity on all operands. The
        enable controls only the Rz and R1 rotations, since each Clifford
        and parity scaffold cancels around its rotation. ``adjoint`` must be
        a Python ``bool``.
        """
        if not isinstance(adjoint, bool):
            raise ValueError("adjoint must be a boolean")
        return self._controlled_inverse if adjoint else self._controlled_forward

    def __reduce__(self):
        # Reconstruct only immutable host data, never cached runtime kernels.
        return (type(self), (self.schedule, self.angle_encoding,
                             self.parameter_registers))


def fermion_basis_change(basis=None,
                         *,
                         structure=None,
                         programmable=False,
                         angle_bits=None,
                         parameter_registers=None,
                         tolerance=1e-12):
    """Plan a full or selected-column fermionic transformation.

    ``basis`` is a unitary square matrix or normalized vector/column of
    at least two modes, or a :class:`GivensBasisChangeSchedule`. Columns
    are transformed modes; the operation preserves vacuum and all
    required determinant phases.
    It acts on an existing state and does not initialize occupations.
    Each index labels one fermionic mode on its own qubit, so a ``2**n`` by
    ``2**n`` qubit unitary is interpreted as ``2**n`` modes on ``2**n``
    qubits; this is not arbitrary qubit-unitary synthesis.

    ``programmable`` must be a Python ``bool``. With ``programmable=True``,
    require ``angle_bits`` from 1 through 52.
    Every slot consumes an unsigned little-endian quantum word encoding
    a full-turn angle. Supplied numerical values provide host reference
    data and encoded table entries; the loaded words determine execution.
    Structure-only programmable plans have no host numerical reference.

    Words default to disjoint contiguous slices after the physical system
    span, including spectators. Explicit words are any distinct qubit
    indices outside the system span, listed least-significant bit first,
    that do not overlap other words. Plan kernels need a view spanning at
    least ``num_qubits`` qubits (``num_qubits + 1`` for
    ``controlled_kernel``); CUDA-Q does not bounds-check views, so a
    narrower view silently addresses the following qubits of the parent
    allocation. ``controlled_kernel`` controls only the Rz and R1
    rotations; ``cudaq.control`` of ``kernel()`` controls every emitted
    gate. The caller composes QROM or arithmetic explicitly.

    ``tolerance`` (``0 < tolerance < 1``) validates input unitarity or
    normalization; it is not a quantization or synthesis budget. Inexact
    input within tolerance compiles to the unitary its angles define, so
    check ``schedule.reconstruction_error`` when loosening it.
    Reconstruction and quantization diagnostics are exposed separately by
    the returned plan.
    """
    tolerance = _validate_tolerance(tolerance)
    if not isinstance(programmable, bool):
        raise ValueError("programmable must be a boolean")
    if not programmable and (angle_bits is not None
                             or parameter_registers is not None):
        raise ValueError(
            "angle_bits and parameter_registers require programmable=True")
    encoding = AngleEncoding(angle_bits) if programmable else None
    if isinstance(basis, GivensBasisChangeSchedule):
        if structure is not None and structure != basis.structure:
            raise ValueError("structure must match the supplied schedule")
        schedule = basis
    else:
        schedule = compile_givens_basis_change(basis,
                                               structure=structure,
                                               tolerance=tolerance)
    return FermionBasisChange(schedule, encoding, parameter_registers)
