# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conservative precision bounds and static Givens resource accounting.

Counts describe the specified two-Pauli-string realization of each mixing
operation on the uncontrolled tape. They are algebraic compilation costs,
not backend-transpiled circuit depth, optimized gate counts, or
fault-tolerant synthesis estimates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ._plan import (GivensBasisChangeSchedule, _integer, _parameters, _real)


@dataclass(frozen=True)
class FermionBasisChangeErrorEstimate:
    """Spectral-norm bounds for a represented basis change and reflection.

    ``parameter_error_sum`` includes every tape occurrence, so a full-plan
    relative phase contributes twice. ``synthesis_error_bound`` is a supplied
    bound for the complete represented basis-change circuit, not per gate.
    ``basis_change_error_bound`` clips their sum at two. The reflection bound
    clips twice their sum at two, for conjugation of any norm-one reflection.
    No assumption about how represented angles were encoded is required.
    The bounds hold in exact arithmetic for the supplied values. They are
    evaluated in double precision, and represented parameters decoded from
    words are rounded, so they are not certified below about 1e-15 rad per
    tape occurrence.
    """

    parameter_deltas: tuple[float, ...]
    parameter_error_sum: float
    synthesis_error_bound: float
    basis_change_error_bound: float
    reflection_error_bound: float


@dataclass(frozen=True)
class FermionBasisChangeResourceEstimate:
    """Static resources for the complete tape, including zero-valued slots.

    Number phases and real mixing gates are abstract algorithm operations.
    Each mixing operation uses two elementary Pauli rotations, with four
    parity CNOTs per physical mode separation and twelve Clifford basis
    changes (eight H and four S/S-dagger). Number phases are one elementary
    rotation each and require no parity ladder.

    Counts describe the uncontrolled tape. ``cudaq.control`` and
    ``FermionBasisChange.controlled_kernel`` put the enable on every emitted
    gate: each H becomes a controlled H, each S or S-dagger a controlled S
    or S-dagger, each parity CNOT a Toffoli, each Rz a controlled Rz and
    each R1 a controlled R1 (bit-controlled rotations become doubly
    controlled). Compare QROM selection against those controlled costs,
    not these counts.

    ``abstract_mixing_layers_upper_bound`` equals the sequential mixing
    count, not a circuit depth; it does not include phase layers,
    programmable-rotation internals, or hardware scheduling. An optional
    angle width counts bit-controlled rotations and defaults to keeping
    every parameter word resident. A caller-supplied program width
    overrides that reported qubit count only; the kernels keep every word
    resident. Address, QROM lookup/unlookup, and synthesis resources are
    excluded.
    """

    num_modes: int
    num_parameters: int
    num_system_qubits: int
    num_program_qubits: int
    num_workspace_qubits: int
    num_givens_rotations: int
    num_number_phases: int
    num_elementary_rotations: int
    num_bit_controlled_rotations: int
    num_parity_cnots: int
    num_hadamards: int
    num_s_phase_gates: int
    num_clifford_basis_changes: int
    abstract_mixing_layers_upper_bound: int


def _schedule(schedule):
    if not isinstance(schedule, GivensBasisChangeSchedule):
        raise ValueError("schedule must be a GivensBasisChangeSchedule")
    return schedule


def _phase_delta(new, old):
    """Principal phase difference without subtracting or wrapping huge angles."""
    new_sin, new_cos = math.sin(new), math.cos(new)
    old_sin, old_cos = math.sin(old), math.cos(old)
    # Compute exp(i*new) * conjugate(exp(i*old)) using libm's accurate
    # argument reduction. A remainder by floating-point tau does not preserve
    # the physical phase of a large finite angle; new-old can also overflow.
    return math.atan2(new_sin * old_cos - new_cos * old_sin,
                      new_cos * old_cos + new_sin * old_sin)


def estimate_fermion_basis_change_error(schedule,
                                        represented_parameters,
                                        *,
                                        synthesis_error_bound=0.):
    """Bound spectral error from supplied represented real angle values.

    A real Givens generator and an exact number-phase generator each have
    spectral norm one, hence a gate perturbation is bounded by the absolute
    angle difference. Telescoping unitary products sums those differences.
    This remains valid on the full fermionic Fock space and under arbitrary
    physical mode embeddings. The optional synthesis bound is added once
    for the complete basis-change implementation. Conjugating a reflection
    incurs at most twice that total error.

    Original numerical schedule values are required. Parameter differences
    are shortest signed circular deltas modulo 2*pi for the complete Givens
    gates and number phases. No binary encoding convention is assumed.
    Host reconstruction discrepancies are excluded. The bounds hold in
    exact arithmetic for the supplied values; they are evaluated in double
    precision, and represented parameters decoded from words are rounded,
    so they are not certified below about 1e-15 rad per tape occurrence.
    """
    schedule = _schedule(schedule)
    if schedule.parameters is None:
        raise ValueError(
            "an error estimate requires original numerical parameters")
    represented = _parameters(represented_parameters, schedule.num_parameters)
    try:
        synthesis = _real(synthesis_error_bound, "synthesis_error_bound")
    except ValueError as exc:
        raise ValueError(
            "synthesis_error_bound must be finite and nonnegative") from exc
    if synthesis < 0:
        raise ValueError(
            "synthesis_error_bound must be finite and nonnegative")
    synthesis += 0.  # Report a -0.0 bound as +0.0.
    deltas = tuple(
        _phase_delta(new, old)
        for new, old in zip(represented, schedule.parameters))
    parameter_sum = sum(
        abs(op.scale * deltas[op.parameter_index])
        for op in schedule.operations)
    combined = parameter_sum + synthesis
    return FermionBasisChangeErrorEstimate(
        parameter_deltas=deltas,
        parameter_error_sum=parameter_sum,
        synthesis_error_bound=synthesis,
        basis_change_error_bound=min(2., combined),
        reflection_error_bound=min(2., 2. * combined))


def estimate_fermion_basis_change_resources(schedule,
                                            *,
                                            angle_bits=None,
                                            num_program_qubits=None):
    """Count static algorithm operations and physical parity-ladder costs.

    For physical endpoint separation ``d``, each of the two mixing Pauli
    strings has weight ``d+1``. A compute/uncompute parity ladder therefore
    has ``2*d`` CNOTs per string, or ``4*d`` for the mixing operation. Every
    zero angle remains present, matching a programmable consumer's tape.
    The algorithm needs no workspace beyond the system and caller-owned
    program register; internal synthesis ancillas are not estimated.

    ``angle_bits`` is an integer from 1 through 52. If given, each elementary
    rotation uses that many bit-controlled rotations; without it, their count
    is zero. ``num_program_qubits`` defaults to all parameter words resident
    (num_parameters * angle_bits), or zero without a width. An explicit count
    overrides that default; with a width it must be at least ``angle_bits``
    (one word). The override changes only this report; the kernels keep
    every word resident. QROM address, lookup and unlookup are excluded.

    Counts describe the uncontrolled tape. Under ``cudaq.control`` or
    ``FermionBasisChange.controlled_kernel`` every counted gate also takes
    the enable as a control; see :class:`FermionBasisChangeResourceEstimate`.
    """
    schedule = _schedule(schedule)
    if angle_bits is not None:
        angle_bits = _integer(angle_bits, "angle_bits")
        if not 1 <= angle_bits <= 52:
            raise ValueError("angle_bits must be between 1 and 52")
    if num_program_qubits is None:
        num_program_qubits = schedule.num_parameters * (angle_bits or 0)
    program_width = _integer(num_program_qubits, "num_program_qubits")
    if program_width < 0:
        raise ValueError("num_program_qubits must be nonnegative")
    if angle_bits is not None and program_width < angle_bits:
        raise ValueError(
            "num_program_qubits must be at least angle_bits (one word)")
    mixing = tuple(op for op in schedule.operations if op.kind == "givens")
    number_phases = sum(op.kind == "phase" for op in schedule.operations)
    mapping = schedule.structure.mode_map
    parity_cnots = 4 * sum(
        abs(mapping[op.first] - mapping[op.second]) for op in mixing)
    count = len(mixing)
    return FermionBasisChangeResourceEstimate(
        num_modes=schedule.structure.num_modes,
        num_parameters=schedule.num_parameters,
        num_system_qubits=schedule.structure.num_system_qubits,
        num_program_qubits=program_width,
        num_workspace_qubits=0,
        num_givens_rotations=count,
        num_number_phases=number_phases,
        num_elementary_rotations=2 * count + number_phases,
        num_bit_controlled_rotations=(angle_bits or 0) *
        (2 * count + number_phases),
        num_parity_cnots=parity_cnots,
        num_hadamards=8 * count,
        num_s_phase_gates=4 * count,
        num_clifford_basis_changes=12 * count,
        abstract_mixing_layers_upper_bound=count)
