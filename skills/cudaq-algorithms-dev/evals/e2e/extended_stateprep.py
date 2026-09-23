# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Completion cases for direct state-preparation device contracts.

The controller-side expectations use only NumPy/SciPy: dense occupation-basis
ladder algebra, dense Pauli matrices, analytic Givens maps, and determinant
minors.  They never import CUDA-Q or the evaluated package.
"""
from __future__ import annotations

from itertools import combinations
import textwrap

import numpy as np
from scipy.linalg import expm


def _pauli_matrix(word: str) -> np.ndarray:
    matrices = {
        "I": np.eye(2),
        "X": np.array([[0, 1], [1, 0]], complex),
        "Y": np.array([[0, -1j], [1j, 0]], complex),
        "Z": np.diag([1, -1]),
    }
    result = np.ones((1, 1), complex)
    for letter in reversed(word):
        result = np.kron(result, matrices[letter])
    return result


def _annihilators(num_modes: int,
                  *,
                  fermionic: bool = True) -> list[np.ndarray]:
    operators = []
    for mode in range(num_modes):
        operator = np.zeros((1 << num_modes, 1 << num_modes), complex)
        for occupation in range(1 << num_modes):
            if occupation & (1 << mode):
                parity = ((occupation &
                           ((1 << mode) - 1)).bit_count() if fermionic else 0)
                operator[occupation ^ (1 << mode), occupation] = (-1)**parity
        operators.append(operator)
    return operators


def _basis_state(num_qubits: int, occupied: list[int]) -> np.ndarray:
    state = np.zeros(1 << num_qubits, complex)
    state[sum(1 << orbital for orbital in occupied)] = 1.0
    return state


def _particle_number(state: np.ndarray) -> float:
    return float(
        sum(index.bit_count() * abs(amplitude)**2
            for index, amplitude in enumerate(state)))


def _single_generator(num_qubits: int, low: int, high: int) -> np.ndarray:
    annihilators = _annihilators(num_qubits)
    transfer = annihilators[high].conj().T @ annihilators[low]
    return -1j * (transfer - transfer.conj().T)


def _double_generator(num_qubits: int, source: tuple[int, int],
                      target: tuple[int, int]) -> np.ndarray:
    annihilators = _annihilators(num_qubits)
    low, high = source
    low_virtual, high_virtual = target
    transfer = (annihilators[low_virtual].conj().T
                @ annihilators[high_virtual].conj().T @ annihilators[high]
                @ annihilators[low])
    return 1j * (transfer - transfer.conj().T)


def _raw_double_state(num_qubits: int, occupied: list[int], theta: float,
                      p: int, q: int, r: int, s: int) -> np.ndarray:
    source = (min(p, q), max(p, q))
    target = (min(r, s), max(r, s))
    effective_theta = (-theta if ((p > q) != (r > s)) else theta)
    generator = _double_generator(num_qubits, source, target)
    return expm(-0.5j * effective_theta * generator) @ _basis_state(
        num_qubits, occupied)


def _pool_generators(kind: str, num_qubits: int) -> list[np.ndarray]:
    if num_qubits != 4:
        raise ValueError("completion pool oracle is intentionally four-qubit")

    if kind == "uccgsd":
        return [
            _single_generator(4, q, p) for p in range(1, 4) for q in range(p)
        ] + [
            _double_generator(4, (0, 1), (2, 3)),
            _double_generator(4, (0, 2), (1, 3)),
            _double_generator(4, (1, 2), (0, 3)),
        ]
    if kind == "upccgsd":
        return [
            _single_generator(4, 0, 2),
            _single_generator(4, 1, 3),
            _double_generator(4, (2, 3), (0, 1)),
        ]
    if kind == "ceo":
        lowering = _annihilators(4, fermionic=False)
        alpha = lowering[2].conj().T @ lowering[0]
        beta = lowering[3].conj().T @ lowering[1]
        return [
            -1j * (alpha - alpha.conj().T),
            -1j * (beta - beta.conj().T),
            1j * (alpha + alpha.conj().T) @ (beta - beta.conj().T),
            1j * (alpha - alpha.conj().T) @ (beta + beta.conj().T),
        ]
    raise ValueError(f"unknown grouped ansatz {kind!r}")


def _ordered_generator_product(state: np.ndarray, amplitudes: list[float],
                               generators: list[np.ndarray]) -> np.ndarray:
    result = np.asarray(state, complex)
    for theta, generator in zip(amplitudes, generators):
        result = expm(1j * float(theta) * generator) @ result
    return result


def _fixed_product(state: np.ndarray, amplitudes: list[float],
                   word_groups: list[list[str]],
                   coefficient_groups: list[list[float]]) -> np.ndarray:
    result = np.asarray(state, complex)
    for theta, words, coefficients in zip(amplitudes, word_groups,
                                          coefficient_groups):
        for word, coefficient in zip(words, coefficients):
            result = expm(1j * float(theta) * float(coefficient) *
                          _pauli_matrix(word)) @ result
    return result


def _slater_state(orbitals) -> np.ndarray:
    matrix = np.asarray(orbitals, dtype=complex)
    num_modes, num_electrons = matrix.shape
    state = np.zeros(1 << num_modes, complex)
    for occupied in combinations(range(num_modes), num_electrons):
        state[sum(1 << orbital for orbital in occupied)] = np.linalg.det(
            matrix[list(occupied), :])
    return state


def _real_givens_count(orbitals, tolerance: float = 1e-12) -> int:
    """Independent elimination count for a real orthonormal matrix."""
    work = np.asarray(orbitals, dtype=float).copy()
    num_modes, num_electrons = work.shape
    count = 0
    for column in range(num_electrons):
        for row in range(num_modes - 1, column, -1):
            lower = work[row, column]
            if abs(lower) <= tolerance:
                continue
            upper = work[row - 1, column]
            radius = np.hypot(upper, lower)
            cosine, sine = upper / radius, lower / radius
            old_upper, old_lower = work[row - 1].copy(), work[row].copy()
            work[row - 1] = cosine * old_upper + sine * old_lower
            work[row] = -sine * old_upper + cosine * old_lower
            count += 1
    return count


def _reference_parameters(variant: int) -> dict:
    if variant == 0:
        return {
            "num_qubits": 4,
            "num_electrons": 2,
            "spin": 0,
            "explicit_occupation": [0, 3],
            "single": {
                "theta": 0.43,
                "occupied": [0, 1],
                "p": 0,
                "q": 3,
            },
            "double": {
                "theta": 0.61,
                "occupied": [0, 1],
                "p": 0,
                "q": 1,
                "r": 2,
                "s": 3,
            },
        }
    return {
        "num_qubits": 4,
        "num_electrons": 2,
        "spin": 2,
        "explicit_occupation": [0, 2],
        "single": {
            "theta": -0.57,
            "occupied": [1, 3],
            "p": 1,
            "q": 2,
        },
        "double": {
            "theta": -0.49,
            "occupied": [0, 2],
            "p": 2,
            "q": 0,
            "r": 1,
            "s": 3,
        },
    }


def _grouped_parameters(variant: int) -> dict:
    rng = np.random.default_rng(7711 + variant)
    return {
        "num_qubits": 4,
        "num_spatial_orbitals": 2,
        "occupied": [0, 1],
        "uccgsd_amplitudes": (0.24 * rng.normal(size=9)).tolist(),
        "upccgsd_amplitudes": (0.29 * rng.normal(size=3)).tolist(),
        "ceo_amplitudes": (0.21 * rng.normal(size=4)).tolist(),
        "fixed_amplitudes": ([0.33, -0.27] if variant == 0 else [-0.41, 0.19]),
        "fixed_words": [["IYXI", "IXYI"], ["YZZX", "XZZY"]],
        "fixed_coefficients": [[0.5, -0.5], [0.5, -0.5]],
    }


def _givens_parameters(variant: int) -> dict:
    rng = np.random.default_rng(8219 + variant)
    raw = rng.normal(size=(4, 2))
    orbitals, _ = np.linalg.qr(raw)
    if variant == 0:
        rotation = (0.37, 0, 1)
        phased = (-0.41, 0.73, 1, 2)
    else:
        rotation = (-0.52, 2, 1)
        phased = (0.46, -0.67, 1, 0)
    return {
        "num_qubits": 4,
        "rotation_theta": rotation[0],
        "rotation_first": rotation[1],
        "rotation_second": rotation[2],
        "phase_theta": phased[0],
        "phase": phased[1],
        "phase_first": phased[2],
        "phase_second": phased[3],
        "orbitals": orbitals.tolist(),
    }


def parameters(case_id: str, variant: int) -> dict:
    if variant not in (0, 1):
        raise ValueError("variant must be 0 (public) or 1 (held out)")
    if case_id == "reference_excitation_primitives":
        return _reference_parameters(variant)
    if case_id == "grouped_ucc_device":
        return _grouped_parameters(variant)
    if case_id == "givens_raw_device":
        return _givens_parameters(variant)
    raise KeyError(case_id)


def expected(case_id: str, params: dict) -> dict[str, object]:
    p = params
    if case_id == "reference_excitation_primitives":
        n = int(p["num_qubits"])
        canonical = _basis_state(n, list(range(int(p["num_electrons"]))))
        explicit = _basis_state(n, p["explicit_occupation"])
        single = p["single"]
        single_state = expm(-0.5j * float(single["theta"]) * _single_generator(
            n, int(single["p"]), int(single["q"]))) @ _basis_state(
                n, single["occupied"])
        double = p["double"]
        double_state = _raw_double_state(n, double["occupied"],
                                         float(double["theta"]),
                                         int(double["p"]), int(double["q"]),
                                         int(double["r"]), int(double["s"]))
        return {
            "canonical_state": canonical,
            "explicit_state": explicit,
            "single_state": single_state,
            "double_state": double_state,
            "canonical_particle_number": _particle_number(canonical),
            "explicit_particle_number": _particle_number(explicit),
            "single_particle_number": _particle_number(single_state),
            "double_particle_number": _particle_number(double_state),
            "canonical_num_qubits": n,
            "canonical_num_electrons": int(p["num_electrons"]),
            "canonical_num_x_gates": int(p["num_electrons"]),
            "explicit_num_qubits": n,
            "explicit_num_electrons": len(p["explicit_occupation"]),
            "explicit_num_x_gates": len(p["explicit_occupation"]),
        }
    if case_id == "grouped_ucc_device":
        n = int(p["num_qubits"])
        initial = _basis_state(n, p["occupied"])
        uccgsd = _ordered_generator_product(initial, p["uccgsd_amplitudes"],
                                            _pool_generators("uccgsd", n))
        upccgsd = _ordered_generator_product(initial, p["upccgsd_amplitudes"],
                                             _pool_generators("upccgsd", n))
        ceo = _ordered_generator_product(initial, p["ceo_amplitudes"],
                                         _pool_generators("ceo", n))
        fixed = _fixed_product(initial, p["fixed_amplitudes"],
                               p["fixed_words"], p["fixed_coefficients"])
        sizes = [len(group) for group in p["fixed_words"]]
        return {
            "uccgsd_state": uccgsd,
            "upccgsd_state": upccgsd,
            "ceo_state": ceo,
            "fixed_state": fixed,
            "uccgsd_particle_number": _particle_number(uccgsd),
            "upccgsd_particle_number": _particle_number(upccgsd),
            "ceo_particle_number": _particle_number(ceo),
            "fixed_particle_number": _particle_number(fixed),
            "fixed_num_qubits": n,
            "fixed_num_excitations": len(sizes),
            "fixed_num_pauli_rotations": sum(sizes),
            "fixed_max_pauli_rotations_per_excitation": max(sizes, default=0),
        }
    if case_id == "givens_raw_device":
        n = int(p["num_qubits"])
        theta = float(p["rotation_theta"])
        first, second = int(p["rotation_first"]), int(p["rotation_second"])
        givens = np.zeros(1 << n, complex)
        givens[1 << first] = np.cos(theta)
        givens[1 << second] = np.sin(theta)

        phase_theta, phase = float(p["phase_theta"]), float(p["phase"])
        phase_first = int(p["phase_first"])
        phase_second = int(p["phase_second"])
        phase_givens = np.zeros(1 << n, complex)
        phase_givens[1 << phase_first] = (np.exp(-0.5j * phase) *
                                          np.cos(phase_theta))
        phase_givens[1 << phase_second] = (np.exp(0.5j * phase) *
                                           np.sin(phase_theta))

        slater = _slater_state(p["orbitals"])
        rotations = _real_givens_count(p["orbitals"])
        electrons = len(p["orbitals"][0])
        return {
            "givens_state": givens,
            "phase_givens_state": phase_givens,
            "slater_state": slater,
            "slater_norm": float(np.linalg.norm(slater)),
            "slater_particle_number": _particle_number(slater),
            "num_spin_orbitals": n,
            "num_electrons": electrons,
            "num_givens_rotations": rotations,
            "num_exp_pauli_calls": 2 * rotations,
            "num_phase_rotations": 0,
            "two_qubit_gate_count_proxy": 2 * rotations,
            "depth_proxy": 2 * rotations,
        }
    raise KeyError(case_id)


def case_specs() -> list[dict]:
    reference_outputs = {
        "canonical_state": "state prepared by raw contiguous Hartree-Fock",
        "explicit_state": "state prepared by raw explicit occupation",
        "single_state": "state after one raw UCCSD single excitation",
        "double_state": "state after one raw UCCSD double excitation",
        "canonical_particle_number": "particle number of canonical state",
        "explicit_particle_number": "particle number of explicit state",
        "single_particle_number": "particle number after the single",
        "double_particle_number": "particle number after the double",
        "canonical_num_qubits": "canonical estimator register width",
        "canonical_num_electrons": "canonical estimator electron count",
        "canonical_num_x_gates": "canonical logical X-call count",
        "explicit_num_qubits": "explicit estimator register width",
        "explicit_num_electrons": "explicit estimator occupation length",
        "explicit_num_x_gates": "explicit logical X-call count",
    }
    grouped_outputs = {
        "uccgsd_state":
        "state from the direct UCCGSD device kernel",
        "upccgsd_state":
        "state from the direct UpCCGSD device kernel",
        "ceo_state":
        "state from the direct CEO device kernel",
        "fixed_state":
        "state from custom grouped fixed-parameter UCC",
        "uccgsd_particle_number":
        "UCCGSD particle number",
        "upccgsd_particle_number":
        "UpCCGSD particle number",
        "ceo_particle_number":
        "CEO particle number",
        "fixed_particle_number":
        "custom fixed-UCC particle number",
        "fixed_num_qubits":
        "fixed-UCC estimator register width",
        "fixed_num_excitations":
        "fixed-UCC estimator group count",
        "fixed_num_pauli_rotations":
        "fixed-UCC logical rotation count",
        "fixed_max_pauli_rotations_per_excitation":
        "largest fixed-UCC group length",
    }
    givens_outputs = {
        "givens_state": "state after one real raw Givens rotation",
        "phase_givens_state": "state after one phase-aware raw rotation",
        "slater_state": "real Slater determinant from the raw flat kernel",
        "slater_norm": "norm of the Slater state",
        "slater_particle_number": "particle number of the Slater state",
        "num_spin_orbitals": "Givens estimator register width",
        "num_electrons": "Givens estimator electron count",
        "num_givens_rotations": "schedule rotation count",
        "num_exp_pauli_calls": "logical exp_pauli call count",
        "num_phase_rotations": "logical phase-rotation count",
        "two_qubit_gate_count_proxy": "documented two-qubit proxy",
        "depth_proxy": "documented serial source-operation proxy",
    }

    shared = "Read JSON from --input and write every requested numeric field to a non-object NPZ at --output. Use actual cudaq_algorithms APIs and CPU qpp-cpu fp64. Qubit zero is the least-significant statevector bit. "
    return [
        {
            "id":
            "reference_excitation_primitives",
            "families": ["state-preparation"],
            "feature_ids": [
                "state-preparation-kernel-hartree-fock",
                "state-preparation-kernel-hartree-fock-occupation",
                "state-preparation-kernel-single-excitation",
                "state-preparation-kernel-double-excitation",
                "state-preparation-resources-hartree-fock",
                "state-preparation-resources-hartree-fock-occupation",
            ],
            "summary":
            "Direct reference preparation and UCCSD excitation primitives",
            "task":
            shared +
            "Call cudaq_algorithms.stateprep.hartree_fock and cudaq_algorithms.stateprep.hartree_fock_occupation in caller-owned kernels and return their states. Separately prepare the supplied occupied states and call cudaq_algorithms.stateprep.single_excitation and cudaq_algorithms.stateprep.double_excitation directly; preserve their half-angle, Jordan-Wigner parity, pair-canonicalization and exactly-one-descending sign conventions. Return all four states and particle numbers. Also call cudaq_algorithms.stateprep.estimate_hartree_fock_resources and cudaq_algorithms.stateprep.estimate_hartree_fock_occupation_resources and return their named integer fields. The resource values are logical source-operation descriptions, not measured or hardware gate counts. This is valid-input small-system evidence only; spin describes the host canonical estimator and does not make the raw contiguous kernel an open-shell UCCSD routine.",
            "dependencies": ["cudaq", "cudaq_algorithms", "numpy"],
            "outputs":
            reference_outputs,
            "atol":
            2e-9,
            "rtol":
            2e-9,
            "required_public_apis": [
                "cudaq_algorithms.stateprep.estimate_hartree_fock_resources",
                "cudaq_algorithms.stateprep.estimate_hartree_fock_occupation_resources",
                "cudaq_algorithms.stateprep.hartree_fock",
                "cudaq_algorithms.stateprep.hartree_fock_occupation",
                "cudaq_algorithms.stateprep.single_excitation",
                "cudaq_algorithms.stateprep.double_excitation",
            ],
            "required_symbols": [
                "cudaq_algorithms.stateprep._hartree_fock.estimate_hartree_fock_resources",
                "cudaq_algorithms.stateprep._hartree_fock.estimate_hartree_fock_occupation_resources",
            ],
            "required_kernels": [
                "cudaq_algorithms.stateprep._kernels.hartree_fock",
                "cudaq_algorithms.stateprep._kernels.hartree_fock_occupation",
                "cudaq_algorithms.stateprep._kernels.single_excitation",
                "cudaq_algorithms.stateprep._kernels.double_excitation",
            ],
            "scope":
            "Two four-qubit valid inputs, including an explicit open-shell occupation; direct statevectors, particle number, exact host estimator formulas. No invalid-device, direct-UCCSD, larger-system, transpiled, or hardware-resource claim.",
        },
        {
            "id":
            "grouped_ucc_device",
            "families": ["state-preparation"],
            "feature_ids": [
                "state-preparation-kernel-uccgsd",
                "state-preparation-kernel-upccgsd",
                "state-preparation-kernel-ceo",
                "state-preparation-kernel-fixed-parameter-ucc",
                "state-preparation-resources-fixed-parameter-ucc",
            ],
            "summary":
            "Direct grouped UCCGSD, UpCCGSD, CEO, and custom UCC products",
            "task":
            shared +
            "Build matching word and coefficient groups with cudaq_algorithms.stateprep.get_uccgsd_pauli_lists, cudaq_algorithms.stateprep.get_upccgsd_pauli_lists, and cudaq_algorithms.stateprep.get_ceo_pauli_lists. In separate caller-owned kernels, prepare occupied and directly call cudaq_algorithms.stateprep.uccgsd, cudaq_algorithms.stateprep.upccgsd, and cudaq_algorithms.stateprep.ceo with their distinct supplied amplitudes. Also directly call cudaq_algorithms.stateprep.fixed_parameter_ucc with the supplied custom non-pool groups. Return all four absolute-phase statevectors and particle numbers. Call cudaq_algorithms.stateprep.estimate_fixed_parameter_ucc_resources on the custom word groups and return its named integer fields. Preserve strict group and product order and the +i, no-half-angle convention. Resource fields are exact structural formulas over the supplied groups, not compiled gate counts.",
            "dependencies": ["cudaq", "cudaq_algorithms", "numpy"],
            "outputs":
            grouped_outputs,
            "atol":
            2e-9,
            "rtol":
            2e-9,
            "required_public_apis": [
                "cudaq_algorithms.stateprep.get_uccgsd_pauli_lists",
                "cudaq_algorithms.stateprep.get_upccgsd_pauli_lists",
                "cudaq_algorithms.stateprep.get_ceo_pauli_lists",
                "cudaq_algorithms.stateprep.estimate_fixed_parameter_ucc_resources",
                "cudaq_algorithms.stateprep.uccgsd",
                "cudaq_algorithms.stateprep.upccgsd",
                "cudaq_algorithms.stateprep.ceo",
                "cudaq_algorithms.stateprep.fixed_parameter_ucc",
            ],
            "required_symbols": [
                "cudaq_algorithms.stateprep._pools.get_uccgsd_pauli_lists",
                "cudaq_algorithms.stateprep._pools.get_upccgsd_pauli_lists",
                "cudaq_algorithms.stateprep._pools.get_ceo_pauli_lists",
                "cudaq_algorithms.stateprep._hartree_fock.estimate_fixed_parameter_ucc_resources",
            ],
            "required_kernels": [
                "cudaq_algorithms.stateprep._kernels.uccgsd",
                "cudaq_algorithms.stateprep._kernels.upccgsd",
                "cudaq_algorithms.stateprep._kernels.ceo",
                "cudaq_algorithms.stateprep._kernels.fixed_parameter_ucc",
            ],
            "scope":
            "Two four-qubit/two-electron direct grouped products and a custom number-preserving pool; state and particle-number oracles plus raw grouping formulas. No optimization, larger pool, invalid-device, transpiled, or hardware-resource claim.",
        },
        {
            "id":
            "givens_raw_device",
            "families": ["state-preparation"],
            "feature_ids": [
                "state-preparation-kernel-givens-rotation",
                "state-preparation-kernel-phase-givens-rotation",
                "state-preparation-kernel-slater-determinant",
                "state-preparation-resources-givens",
            ],
            "summary":
            "Direct real and phase-aware Givens operations and raw Slater preparation",
            "task":
            shared +
            "In separate caller-owned kernels, occupy rotation_first then directly call cudaq_algorithms.stateprep.givens_rotation, and occupy phase_first then directly call cudaq_algorithms.stateprep.phase_givens_rotation. Return both statevectors with the raw rz absolute phase retained. Build a real schedule from orbitals with cudaq_algorithms.stateprep.make_givens_rotation_schedule, flatten it with cudaq_algorithms.stateprep.get_givens_rotation_indices and cudaq_algorithms.stateprep.get_givens_rotation_angles, and directly call cudaq_algorithms.stateprep.slater_determinant. Return the Slater state, norm and particle number; only this determinant state may differ by one global phase. Call cudaq_algorithms.stateprep.estimate_givens_resources and return every named integer field. These are logical formula proxies, not physical gate measurements.",
            "dependencies": ["cudaq", "cudaq_algorithms", "numpy"],
            "outputs":
            givens_outputs,
            "atol":
            2e-9,
            "rtol":
            2e-9,
            "phase_invariant_fields": ["slater_state"],
            "required_public_apis": [
                "cudaq_algorithms.stateprep.make_givens_rotation_schedule",
                "cudaq_algorithms.stateprep.get_givens_rotation_indices",
                "cudaq_algorithms.stateprep.get_givens_rotation_angles",
                "cudaq_algorithms.stateprep.estimate_givens_resources",
                "cudaq_algorithms.stateprep.givens_rotation",
                "cudaq_algorithms.stateprep.phase_givens_rotation",
                "cudaq_algorithms.stateprep.slater_determinant",
            ],
            "required_symbols": [
                "cudaq_algorithms.stateprep._givens.make_givens_rotation_schedule",
                "cudaq_algorithms.stateprep._givens.get_givens_rotation_indices",
                "cudaq_algorithms.stateprep._givens.get_givens_rotation_angles",
                "cudaq_algorithms.stateprep._givens.estimate_givens_resources",
            ],
            "required_kernels": [
                "cudaq_algorithms.stateprep._givens.givens_rotation",
                "cudaq_algorithms.stateprep._givens.phase_givens_rotation",
                "cudaq_algorithms.stateprep._givens.slater_determinant",
            ],
            "scope":
            "Two valid adjacent raw rotations and dense real 4x2 determinants; analytic/minor, norm, particle-number and schedule-formula checks. No complex-Slater, malformed-list, nonadjacent, larger-system, transpiled, or hardware-resource claim.",
        },
    ]


_BOOTSTRAP = '''\
import argparse
import json
import numpy as np
import cudaq
from cudaq_algorithms import stateprep
cudaq.set_target("qpp-cpu", precision="fp64")
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
with open(args.input) as handle:
    p = json.load(handle)

def particle_number(state):
    return float(sum(index.bit_count()*abs(amplitude)**2
                     for index, amplitude in enumerate(state)))
'''


def reference_source(case_id: str) -> str:
    if case_id == "reference_excitation_primitives":
        return _BOOTSTRAP + textwrap.dedent('''\
            @cudaq.kernel
            def canonical_entry(n: int, electrons: int):
                q = cudaq.qvector(n)
                stateprep.hartree_fock(q, electrons)

            @cudaq.kernel
            def explicit_entry(n: int, occupied: list[int]):
                q = cudaq.qvector(n)
                stateprep.hartree_fock_occupation(q, occupied)

            @cudaq.kernel
            def single_entry(n: int, occupied: list[int], theta: float,
                             first: int, second: int):
                q = cudaq.qvector(n)
                for orbital in occupied:
                    x(q[orbital])
                stateprep.single_excitation(q, theta, first, second)

            @cudaq.kernel
            def double_entry(n: int, occupied: list[int], theta: float,
                             first: int, second: int, third: int, fourth: int):
                q = cudaq.qvector(n)
                for orbital in occupied:
                    x(q[orbital])
                stateprep.double_excitation(q, theta, first, second, third, fourth)

            n = int(p["num_qubits"])
            canonical = np.asarray(cudaq.get_state(
                canonical_entry, n, int(p["num_electrons"])), dtype=complex)
            explicit = np.asarray(cudaq.get_state(
                explicit_entry, n, p["explicit_occupation"]), dtype=complex)
            single = p["single"]
            single_state = np.asarray(cudaq.get_state(
                single_entry, n, single["occupied"], float(single["theta"]),
                int(single["p"]), int(single["q"])), dtype=complex)
            double = p["double"]
            double_state = np.asarray(cudaq.get_state(
                double_entry, n, double["occupied"], float(double["theta"]),
                int(double["p"]), int(double["q"]), int(double["r"]),
                int(double["s"])), dtype=complex)
            canonical_resource = stateprep.estimate_hartree_fock_resources(
                n, int(p["num_electrons"]), int(p["spin"]))
            explicit_resource = stateprep.estimate_hartree_fock_occupation_resources(
                n, p["explicit_occupation"])
            np.savez(args.output,
                canonical_state=canonical, explicit_state=explicit,
                single_state=single_state, double_state=double_state,
                canonical_particle_number=particle_number(canonical),
                explicit_particle_number=particle_number(explicit),
                single_particle_number=particle_number(single_state),
                double_particle_number=particle_number(double_state),
                canonical_num_qubits=canonical_resource.num_qubits,
                canonical_num_electrons=canonical_resource.num_electrons,
                canonical_num_x_gates=canonical_resource.num_x_gates,
                explicit_num_qubits=explicit_resource.num_qubits,
                explicit_num_electrons=explicit_resource.num_electrons,
                explicit_num_x_gates=explicit_resource.num_x_gates)
            ''')
    if case_id == "grouped_ucc_device":
        return _BOOTSTRAP + textwrap.dedent('''\
            @cudaq.kernel
            def uccgsd_entry(n: int, occupied: list[int], thetas: list[float],
                             words: list[list[cudaq.pauli_word]],
                             coefficients: list[list[float]]):
                q = cudaq.qvector(n)
                for orbital in occupied:
                    x(q[orbital])
                stateprep.uccgsd(q, thetas, words, coefficients)

            @cudaq.kernel
            def upccgsd_entry(n: int, occupied: list[int], thetas: list[float],
                              words: list[list[cudaq.pauli_word]],
                              coefficients: list[list[float]]):
                q = cudaq.qvector(n)
                for orbital in occupied:
                    x(q[orbital])
                stateprep.upccgsd(q, thetas, words, coefficients)

            @cudaq.kernel
            def ceo_entry(n: int, occupied: list[int], thetas: list[float],
                          words: list[list[cudaq.pauli_word]],
                          coefficients: list[list[float]]):
                q = cudaq.qvector(n)
                for orbital in occupied:
                    x(q[orbital])
                stateprep.ceo(q, thetas, words, coefficients)

            @cudaq.kernel
            def fixed_entry(n: int, occupied: list[int], thetas: list[float],
                            words: list[list[cudaq.pauli_word]],
                            coefficients: list[list[float]]):
                q = cudaq.qvector(n)
                for orbital in occupied:
                    x(q[orbital])
                stateprep.fixed_parameter_ucc(q, thetas, words, coefficients)

            n = int(p["num_qubits"])
            occupied = p["occupied"]
            uccgsd_words, uccgsd_coefficients = stateprep.get_uccgsd_pauli_lists(n)
            upccgsd_words, upccgsd_coefficients = stateprep.get_upccgsd_pauli_lists(n)
            ceo_words, ceo_coefficients = stateprep.get_ceo_pauli_lists(
                int(p["num_spatial_orbitals"]))
            fixed_words = [[cudaq.pauli_word(word) for word in group]
                           for group in p["fixed_words"]]
            fixed_coefficients = p["fixed_coefficients"]
            uccgsd_state = np.asarray(cudaq.get_state(
                uccgsd_entry, n, occupied, p["uccgsd_amplitudes"],
                uccgsd_words, uccgsd_coefficients), dtype=complex)
            upccgsd_state = np.asarray(cudaq.get_state(
                upccgsd_entry, n, occupied, p["upccgsd_amplitudes"],
                upccgsd_words, upccgsd_coefficients), dtype=complex)
            ceo_state = np.asarray(cudaq.get_state(
                ceo_entry, n, occupied, p["ceo_amplitudes"], ceo_words,
                ceo_coefficients), dtype=complex)
            fixed_state = np.asarray(cudaq.get_state(
                fixed_entry, n, occupied, p["fixed_amplitudes"], fixed_words,
                fixed_coefficients), dtype=complex)
            resource = stateprep.estimate_fixed_parameter_ucc_resources(
                n, fixed_words)
            np.savez(args.output,
                uccgsd_state=uccgsd_state, upccgsd_state=upccgsd_state,
                ceo_state=ceo_state, fixed_state=fixed_state,
                uccgsd_particle_number=particle_number(uccgsd_state),
                upccgsd_particle_number=particle_number(upccgsd_state),
                ceo_particle_number=particle_number(ceo_state),
                fixed_particle_number=particle_number(fixed_state),
                fixed_num_qubits=resource.num_qubits,
                fixed_num_excitations=resource.num_excitations,
                fixed_num_pauli_rotations=resource.num_pauli_rotations,
                fixed_max_pauli_rotations_per_excitation=(
                    resource.max_pauli_rotations_per_excitation))
            ''')
    if case_id == "givens_raw_device":
        return _BOOTSTRAP + textwrap.dedent('''\
            @cudaq.kernel
            def givens_entry(n: int, theta: float, first: int, second: int):
                q = cudaq.qvector(n)
                x(q[first])
                stateprep.givens_rotation(q, theta, first, second)

            @cudaq.kernel
            def phase_givens_entry(n: int, theta: float, phase: float,
                                   first: int, second: int):
                q = cudaq.qvector(n)
                x(q[first])
                stateprep.phase_givens_rotation(q, theta, phase, first, second)

            @cudaq.kernel
            def slater_entry(n: int, indices: list[int], angles: list[float],
                             electrons: int):
                q = cudaq.qvector(n)
                stateprep.slater_determinant(q, indices, angles, electrons)

            n = int(p["num_qubits"])
            givens_state = np.asarray(cudaq.get_state(
                givens_entry, n, float(p["rotation_theta"]),
                int(p["rotation_first"]), int(p["rotation_second"])),
                dtype=complex)
            phase_givens_state = np.asarray(cudaq.get_state(
                phase_givens_entry, n, float(p["phase_theta"]), float(p["phase"]),
                int(p["phase_first"]), int(p["phase_second"])), dtype=complex)
            schedule = stateprep.make_givens_rotation_schedule(p["orbitals"])
            indices = stateprep.get_givens_rotation_indices(schedule)
            angles = stateprep.get_givens_rotation_angles(schedule)
            slater_state = np.asarray(cudaq.get_state(
                slater_entry, n, indices, angles, schedule.num_electrons),
                dtype=complex)
            resource = stateprep.estimate_givens_resources(schedule)
            np.savez(args.output,
                givens_state=givens_state,
                phase_givens_state=phase_givens_state,
                slater_state=slater_state,
                slater_norm=np.linalg.norm(slater_state),
                slater_particle_number=particle_number(slater_state),
                num_spin_orbitals=resource.num_spin_orbitals,
                num_electrons=resource.num_electrons,
                num_givens_rotations=resource.num_givens_rotations,
                num_exp_pauli_calls=resource.num_exp_pauli_calls,
                num_phase_rotations=resource.num_phase_rotations,
                two_qubit_gate_count_proxy=resource.two_qubit_gate_count_proxy,
                depth_proxy=resource.depth_proxy)
            ''')
    raise KeyError(case_id)
