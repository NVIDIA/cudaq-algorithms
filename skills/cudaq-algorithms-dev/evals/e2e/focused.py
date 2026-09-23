# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Separate natural-language discovery campaign using existing quantum oracles.

Usage mirrors run.py: focused.py prepare|preflight|run|report --output PATH.
The historical registry remains the default when run.py is invoked directly.
"""
from __future__ import annotations

from copy import deepcopy
import sys

import numpy as np

import quantum_cases
import run

SUITE = "focused_natural_language"
ORACLES = {
    "focused_determinant_energy": "slater_energy",
    "focused_signed_action": "pauli_action",
    "focused_phase_filter": "qsvt_filter"
}


def case_specs():
    base = {spec["id"]: spec for spec in quantum_cases.case_specs()}
    tasks = {
        "focused_determinant_energy":
        ("Orthonormal orbitals to an electronic state and observables",
         "The input contains orthonormal occupied-orbital columns, encoded as "
         "orbitals_real + i orbitals_imag. Prepare their two-electron antisymmetric "
         "determinant on four spin orbitals using the package. Obtain the prepared "
         "system statevector and measure its energy for the supplied Hamiltonian "
         "terms=[coefficient,word], including identity, and the expected number "
         "of occupied spin orbitals. Return state, energy and particle_number. "
         "Only the overall phase of state is immaterial."),
        "focused_signed_action":
        ("Signed Hamiltonian action through a quantum encoding",
         "Encode the real signed sum of Pauli terms supplied as "
         "terms=[coefficient,word], including identity, with the package's "
         "quantum-circuit primitives. Prepare ket_real+i ket_imag, apply the "
         "encoding and extract its unnormalized all-zero-ancilla component. "
         "Return block=H ket/alpha, action=H ket, alpha=sum absolute coefficients, "
         "success_probability=norm(block)^2 and energy=<ket|H|ket>. Preserve "
         "absolute and relative complex phases; do not normalize postselection."
         ),
        "focused_phase_filter":
        ("Supplied-phase spectral transformation with directed steps",
         "Use the package's quantum-circuit primitives to apply the supplied "
         "phase sequence to the signed Hamiltonian terms and input ket. "
         "The signal walk has scalar invariant-subspace matrix "
         "[[-x,-sqrt(1-x*x)],[sqrt(1-x*x),-x]] for an eigenvalue x of H/alpha, "
         "where alpha=sum absolute coefficients, including identity. Apply "
         "the first phase, then each directed walk step and next phase in "
         "order; directions 0 and 1 mean forward and adjoint. A phase tagged "
         "by convention='qsvt' rotates signal zero by exp(i*phase); 'qsp' "
         "doubles that angle. Return the complex, unnormalized all-zero-signal "
         "block, success_probability=norm(block)^2 and alpha. Preserve the "
         "actual circuit phase. The phases are scientific inputs: no phase "
         "synthesis is requested, and this is not a normalized postselected state."
         ),
    }
    # Discovery prompts do not prescribe a convenience helper. Require the
    # computational entries shared by both the helper and direct circuit paths.
    required = {
        "focused_determinant_energy": {
            "required_public_apis": [
                "cudaq_algorithms.stateprep.make_givens_rotation_schedule",
                "cudaq_algorithms.stateprep.complex_slater_determinant"
            ],
            "required_symbols": [
                "cudaq_algorithms.stateprep._givens.make_givens_rotation_schedule"
            ],
            "required_kernels":
            ["cudaq_algorithms.stateprep._givens.complex_slater_determinant"],
        },
        "focused_signed_action": {
            "required_public_apis": ["cudaq_algorithms.PauliLCU"],
            "required_symbols":
            ["cudaq_algorithms.pauli_lcu.PauliLCU.__init__"],
            "required_kernels": ["cudaq_algorithms.pauli_lcu.apply"],
        },
        "focused_phase_filter": {
            "required_public_apis":
            ["cudaq_algorithms.PhaseSequence", "cudaq_algorithms.QSVT.kernel"],
            "required_symbols": [
                "cudaq_algorithms.qsvt.PhaseSequence.__init__",
                "cudaq_algorithms.qsvt.QSVT.kernel"
            ],
            "required_kernels": [
                "cudaq_algorithms.pauli_lcu.apply",
                "cudaq_algorithms.common_kernels.signal_phase",
                "cudaq_algorithms.common_kernels.reflect_about_zero"
            ],
        },
    }
    specs = []
    for cid, oracle in ORACLES.items():
        spec = deepcopy(base[oracle])
        summary, task = tasks[cid]
        spec.update(
            id=cid,
            oracle_case=oracle,
            summary=summary,
            task=task +
            " Pauli words list q0 first, and q0 is the least significant statevector index bit.",
            disclose_required_apis=False,
            **required[cid])
        specs.append(spec)
    return specs


def parameters(case_id, variant):
    """New fixed scientific inputs, with distinct public and held-out variants."""
    if variant not in (0, 1):
        raise ValueError("variant must be 0 (public) or 1 (held out)")
    oracle = ORACLES[case_id]
    params = quantum_cases.parameters(oracle, variant)
    rng = np.random.default_rng(90317 + 101 * list(ORACLES).index(case_id) +
                                variant)
    params["terms"] = [[float(
        coefficient * scale), word] for (coefficient, word), scale in zip(
            params["terms"], rng.uniform(.7, 1.3, len(params["terms"])))]
    if oracle == "slater_energy":
        raw = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
        orbitals, _ = np.linalg.qr(raw)
        params.update(orbitals_real=orbitals.real.tolist(),
                      orbitals_imag=orbitals.imag.tolist())
    else:
        ket = rng.normal(size=4) + 1j * rng.normal(size=4)
        ket /= np.linalg.norm(ket)
        params.update(ket_real=ket.real.tolist(), ket_imag=ket.imag.tolist())
        if oracle == "qsvt_filter":
            params.update(phases=rng.uniform(-.6, .6, 4).tolist(),
                          directions=[1, 0, 1])
    return params


def expected(case_id, params):
    return quantum_cases.expected(ORACLES[case_id], params)


def reference_source(case_id):
    return quantum_cases.reference_source(ORACLES[case_id])


def main(argv=None):
    historical, suite = run.CASES, run.SUITE
    try:
        run.CASES = {
            spec["id"]: (sys.modules[__name__], spec)
            for spec in case_specs()
        }
        run.SUITE = SUITE
        run.main(argv)
    finally:
        run.CASES, run.SUITE = historical, suite


if __name__ == "__main__":
    main()
