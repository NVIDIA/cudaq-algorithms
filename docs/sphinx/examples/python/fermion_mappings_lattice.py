#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# [Begin Documentation]
"""Compare fermion mappings on one interacting four-site spinless ring.

Before simulating electrons or other fermions, we must translate their
creation, annihilation, and interaction terms into operations on qubits. This
example applies several translations to the same small lattice model, where
particles can hop between sites and interact when neighboring sites are
occupied. The aim is to keep the physical model fixed while examining how its
qubit representation changes.

You should learn how to check that different mappings preserve the model's
energies and how to interpret differences in Pauli terms, operator weights,
and qubit counts. Some representations include states that do not belong to
the intended physical problem; their parity and stabilizer constraints
identify the subspace to compare. An independent occupation-basis Hamiltonian
supplies the reference, making the signs and physical-subspace checks explicit
before any resource comparison is interpreted.

H = -t sum_<i,j> (adag_i a_j + adag_j a_i) + V sum_<i,j> n_i n_j,
with t=1, V=2 and each undirected bond listed once. Sparse tensor entries
multiply adag_i a_j and adag_i adag_j a_k a_l directly, without a chemistry
prefactor. NumPy builds an independent occupation-basis reference.

Superfast encodings require restriction to their stabilizers' joint +1
eigenspace, separately for each requested fermion parity. This host-side
tutorial validates mappings; it does not prepare superfast code states.
Run: python3 fermion_mappings_lattice.py
Add --verbose to show individual validation checks; failures are always reported.
"""

import argparse
import time

import numpy as np

from cudaq_algorithms.fermion import (bravyi_kitaev, bravyi_kitaev_superfast,
                                      bravyi_kitaev_superfast_stabilizers,
                                      jordan_wigner)

ATOL = 1e-10
PRUNE = 1e-12
MODES = 4
BONDS = [(0, 1), (1, 2), (2, 3), (0, 3)]
PAULIS = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1, -1]),
}


def check(label, actual, expected, tolerance=ATOL, *, verbose=False):
    error = float(np.max(np.abs(np.asarray(actual) - expected)))
    passed = np.isfinite(error) and error <= tolerance
    if verbose or not passed:
        print(f"CHECK {label}: {error:.3e} <= {tolerance:.3e} -> "
              f"{'PASS' if passed else 'FAIL'}")
    if not passed:
        raise AssertionError(label)
    return error


def annihilator(mode):
    """Occupation bits are little-endian; occupied lower modes give the sign."""
    result = np.zeros((1 << MODES, 1 << MODES), dtype=complex)
    for occupation in range(1 << MODES):
        if occupation & (1 << mode):
            lower_occupations = occupation & ((1 << mode) - 1)
            result[occupation ^ (1 << mode),
                   occupation] = (-1 if lower_occupations.bit_count() %
                                  2 else 1)
    return result


def fock_hamiltonian(hopping, interaction, *, verbose=False):
    lower = [annihilator(mode) for mode in range(MODES)]
    number = [a.conj().T @ a for a in lower]
    result = np.zeros((1 << MODES, 1 << MODES), dtype=complex)
    for i, j in BONDS:
        density_body = lower[i].conj().T @ lower[j].conj(
        ).T @ lower[j] @ lower[i]
        check(f"tensor convention on bond {i},{j}",
              density_body,
              number[i] @ number[j],
              verbose=verbose)
        result += -hopping * (lower[i].conj().T @ lower[j] +
                              lower[j].conj().T @ lower[i])
        result += interaction * number[i] @ number[j]
    return result


def pauli_data(operator, width):
    """Use public terms at an explicit width; to_matrix can compact support."""
    result = np.zeros((1 << width, 1 << width), dtype=complex)
    weights = []
    if operator.qubit_count > width:
        raise AssertionError("mapped support exceeds the declared register")
    for term in operator:
        coefficient = complex(term.evaluate_coefficient())
        if abs(coefficient) <= PRUNE:
            continue
        word = term.get_pauli_word(width)
        factor = np.ones((1, 1), dtype=complex)
        for letter in word:
            factor = np.kron(PAULIS[letter], factor)
        result += coefficient * factor
        weights.append(sum(letter != "I" for letter in word))
    return result, weights


def code_basis(hamiltonian, stabilizers, width, *, verbose=False):
    dimension = 1 << width
    identity = np.eye(dimension)
    projector = identity.astype(complex)
    matrices = [pauli_data(stabilizer, width)[0] for stabilizer in stabilizers]
    for index, stabilizer in enumerate(matrices):
        check(f"S{index} Hermitian",
              stabilizer,
              stabilizer.conj().T,
              verbose=verbose)
        check(f"S{index} involution",
              stabilizer @ stabilizer,
              identity,
              verbose=verbose)
        check(f"S{index} commutes with H",
              stabilizer @ hamiltonian,
              hamiltonian @ stabilizer,
              verbose=verbose)
        for other in matrices[:index]:
            check("stabilizers commute",
                  stabilizer @ other,
                  other @ stabilizer,
                  verbose=verbose)
        projector = projector @ ((identity + stabilizer) / 2)
    check("joint projector is Hermitian",
          projector,
          projector.conj().T,
          verbose=verbose)
    check("joint projector is idempotent",
          projector @ projector,
          projector,
          verbose=verbose)
    values, vectors = np.linalg.eigh(projector)
    basis = vectors[:, values > 0.5]
    check("code basis is orthonormal",
          basis.conj().T @ basis,
          np.eye(basis.shape[1]),
          verbose=verbose)
    for stabilizer in matrices:
        check("code basis has stabilizer +1",
              stabilizer @ basis,
              basis,
              verbose=verbose)
    return basis


def row(name, weights, stabilizers, dimension, parity, error):
    return (f"{name:15s} {MODES:6d} {len(weights):5d} "
            f"{max(weights):6d} {np.mean(weights):7.3f} "
            f"{stabilizers:4d} {dimension:8d} {parity:>6s} {error:10.3e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show each validation check (failures are always reported)")
    verbose = parser.parse_args().verbose
    started = time.perf_counter()
    hopping, interaction = 1.0, 2.0
    degrees = [sum(mode in bond for bond in BONDS) for mode in range(MODES)]
    check("all four modes have degree two",
          degrees,
          np.full(MODES, 2),
          0,
          verbose=verbose)
    # Starting at mode zero, these explicit bonds reach the whole ring.
    reached = {0}
    for _ in range(MODES):
        reached |= {v for u, v in BONDS if u in reached}
        reached |= {u for u, v in BONDS if v in reached}
    if reached != set(range(MODES)):
        raise AssertionError("interaction graph is disconnected")
    one_body = {}
    for i, j in BONDS:
        one_body[i, j] = one_body[j, i] = -hopping
    two_body = {(i, j, j, i): interaction for i, j in BONDS}
    print(
        f"Spinless ring: modes={MODES}, bonds={BONDS}, t={hopping}, V={interaction}"
    )
    print(f"Input/output coefficient pruning threshold: {PRUNE:.1e}")
    reference = fock_hamiltonian(hopping, interaction, verbose=verbose)
    check("physical Hamiltonian is Hermitian",
          reference,
          reference.conj().T,
          verbose=verbose)
    spectra = np.linalg.eigvalsh(reference)
    rows = []
    common = dict(num_modes=MODES, tolerance=PRUNE)
    for name, compiler in (("JW", jordan_wigner), ("BK", bravyi_kitaev)):
        mapped, weights = pauli_data(compiler(one_body, two_body, **common),
                                     MODES)
        check(f"{name} Hermitian", mapped, mapped.conj().T, verbose=verbose)
        if name == "JW":
            check("JW full matrix versus occupation reference",
                  mapped,
                  reference,
                  verbose=verbose)
        error = check(f"{name} full spectrum",
                      np.linalg.eigvalsh(mapped),
                      spectra,
                      verbose=verbose)
        rows.append(row(name, weights, 0, 1 << MODES, "both", error))

    # No degree padding is needed: both constructions have four qubits.
    width = len(BONDS)
    check("edge/local-mode width for degree-two ring",
          width,
          MODES,
          0,
          verbose=verbose)
    for local_modes in ("standard", "binary_tree"):
        for parity in (0, 1):
            options = dict(common,
                           interaction_graph=BONDS,
                           local_modes=local_modes,
                           parity=parity)
            operator = bravyi_kitaev_superfast(one_body, two_body, **options)
            stabilizers = bravyi_kitaev_superfast_stabilizers(
                one_body, two_body, **options)
            check("independent cycle count",
                  len(stabilizers),
                  len(BONDS) - MODES + 1,
                  0,
                  verbose=verbose)
            mapped, weights = pauli_data(operator, width)
            check("superfast Hamiltonian Hermitian",
                  mapped,
                  mapped.conj().T,
                  verbose=verbose)
            basis = code_basis(mapped, stabilizers, width, verbose=verbose)
            indices = [
                occupation for occupation in range(1 << MODES)
                if occupation.bit_count() % 2 == parity
            ]
            check("physical parity dimension",
                  len(indices),
                  8,
                  0,
                  verbose=verbose)
            check("joint +1 code dimension",
                  basis.shape[1],
                  len(indices),
                  0,
                  verbose=verbose)
            physical = reference[np.ix_(indices, indices)]
            restricted = basis.conj().T @ mapped @ basis
            error = check(
                f"{local_modes}, parity={parity}, full code spectrum",
                np.linalg.eigvalsh(restricted),
                np.linalg.eigvalsh(physical),
                verbose=verbose)
            rows.append(
                row(local_modes, weights, len(stabilizers), basis.shape[1],
                    str(parity), error))

    print(
        "\nmapping         qubits terms max-wt mean-wt stab code-dim parity spec-error"
    )
    for result in rows:
        print(result)
    print(
        "JW and BK encode both parities; BK occupation bits use another basis."
    )
    print("Superfast encodes one parity in the joint +1 stabilizer subspace.")
    print("Do not reuse JW occupation preparation on a superfast register.")
    print(
        "binary_tree uses local Fenwick Majoranas; standard uses edge operators."
    )
    print("error_correcting needs a 3-connected graph of even degree >=6; "
          "this degree-two ring does not satisfy those conditions.")
    print(
        "This small fixture establishes contracts, not an asymptotic winner.")
    print(f"Host-side mapping only; maximum declared qubits: {width}; "
          f"elapsed: {time.perf_counter()-started:.2f}s")


if __name__ == "__main__":
    main()
