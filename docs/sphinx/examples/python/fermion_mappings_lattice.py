#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# [Begin Documentation]
"""Compare fermion mappings on an interacting spinless lattice model.

Before simulating electrons or other fermions, we must translate their
creation, annihilation, and interaction terms into operations on qubits. This
example applies several translations to the same lattice model, where
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

The example has two parts. A four-site ring is small enough to diagonalize,
so every mapping is checked against the reference spectrum there. Open square
lattices from 16 to 400 modes then show how the mappings separate with
size: Pauli weights, qubit counts, and stabilizer counts are read from the
mapped operators, and nothing is simulated at those sizes.

H = -t sum_<i,j> (adag_i a_j + adag_j a_i) + V sum_<i,j> n_i n_j,
with t=1, V=2 and each undirected bond listed once. Sparse tensor entries
multiply adag_i a_j and adag_i adag_j a_k a_l directly, without a chemistry
prefactor. NumPy builds an independent occupation-basis reference.

Superfast encodings require restriction to their stabilizers' joint +1
eigenspace, separately for each requested fermion parity. This host-side
tutorial validates mappings; it does not prepare superfast code states,
and its weights describe operators, not circuits or gate counts.
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
LATTICE_SIDES = (4, 7, 10, 14, 20)
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


def square_lattice(side):
    """Open side-by-side lattice: modes numbered row by row, each bond once."""
    bonds = []
    for y in range(side):
        for x in range(side):
            mode = x + side * y
            if x + 1 < side:
                bonds.append((mode, mode + 1))
            if y + 1 < side:
                bonds.append((mode, mode + side))
    return side * side, bonds


def pauli_bits(operator, width):
    """Coefficients and X/Z support bits of the retained Pauli words.

    Two words commute exactly when their supports overlap with X-or-Y on one
    side and Z-or-Y on the other at an even number of positions, so these
    bit rows check commutation without any matrix.
    """
    if operator.max_degree >= width:
        raise AssertionError("mapped support exceeds the declared register")
    coefficients, x_bits, z_bits = [], [], []
    for term in operator:
        coefficient = complex(term.evaluate_coefficient())
        if abs(coefficient) <= PRUNE:
            continue
        letters = np.frombuffer(term.get_pauli_word(width).encode(),
                                dtype=np.uint8)
        x_bits.append((letters == ord("X")) | (letters == ord("Y")))
        z_bits.append((letters == ord("Z")) | (letters == ord("Y")))
        coefficients.append(coefficient)
    return (np.array(coefficients), np.array(x_bits, dtype=int),
            np.array(z_bits, dtype=int))


def stabilizer_bits(stabilizers, width):
    """Signs and X/Z support bits of the stabilizers, one word per row."""
    parts = [pauli_bits(stabilizer, width) for stabilizer in stabilizers]
    return tuple(np.concatenate(column) for column in zip(*parts))


def anticommuting_pairs(x_bits, z_bits, other_x, other_z):
    """Number of word pairs, one from each list, that anticommute."""
    return int(((x_bits @ other_z.T + z_bits @ other_x.T) % 2).sum())


def lattice_rows(hopping, interaction, *, verbose=False):
    """Operator-level comparison on open square lattices; no simulation.

    The spectra of these registers are out of reach, so the checks are
    exact statements about the Pauli words: real coefficients, equal term
    counts, the Jordan-Wigner string length, the Bravyi-Kitaev crossover,
    one stabilizer per independent cycle, and stabilizers that commute with
    every Hamiltonian term.
    """
    rows, largest = [], 0
    superfast_max = {"standard": [], "binary_tree": []}
    linear_max = {"JW": [], "BK": []}
    for side in LATTICE_SIDES:
        modes, bonds = square_lattice(side)
        size = f"{side}x{side}"
        one_body = {}
        for i, j in bonds:
            one_body[i, j] = one_body[j, i] = -hopping
        two_body = {(i, j, j, i): interaction for i, j in bonds}
        common = dict(num_modes=modes, tolerance=PRUNE)

        # Each entry: mapping name, operator, register width, stabilizers.
        mapped = [(name, compiler(one_body, two_body, **common), modes, None)
                  for name, compiler in (("JW", jordan_wigner),
                                         ("BK", bravyi_kitaev))]
        for local_modes in superfast_max:
            options = dict(common,
                           interaction_graph=bonds,
                           local_modes=local_modes,
                           parity=0)
            operator = bravyi_kitaev_superfast(one_body, two_body, **options)
            stabilizers = bravyi_kitaev_superfast_stabilizers(
                one_body, two_body, **options)
            # The register is one qubit per graph edge; binary_tree first
            # pads odd-degree vertices with zero-coefficient edges.
            width = 1 + max(part.max_degree
                            for part in (operator, *stabilizers))
            padding = width - len(bonds)
            check(f"{size} {local_modes} register holds every bond",
                  min(padding, 0),
                  0,
                  0,
                  verbose=verbose)
            if local_modes == "standard":
                check(f"{size} standard uses one qubit per bond",
                      padding,
                      0,
                      0,
                      verbose=verbose)
            mapped.append((local_modes, operator, width, stabilizers))

        term_counts = []
        for name, operator, width, stabilizers in mapped:
            coefficients, x_bits, z_bits = pauli_bits(operator, width)
            check(f"{size} {name} coefficients are real",
                  coefficients.imag,
                  0,
                  verbose=verbose)
            weights = (x_bits | z_bits).sum(axis=1)
            term_counts.append(len(weights))
            stabilizer_cells = f"{'-':>4s} {'-':>11s}"
            if name in linear_max:
                linear_max[name].append(int(weights.max()))
            if name == "JW":
                # A hop between rows spans one full row of the row-by-row
                # mode order: X or Y on both ends and Z on the modes between.
                check(f"{size} JW max weight is side + 1",
                      weights.max(),
                      side + 1,
                      0,
                      verbose=verbose)
            if stabilizers is not None:
                check(f"{size} {name} one stabilizer per independent cycle",
                      len(stabilizers),
                      width - modes + 1,
                      0,
                      verbose=verbose)
                signs, stab_x, stab_z = stabilizer_bits(stabilizers, width)
                check(
                    f"{size} {name} stabilizers are signed Pauli words",
                    [len(signs), np.max(np.abs(np.abs(signs) - 1))],
                    [len(stabilizers), 0],
                    verbose=verbose)
                check(f"{size} {name} stabilizers commute with every term",
                      anticommuting_pairs(x_bits, z_bits, stab_x, stab_z),
                      0,
                      0,
                      verbose=verbose)
                check(f"{size} {name} stabilizers commute with each other",
                      anticommuting_pairs(stab_x, stab_z, stab_x, stab_z),
                      0,
                      0,
                      verbose=verbose)
                stabilizer_weight = (stab_x | stab_z).sum(axis=1).max()
                stabilizer_cells = (f"{len(stabilizers):4d} "
                                    f"{stabilizer_weight:11d}")
                superfast_max[name].append(int(weights.max()))
            rows.append(f"{size:7s} {modes:5d} {name:12s} {width:6d} "
                        f"{len(weights):5d} {weights.max():6d} "
                        f"{weights.mean():7.3f} {stabilizer_cells}")
            largest = max(largest, width)
        check(f"{size} all mappings keep the same number of terms",
              term_counts,
              np.full(len(term_counts), term_counts[0]),
              0,
              verbose=verbose)
    for name, max_weights in superfast_max.items():
        check(f"{name} max weight does not grow with the lattice",
              max_weights,
              np.full(len(max_weights), max_weights[0]),
              0,
              verbose=verbose)
    # BK strings grow with log(modes) and JW strings with the lattice side,
    # so BK is the heavier mapping on the smallest lattice and the lighter
    # one on the largest.
    excess = np.subtract(linear_max["BK"], linear_max["JW"])
    check("BK max weight crosses below JW as the lattice grows",
          np.sign(excess[[0, -1]]), [1, -1],
          0,
          verbose=verbose)
    return rows, largest


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
    print("The ring fixes the contracts; its four qubits cannot separate the "
          "mappings.")

    lattice, largest = lattice_rows(hopping, interaction, verbose=verbose)
    print("\nOpen L x L lattices, same model; superfast rows use parity 0.")
    print("lattice modes mapping      qubits terms max-wt mean-wt stab "
          "stab-max-wt")
    for result in lattice:
        print(result)
    print("JW max weight is L+1 in this row-by-row mode order; BK grows with "
          "log(modes)")
    print("and is the heavier of the two up to 10x10, equal at 14x14, and "
          "lighter at 20x20.")
    print("Both superfast forms keep a fixed max weight, paid for in qubits "
          "(one per")
    print("graph edge, after binary_tree pads odd-degree vertices) and one "
          "stabilizer")
    print("per independent cycle.")
    print("The returned stabilizers follow a cycle basis that is not "
          "plaquette-local,")
    print("so their weights grow with the lattice although the Hamiltonian "
          "terms do not.")
    print("Weights describe mapped operators only: no state preparation, "
          "circuits, or")
    print("gate counts are compared, and no spectrum is computed at these "
          "sizes.")
    print(f"Host-side mapping only; maximum declared qubits: {largest}; "
          f"elapsed: {time.perf_counter()-started:.2f}s")


if __name__ == "__main__":
    main()
