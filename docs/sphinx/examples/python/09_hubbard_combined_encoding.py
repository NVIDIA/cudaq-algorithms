#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# [Begin Documentation]
"""Example 9 — Combine block encodings: Fermi-Hubbard with T in momentum space.

The 2D Fermi-Hubbard Hamiltonian splits into two pieces that are each
diagonal in a different basis::

    H = T + U,   T = -t sum_<ij>,s (a+_is a_js + h.c.),   U = u sum_i n_iu n_id

* ``T`` is diagonal in momentum space. On a periodic lattice its
  one-particle eigenmodes are known in closed form: plane waves with
  momentum ``k = (2 pi kx / lx, 2 pi ky / ly)`` and dispersion
  ``eps(k) = -2 t (cos kx + cos ky)``. Since ``eps(k) = eps(-k)``, the real
  standing waves ``cos(k . r)`` and ``sin(k . r)`` diagonalize ``T`` just
  as well, giving a real orthogonal basis in which ``T = sum_k eps(k) n_k``
  is a sum of single-qubit Z words with ``alpha_T = sum_k |eps(k)|``.
* ``U`` is diagonal in real space already: ``n_iu n_id`` is Z words in the
  computational basis.

Encode each piece where it is cheap, then combine. `RotatedEncoding`
conjugates the momentum-diagonal `PauliLCU` of ``T`` by the orbital
rotation into the momentum basis (`from_orbital_rotation` compiles the
real orthogonal mode matrix to a Givens-rotation network on both spin
sectors), and `SumEncoding` adds the two encodings with one slot qubit and
one flag qubit: ``alpha = alpha_T + alpha_U``. The result satisfies the
`BlockEncoding` protocol, so `Walk` and `QSVT` consume it unchanged, and
it beats a flat `PauliLCU` of the same Hamiltonian on ``alpha`` because the
hopping is encoded with ``sum_k |eps(k)|`` instead of the Jordan-Wigner
``XX + YY`` strings' one-norm.

Splitting ``T`` from ``U`` and treating ``T`` in the Fourier basis is the
structure exploited in Campbell's Hubbard resource analysis (Early
fault-tolerant simulations of the Hubbard model, arXiv:2012.09238), there
in split-operator Trotter form; this example realizes the same
decomposition at the block-encoding level.

Every encoded block is checked against an independent sparse Jordan-Wigner
reference, including a negative-coefficient sum (``H - E_ref I``) and a
short QSVT sequence against the dense 2x2 signal model. The default 2x2
ring (8 system + 6 ancilla qubits) runs on the CPU simulator; a larger
lattice (``--lx``, ``--ly``) above ``--max-qubits`` prints the alpha table
only.

Run: python3 09_hubbard_combined_encoding.py [--lx 2 --ly 2 --t 1 --u 4]
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import cudaq
import numpy as np
import scipy.sparse as sp

from cudaq_algorithms import (BlockEncoding, PauliLCU, PhaseSequence, QSVT,
                              RotatedEncoding, SumEncoding, Walk, state_from)
from cudaq_algorithms import sim_utils as sim

# ----------------------------------------------------------------------
# The lattice. Sites are indexed site(ix, iy) = ix + lx * iy; spin
# orbitals are interleaved (2p up, 2p + 1 down), the library's
# Jordan-Wigner convention.
# ----------------------------------------------------------------------


def hopping_bonds(lx: int, ly: int) -> list[tuple[int, int]]:
    """Unique nearest-neighbor bonds of the lx x ly periodic lattice.

    Wrap-around bonds that coincide with an existing bond (length-2
    directions) are deduplicated, so the 2x2 lattice is the 4-site ring.
    """

    def site(ix, iy):
        return ix % lx + lx * (iy % ly)

    bonds = set()
    for iy in range(ly):
        for ix in range(lx):
            if lx > 1:
                bonds.add(tuple(sorted((site(ix, iy), site(ix + 1, iy)))))
            if ly > 1:
                bonds.add(tuple(sorted((site(ix, iy), site(ix, iy + 1)))))
    return sorted(bonds)


def hopping_matrix(lx: int, ly: int, t: float) -> np.ndarray:
    n = lx * ly
    matrix = np.zeros((n, n))
    for p, q in hopping_bonds(lx, ly):
        matrix[p, q] = matrix[q, p] = -t
    return matrix


def dispersion(lx: int, ly: int, t: float, kx: int, ky: int) -> float:
    """eps(k) with hopping_bonds' bond convention: a direction of length 1
    contributes nothing and a direction of length 2 has a single bond."""

    def direction(length: int, index: int) -> float:
        if length == 1:
            return 0.0
        factor = 1.0 if length == 2 else 2.0
        return -factor * t * math.cos(2.0 * math.pi * index / length)

    return direction(lx, kx) + direction(ly, ky)


def momentum_modes(lx: int, ly: int, t: float):
    """Real orthonormal momentum eigenbasis of the hopping matrix.

    Returns ``(modes, energies, labels)``: ``modes`` is ``n x n`` with the
    modes as columns, ``energies[m] = eps(k_m)``, and ``labels[m]`` names
    the momentum. Self-conjugate momenta (components 0 or pi) are already
    real plane waves; every other pair ``{k, -k}`` becomes the standing
    waves ``sqrt(2/N) cos(k . r)`` and ``sqrt(2/N) sin(k . r)``. Modes are
    ordered by increasing energy.
    """
    n = lx * ly
    ix = np.arange(n) % lx
    iy = np.arange(n) // lx
    columns, energies, labels = [], [], []
    seen = set()
    for ky in range(ly):
        for kx in range(lx):
            if (kx, ky) in seen:
                continue
            partner = ((lx - kx) % lx, (ly - ky) % ly)
            seen.add((kx, ky))
            seen.add(partner)
            phase = 2.0 * math.pi * (kx * ix / lx + ky * iy / ly)
            eps = dispersion(lx, ly, t, kx, ky)
            if partner == (kx, ky):
                columns.append(np.cos(phase) / math.sqrt(n))
                energies.append(eps)
                labels.append(f"k=({kx},{ky})")
            else:
                scale = math.sqrt(2.0 / n)
                columns.append(scale * np.cos(phase))
                energies.append(eps)
                labels.append(f"k=+-({kx},{ky}) cos")
                columns.append(scale * np.sin(phase))
                energies.append(eps)
                labels.append(f"k=+-({kx},{ky}) sin")
    order = np.argsort(energies, kind="stable")
    modes = np.column_stack([columns[i] for i in order])
    return (modes, np.array([energies[i]
                             for i in order]), [labels[i] for i in order])


def _z_word(num_qubits: int, *qubits: int) -> str:
    word = ["I"] * num_qubits
    for q in qubits:
        word[q] = "Z"
    return "".join(word)


def momentum_diagonal_words(energies: np.ndarray,
                            num_qubits: int) -> dict[str, float]:
    """T in its eigenbasis: sum_m eps_m n_m per spin, with n = (I - Z)/2."""
    words: dict[str, float] = {}
    constant = float(np.sum(energies))  # eps_m / 2 over both spins
    if abs(constant) > 1e-12:
        words["I" * num_qubits] = constant
    for m, eps in enumerate(energies):
        if abs(eps) > 1e-12:
            for spin in range(2):
                words[_z_word(num_qubits, 2 * m + spin)] = -0.5 * float(eps)
    return words


def interaction_words(num_sites: int, u: float,
                      num_qubits: int) -> dict[str, float]:
    """U = u sum_i n_iu n_id = u/4 sum_i (I - Z_2i - Z_2i+1 + Z_2i Z_2i+1)."""
    words: dict[str, float] = {"I" * num_qubits: 0.25 * u * num_sites}
    for i in range(num_sites):
        up, down = 2 * i, 2 * i + 1
        words[_z_word(num_qubits, up)] = -0.25 * u
        words[_z_word(num_qubits, down)] = -0.25 * u
        words[_z_word(num_qubits, up, down)] = 0.25 * u
    return words


def flat_hubbard_words(lx: int, ly: int, t: float,
                       u: float) -> dict[str, float]:
    """The full Hamiltonian as computational-frame Pauli words (baseline).

    A hopping bond (p < q) per spin is (X Z..Z X + Y Z..Z Y)/2 on the
    Jordan-Wigner string between the two modes, coefficient -t/2 each.
    """
    n = lx * ly
    num_qubits = 2 * n
    words = dict(interaction_words(n, u, num_qubits))
    for p, q in hopping_bonds(lx, ly):
        for spin in range(2):
            m1, m2 = 2 * p + spin, 2 * q + spin
            for pauli in ("X", "Y"):
                word = ["I"] * num_qubits
                word[m1] = pauli
                word[m2] = pauli
                for z in range(m1 + 1, m2):
                    word[z] = "Z"
                key = "".join(word)
                words[key] = words.get(key, 0.0) - 0.5 * t
    return {w: c for w, c in words.items() if abs(c) > 1e-12}


# ----------------------------------------------------------------------
# Independent sparse Jordan-Wigner reference (interleaved spins, qubit 0
# least significant, matching CUDA-Q's statevector order).
# ----------------------------------------------------------------------


def _sparse_annihilators(num_qubits: int) -> list[sp.csr_matrix]:
    z2 = sp.csr_matrix(np.diag([1.0, -1.0]))
    identity = sp.identity(2, format="csr")
    lowering = sp.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]]))
    out = []
    for mode in range(num_qubits):
        ops = ([z2] * mode + [lowering] + [identity] *
               (num_qubits - mode - 1))[::-1]
        matrix = sp.identity(1, format="csr")
        for op in ops:
            matrix = sp.kron(matrix, op, format="csr")
        out.append(matrix)
    return out


def reference_hamiltonians(lx: int, ly: int, t: float, u: float):
    """Sparse (T, U) built directly from bonds and number operators."""
    n = lx * ly
    num_qubits = 2 * n
    lower = _sparse_annihilators(num_qubits)
    raise_ = [op.conj().T.tocsr() for op in lower]
    dim = 1 << num_qubits
    hopping = sp.csr_matrix((dim, dim), dtype=complex)
    for p, q in hopping_bonds(lx, ly):
        for spin in range(2):
            m1, m2 = 2 * p + spin, 2 * q + spin
            hopping = hopping - t * (raise_[m1] @ lower[m2] +
                                     raise_[m2] @ lower[m1])
    interaction = sp.csr_matrix((dim, dim), dtype=complex)
    for i in range(n):
        n_up = raise_[2 * i] @ lower[2 * i]
        n_down = raise_[2 * i + 1] @ lower[2 * i + 1]
        interaction = interaction + u * (n_up @ n_down)
    return hopping.tocsr(), interaction.tocsr()


def chebyshev_moment(h_scaled, ket: np.ndarray, order: int) -> float:
    """<ket| T_order(h_scaled) |ket> by the Chebyshev recurrence."""
    t_prev, t_cur = ket, h_scaled @ ket
    if order == 0:
        return float(np.real(ket.conj() @ t_prev))
    for _ in range(order - 1):
        t_prev, t_cur = t_cur, 2.0 * (h_scaled @ t_cur) - t_prev
    return float(np.real(ket.conj() @ t_cur))


def qsvt_response(sequence: PhaseSequence, x: float) -> complex:
    """Upper-left element of the 2x2 signal model of ``QSVT.kernel``.

    On an eigenstate with scaled eigenvalue ``x = lambda / alpha`` the
    good-subspace block of the circuit multiplies the state by this
    number. One forward walk step on the invariant 2D subspace is
    ``reflect_about_zero * block_encoding = diag(-1, 1) @ [[x, s], [s, -x]]``
    with ``s = sqrt(1 - x^2)`` (the walk's ``-H/alpha`` sign is folded in),
    and each qsvt-convention phase is ``diag(exp(i phi), 1)``.
    """
    s = math.sqrt(max(0.0, 1.0 - x * x))
    step = np.array([[-x, -s], [s, -x]], dtype=np.complex128)
    matrix = np.diag([np.exp(1.0j * sequence.phases[0]), 1.0])
    for phi in sequence.phases[1:]:
        matrix = np.diag([np.exp(1.0j * phi), 1.0]) @ step @ matrix
    return complex(matrix[0, 0])


# ----------------------------------------------------------------------
# Simulation helpers: the good-subspace block of each protocol kernel.
# ----------------------------------------------------------------------


def check(label: str, condition: bool):
    print(f"  [check] {label} ... {'OK' if condition else 'FAILED'}")
    if not condition:
        sys.exit(1)


def encoded_block(encoding, ket: np.ndarray) -> np.ndarray:
    state = cudaq.get_state(encoding.encode_kernel(), state_from(ket))
    return sim.good_subspace(encoding, state)


def controlled_blocks(encoding, ket: np.ndarray):
    """Good-subspace output of controlled_apply at control |0> and |1>."""
    controlled = encoding.controlled_apply_kernel()
    n_anc = encoding.num_ancilla
    dim = 1 << encoding.num_system

    @cudaq.kernel
    def circuit(state: cudaq.State, flip: bool):
        system = cudaq.qvector(state)
        control_and_ancilla = cudaq.qvector(1 + n_anc)
        if flip:
            x(control_and_ancilla[0])
        controlled(control_and_ancilla, system)
        if flip:
            x(control_and_ancilla[0])

    return [
        np.array(cudaq.get_state(circuit, state_from(ket), flip))[:dim]
        for flip in (False, True)
    ]


def controlled_roundtrips(encoding, ket: np.ndarray):
    """Controlled walk step followed by its controlled adjoint, at |0>, |1>."""
    step = encoding.controlled_walk_step_kernel()
    adjoint = encoding.controlled_adjoint_walk_step_kernel()
    n_anc = encoding.num_ancilla
    dim = 1 << encoding.num_system

    @cudaq.kernel
    def circuit(state: cudaq.State, flip: bool):
        system = cudaq.qvector(state)
        control_and_ancilla = cudaq.qvector(1 + n_anc)
        if flip:
            x(control_and_ancilla[0])
        step(control_and_ancilla, system)
        adjoint(control_and_ancilla, system)
        if flip:
            x(control_and_ancilla[0])

    return [
        np.array(cudaq.get_state(circuit, state_from(ket), flip))[:dim]
        for flip in (False, True)
    ]


def max_diff(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


# ----------------------------------------------------------------------
# The story
# ----------------------------------------------------------------------


def main() -> int:
    cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lx", type=int, default=2)
    parser.add_argument("--ly", type=int, default=2)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--u", type=float, default=4.0)
    parser.add_argument("--max-qubits",
                        type=int,
                        default=18,
                        help="skip the statevector checks above this many "
                        "qubits and report alphas only")
    args = parser.parse_args()
    lx, ly, t, u = args.lx, args.ly, args.t, args.u

    n = lx * ly
    num_qubits = 2 * n
    print(f"=== {lx}x{ly} Fermi-Hubbard (t = {t}, u = {u}): "
          f"{n} sites -> {num_qubits} system qubits ===")

    # 1. The analytic momentum basis: orthonormal, and it diagonalizes the
    #    hopping matrix with the closed-form dispersion.
    one_particle = hopping_matrix(lx, ly, t)
    modes, energies, labels = momentum_modes(lx, ly, t)
    check("momentum modes are orthonormal",
          max_diff(modes.T @ modes, np.eye(n)) < 1e-12)
    check("momentum modes diagonalize the hopping matrix with eps(k)",
          max_diff(modes.T @ one_particle @ modes, np.diag(energies)) < 1e-12)
    print("  dispersion, one line per mode (both spins share it):")
    for label, eps in zip(labels, energies):
        print(f"    {label:20s} eps = {eps:+.6f}")

    # 2. T diagonal in momentum space, rotated into the lattice frame; U
    #    diagonal in real space; the two summed.
    diagonal_t = PauliLCU(momentum_diagonal_words(energies, num_qubits))
    hopping = RotatedEncoding.from_orbital_rotation(diagonal_t, modes)
    interaction = PauliLCU(interaction_words(n, u, num_qubits))
    combined = SumEncoding([(1.0, hopping), (1.0, interaction)])
    check("the combined encoding satisfies the BlockEncoding protocol",
          isinstance(combined, BlockEncoding))

    # 3. The alpha table against a flat Pauli LCU of the same Hamiltonian.
    flat = PauliLCU(flat_hubbard_words(lx, ly, t, u))
    print(f"\n  alpha_T (momentum space) = {hopping.alpha:8.4f}  "
          f"({hopping.num_ancilla} ancillas)")
    print(f"  alpha_U (real space)     = {interaction.alpha:8.4f}  "
          f"({interaction.num_ancilla} ancillas)")
    print(f"  alpha (combined)         = {combined.alpha:8.4f}  "
          f"(= alpha_T + alpha_U; {combined.num_ancilla} ancillas: "
          f"{combined.num_slots} slot + flag + shared pool)")
    print(f"  alpha (flat PauliLCU)    = {flat.alpha:8.4f}  "
          f"({flat.num_terms} Pauli words, {flat.num_ancilla} ancillas)")
    check("combined alpha is below the flat PauliLCU alpha", combined.alpha
          < flat.alpha - 1e-12)

    total_qubits = num_qubits + combined.num_ancilla
    if total_qubits > args.max_qubits:
        print(f"\n  Statevector checks skipped: {total_qubits} qubits "
              f"> --max-qubits {args.max_qubits}.")
        return 0
    print(f"\n  Circuits: {num_qubits} system + {combined.num_ancilla} "
          f"ancilla = {total_qubits} qubits")

    # 4. Every block against the sparse Jordan-Wigner reference.
    hopping_ref, interaction_ref = reference_hamiltonians(lx, ly, t, u)
    hubbard = (hopping_ref + interaction_ref).tocsr()
    rng = np.random.default_rng(7)
    ket = rng.normal(
        size=1 << num_qubits) + 1.0j * rng.normal(size=1 << num_qubits)
    ket = (ket / np.linalg.norm(ket)).astype(np.complex128)

    error = max_diff(encoded_block(hopping, ket),
                     hopping_ref @ ket / hopping.alpha)
    check(f"T block == T/alpha_T (max |diff| = {error:.2e})", error < 1e-10)
    error = max_diff(encoded_block(interaction, ket),
                     interaction_ref @ ket / interaction.alpha)
    check(f"U block == U/alpha_U (max |diff| = {error:.2e})", error < 1e-10)
    error = max_diff(encoded_block(combined, ket),
                     hubbard @ ket / combined.alpha)
    check(f"combined block == (T + U)/alpha (max |diff| = {error:.2e})", error
          < 1e-10)

    # Control conventions: identity at |0>, the block at |1>; a controlled
    # walk step and its controlled adjoint round-trip at either value.
    at_zero, at_one = controlled_blocks(combined, ket)
    check("controlled apply is the identity at control |0>",
          max_diff(at_zero, ket) < 1e-10)
    check("controlled apply reproduces the block at control |1>",
          max_diff(at_one, hubbard @ ket / combined.alpha) < 1e-10)
    for value, state in enumerate(controlled_roundtrips(combined, ket)):
        check(
            f"controlled walk roundtrip is the identity at control |{value}>",
            max_diff(state, ket) < 1e-10)

    # 5. The combined object drops into Walk unchanged: even Chebyshev
    #    moments (the odd-moment observable hook is not available for a
    #    sum, see SumEncoding.select_observable).
    print("\n  Even Chebyshev moments <T_k(H/alpha)> through Walk:")
    scaled = hubbard / combined.alpha
    walk = Walk(combined)
    for order in (0, 2, 4):
        measured = walk.moment(ket, order)
        exact = chebyshev_moment(scaled, ket, order)
        print(
            f"    T_{order}: measured {measured:+.10f}   exact {exact:+.10f}")
        check(f"T_{order} moment matches the reference",
              abs(measured - exact) < 1e-8)

    # 6. A negative coefficient: shift the spectrum by the ground energy.
    #    SumEncoding([(1, H), (-E_ref, I)]) encodes (H - E_ref I) with
    #    alpha = alpha_H + |E_ref|; the identity is a one-term PauliLCU.
    dense = hubbard.toarray()
    e_ref = float(np.linalg.eigvalsh(dense)[0])
    identity = PauliLCU({"I" * num_qubits: 1.0})
    shifted = SumEncoding([(1.0, combined), (-e_ref, identity)])
    print(f"\n  Shifted H - E_ref I with E_ref = {e_ref:+.6f}: "
          f"alpha = {shifted.alpha:.4f} (= {combined.alpha:.4f} + "
          f"{abs(e_ref):.4f}), {shifted.num_ancilla} ancillas")
    error = max_diff(encoded_block(shifted, ket),
                     (hubbard @ ket - e_ref * ket) / shifted.alpha)
    check(f"shifted block == (H - E_ref I)/alpha (max |diff| = {error:.2e})",
          error < 1e-10)

    # 7. QSVT on the combined encoding: a degree-2 phase sequence against
    #    the dense signal model, eigenvalue by eigenvalue.
    sequence = PhaseSequence([0.3, -0.4, 0.25])
    device = sim.transform(QSVT(combined), ket, sequence)
    eigenvalues, eigenvectors = np.linalg.eigh(dense / combined.alpha)
    response = np.array(
        [qsvt_response(sequence, float(x)) for x in eigenvalues])
    host = eigenvectors @ (response * (eigenvectors.conj().T @ ket))
    error = max_diff(device, host)
    check(
        f"QSVT degree-{sequence.degree} sequence matches the signal model "
        f"(max |diff| = {error:.2e})", error < 1e-10)

    print("\nOK — T encoded in momentum space and U in real space, summed "
          "into one BlockEncoding that Walk and QSVT consume unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
